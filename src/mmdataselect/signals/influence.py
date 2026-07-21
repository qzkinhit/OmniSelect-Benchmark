"""Influence signal — downstream-model-aligned value of a record (PDS/LESS style).

The real path uses the downstream base model's own per-sample loss / perplexity as a
learnability proxy (higher loss = more the model can still learn = higher influence).
Torch/transformers are imported lazily; without them (or the ``train`` extra) the
signal transparently falls back to a deterministic, model-free CPU proxy so the
end-to-end selection loop stays runnable offline.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..datatypes import UnifiedRecord
from ..utils.logger import get_logger
from .base import Signal, minmax

log = get_logger("signals.influence")


class InfluenceSignal(Signal):
    name = "influence"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def score(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        if self.model_name:
            try:
                return self._model_loss(records)
            except Exception as e:  # noqa: BLE001 - any failure -> safe CPU proxy
                log.warning("influence: model path unavailable (%s); using CPU proxy", e)
        return self._cpu_proxy(records)

    # ---- model-free fallback (deterministic, runs anywhere) ----
    def _cpu_proxy(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        """Lexical-richness learnability proxy = unique/total token ratio.

        A crude but deterministic stand-in for downstream loss, used only when no
        model is available (sanity smoke / CPU-only runs).
        """
        vals = []
        for r in records:
            toks = (r.text or "").split()
            vals.append(len(set(toks)) / max(1, len(toks)))
        return minmax(vals)

    # ---- real path: downstream base model per-sample loss ----
    def _model_loss(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        tok = AutoTokenizer.from_pretrained(self.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device).eval()

        losses = np.zeros(len(records), dtype=float)
        texts = [r.text for r in records]
        with torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                batch = texts[s : s + self.batch_size]
                enc = tok(batch, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True).to(device)
                logits = model(**enc).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = enc["input_ids"][:, 1:].contiguous()
                mask = enc["attention_mask"][:, 1:].contiguous()
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.transpose(1, 2), shift_labels, reduction="none"
                )
                per = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                losses[s : s + len(batch)] = per.float().cpu().numpy()
        # higher per-sample loss = more learnable = higher influence
        return minmax(losses)
