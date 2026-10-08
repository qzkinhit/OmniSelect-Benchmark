"""Influence channels: reference-model loss for text and probe log-likelihood for labelled features.

``InfluenceSignal`` returns the min-max normalized per-record causal-LM loss of ``model_name`` when
the model loads, and a unique-token ratio otherwise. ``probe_log_likelihood`` fits a logistic probe
on a reference sample and returns log p(observed label) per record (-20 for labels the probe has
not seen), the probe probabilities and the probe classes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord
from omniselect.utils.logger import get_logger
from omniselect.core.signals.base import Signal, minmax

log = get_logger("signals.influence")


class InfluenceSignal(Signal):
    """Per-record causal-LM loss of a reference model, with a model-free fallback when it cannot load."""

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
        """Min-max normalized loss per record, or the unique-token ratio without a loadable model."""
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

        mps = torch.backends.mps.is_available()
        device = self.device or ("cuda" if torch.cuda.is_available() else ("mps" if mps else "cpu"))
        tok = AutoTokenizer.from_pretrained(self.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device).eval()

        losses = np.zeros(len(records), dtype=float)
        texts = [r.text for r in records]
        with torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                batch = texts[s : s + self.batch_size]
                enc = tok(batch, return_tensors="pt", truncation=True, max_length=self.max_length,
                          padding=True).to(device)
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


def within_class_rank(score: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Percentile rank of ``score`` inside each observed class, in [0, 1] (0.5 for a singleton class).

    Log-likelihoods of a reference probe are not comparable across classes, because the probe is more
    confident on some classes than on others. Ranking inside the observed class keeps the class shares of
    a top-k selection close to those of the pool while it still orders the records of a class by how well
    the clean reference supports their label.
    """
    score = np.asarray(score, dtype=float); labels = np.asarray(labels)
    out = np.full(len(score), 0.5)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) > 1:
            out[idx] = np.argsort(np.argsort(score[idx], kind="stable"), kind="stable") / (len(idx) - 1)
    return out


def probe_log_likelihood(
    ref_features: np.ndarray,
    ref_labels: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Influence of each record as log p(observed label) under a probe fit on a reference sample.

    A LogisticRegression(max_iter, C) is fit on (ref_features, ref_labels). A record whose label
    the probe has not seen gets -20.0. Returns (influence, predict_proba on ``features``,
    probe classes). This is the influence channel of the vision, TEP and tabular runners.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, C=C).fit(ref_features, ref_labels)
    proba = clf.predict_proba(features)
    classes = clf.classes_
    column = {c: j for j, c in enumerate(classes)}
    influence = np.array([
        np.log(proba[i, column[labels[i]]] + 1e-9) if labels[i] in column else -20.0
        for i in range(len(labels))
    ])
    return influence, proba, classes
