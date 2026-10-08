"""Frozen-LM token error and direct output-head gradient proxies, pool only."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np

from omniselect.utils.hashing import file_sha256


def token_error_gradient_scores(logits, hidden, targets, *, head_has_bias=False):
    """RMS ||p_t-onehot(y_t)|| and RMS token output-head parameter-gradient norm.

    Each token's direct linear-head gradient has norm
    ||p_t-onehot(y_t)|| sqrt(||h_t||² + 1[bias]). We aggregate squared norms
    across tokens, avoiding a length preference and a vocabulary-by-hidden
    gradient matrix. Cross-token cancellation and backbone/embedding gradients
    are not included. This is not the gradient norm of the document mean loss.
    """
    import torch

    if logits.ndim != 2 or hidden.ndim != 2 or targets.ndim != 1:
        raise ValueError("token arrays must have shapes (T,V), (T,D), (T,)")
    if not len(targets) or len(hidden) != len(targets) or len(logits) != len(targets):
        raise ValueError("a scored record must contain at least one prediction token")
    p = torch.softmax(logits.float(), dim=-1)
    truth = p.gather(1, targets.long()[:, None]).squeeze(1)
    error2 = (p.square().sum(dim=-1) - 2 * truth + 1).clamp_min(0)
    phi2 = hidden.float().square().sum(dim=-1) + int(head_has_bias)
    result = torch.stack((error2.mean().sqrt(), (error2 * phi2).mean().sqrt()))
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite token adaptation score")
    return result


def frozen_text_scores(lm, records, tcfg, pool_path):
    """Content-bound atomic cache; never reads validation or test records."""
    identity = {
        "version": "token-rms-head-gradient-v1", "pool_sha256": file_sha256(pool_path),
        "ids_sha256": hashlib.sha256(json.dumps([r.id for r in records]).encode()).hexdigest(),
        "model": lm.model_name, "revision": lm.revision, "ctx": tcfg.ctx,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = os.path.join(tcfg.data_root, "processed", f"text_score_adapt_{key}.npz")
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as archive:
            scores = {name: np.asarray(archive[name], dtype=float) for name in ("el2n", "grand")}
    else:
        import torch

        model = lm.model().eval()
        head = model.get_output_embeddings()
        values = []
        try:
            with torch.no_grad():
                for record in records:
                    ids = lm.tok(record.text, truncation=True, max_length=tcfg.ctx,
                                 return_tensors="pt").to(lm.device)["input_ids"]
                    if ids.shape[1] < 2:
                        raise ValueError(f"record {record.id} has no next-token target")
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
                    values.append(token_error_gradient_scores(
                        head(hidden), hidden, ids[0, 1:], head_has_bias=getattr(head, "bias", None) is not None
                    ).cpu().numpy())
        finally:
            model.to("cpu")
            del head, model
        array = np.asarray(values)
        scores = {"el2n": array[:, 0], "grand": array[:, 1]}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz")
        os.close(fd)
        try:
            np.savez(temporary, **scores)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if any(v.shape != (len(records),) or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("invalid cached text adaptation scores")
    return {**scores, "info": {
        "score_source": "fixed pretrained LM on pool only", **identity,
        "el2n_definition": "RMS token probability-error L2 norm",
        "grand_definition": "RMS direct token output-head gradient norm; excludes cross-token and backbone terms",
        "tied_weights": "only the direct output-head contribution is included",
        "cache_path": path,
    }}
