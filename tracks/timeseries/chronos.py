"""chronos-bolt learner for the forecasting track: few-epoch fine-tuning on selected windows.

``train_chronos`` deep-copies the pinned chronos-bolt model (tiny or small), fine-tunes it with
AdamW (lr from the track config, batch 64, gradient norm clipped at ``chronos_grad_clip``) for at
most ``epochs`` epochs capped by ``chronos_max_epochs``. With validation windows (V_con) it
evaluates the quantile loss after every epoch, keeps the state with the lowest loss and stops after
``chronos_patience`` epochs without improvement. It returns a callable that maps input windows to
the median quantile forecast. Each pipeline is loaded once per process. Missing weights raise
SkipCell so the cell is skipped with a message.
"""
from __future__ import annotations

import copy
import hashlib

_PIPELINE: dict = {}


class ChronosForecaster:
    """Callable wrapper returning the median forecast of H steps."""

    def __init__(self, model, horizon: int):
        self.model, self.horizon = model, horizon

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        """Move the fitted weights without refitting or changing their dtype."""
        self.model.to(device)
        return self

    def __call__(self, x):
        import torch

        with torch.no_grad():
            out = self.model(context=x)
        q = out.quantile_preds
        return q[:, q.shape[1] // 2, : self.horizon]

    def state_dict(self):
        return self.model.state_dict()


def _pipeline(tcfg, device: str):
    """The pinned chronos-bolt pipeline of ``tcfg`` (loaded once per model, revision and device)."""
    import torch

    from tracks.common.experiment import SkipCell

    key = (tcfg.chronos_model, tcfg.chronos_revision, device)
    if key not in _PIPELINE:
        try:
            from chronos import BaseChronosPipeline

            _PIPELINE[key] = BaseChronosPipeline.from_pretrained(
                tcfg.chronos_model, revision=tcfg.chronos_revision, device_map=device, torch_dtype=torch.float32)
        except (ImportError, OSError) as exc:
            raise SkipCell(f"chronos-bolt unavailable ({tcfg.chronos_model}): {exc}") from exc
    return _PIPELINE[key]


def _loss(model, xb, yb, batch: int = 256) -> float:
    """Mean quantile loss of ``model`` over (xb, yb) in evaluation mode."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(xb), batch):
            loss = model(context=xb[s:s + batch], target=yb[s:s + batch]).loss
            total += float(loss) * len(xb[s:s + batch])
            count += len(xb[s:s + batch])
    model.train()
    return total / max(count, 1)


def train_chronos(X, Y, device: str, seed: int, epochs: int, tcfg, val=None) -> ChronosForecaster:
    """Fine-tune chronos-bolt on (X, Y) windows and return the forecaster (see the module docstring)."""
    import torch

    torch.manual_seed(seed)
    model = copy.deepcopy(_pipeline(tcfg, device).model).to(device).train()
    n_epochs = min(int(epochs), int(tcfg.chronos_max_epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.chronos_lr)
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    yb = torch.tensor(Y, dtype=torch.float32, device=device)
    clip = float(getattr(tcfg, "chronos_grad_clip", 0.0) or 0.0)
    patience = int(getattr(tcfg, "chronos_patience", 1))
    xv = yv = None
    if val is not None and len(val[0]):
        xv = torch.tensor(val[0], dtype=torch.float32, device=device)
        yv = torch.tensor(val[1], dtype=torch.float32, device=device)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    order = hashlib.sha256()
    history = []
    for epoch in range(n_epochs):
        perm = torch.randperm(len(xb), device=device)
        order.update(perm.detach().cpu().numpy().tobytes())
        for s in range(0, len(xb), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            loss = model(context=xb[idx], target=yb[idx]).loss
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if xv is not None:
            value = _loss(model, xv, yv)
            history.append(value)
            if value < best_loss - 1e-12:
                best_loss, best_epoch, stale = value, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    forecaster = ChronosForecaster(model.eval(), horizon=yb.shape[1])
    forecaster.train_order_sha12 = order.hexdigest()[:12]
    forecaster.early_stopping = {"val_loss": history, "best_epoch": best_epoch} if xv is not None else None
    return forecaster
