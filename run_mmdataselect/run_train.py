"""Application-layer entry: fine-tune a downstream base model on a selected subset.

Wraps HuggingFace Trainer for a small causal-LM SFT/continued-training run on the
``selected.jsonl`` produced by run_select. Torch/transformers are imported lazily;
without the ``train`` extra this exits with a clear message (code 3 = skipped) so
the pipeline can still be demonstrated CPU-only.

    python run_mmdataselect/run_train.py --config configs/experiments/<exp>.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from mmdataselect.utils.io import ensure_dir, read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402

log = get_logger("run_train")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a base model on the selected subset.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", exp_id)
    sel_path = os.path.join(out_dir, "manifests", "selected.jsonl")
    if not os.path.exists(sel_path):
        log.error("no selected.jsonl at %s — run run_select first", sel_path)
        return 2

    try:
        import torch  # noqa: F401
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception:
        log.warning("transformers/torch not installed; install with: pip install -e \".[train]\"")
        print("TRAIN SKIPPED | missing 'train' extra")
        return 3

    tcfg = cfg.get("train", {})
    model_name = tcfg.get("model_name", "HuggingFaceTB/SmolLM2-135M")
    max_length = int(tcfg.get("max_length", 512))
    epochs = float(tcfg.get("epochs", 1))
    batch_size = int(tcfg.get("batch_size", 4))
    lr = float(tcfg.get("learning_rate", 2e-5))

    rows = list(read_jsonl(sel_path))
    texts = [r.get("text", "") for r in rows]
    log.info("training %s on %d selected records", model_name, len(texts))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import torch
    from torch.utils.data import Dataset

    class JsonlDS(Dataset):
        def __init__(self, texts):
            self.enc = [
                tok(t, truncation=True, max_length=max_length, padding="max_length")
                for t in texts
            ]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            item = {k: torch.tensor(v) for k, v in self.enc[i].items()}
            return item

    model = AutoModelForCausalLM.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    ckpt_dir = ensure_dir(os.path.join(out_dir, "models"))
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=JsonlDS(texts), data_collator=collator)
    trainer.train()
    trainer.save_model(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    log.info("saved checkpoint -> %s", ckpt_dir)
    print(f"TRAIN OK | {model_name} -> {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
