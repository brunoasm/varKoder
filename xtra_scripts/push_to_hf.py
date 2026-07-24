#!/usr/bin/env python
"""Publish a varKoder model to Hugging Face as safetensors weights + config.json.

Loads a trusted local .pkl, exports the safetensors artifact, verifies that the
rebuilt model reproduces the pkl's predictions on sample images, and only then
uploads. Legacy files already in the repo are left untouched.
"""

import argparse
import tempfile
from pathlib import Path

import torch
from fastai.vision.all import load_learner
from huggingface_hub import HfApi

from varKoder.core.model_io import (
    save_varkoder_model, build_learner, resolve_model, recover_architecture,
    MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME,
)


def verify_fidelity(learn, sample_image_dir, out_dir, atol=1e-5):
    import pandas as pd
    imgs = [str(p) for p in Path(sample_image_dir).rglob("*.png")]
    if not imgs:
        raise SystemExit("No sample PNGs found for fidelity check.")
    df = pd.DataFrame({"path": imgs})
    baseline, _ = learn.get_preds(dl=learn.dls.test_dl(df))

    state, cfg = resolve_model(str(out_dir))
    rebuilt = build_learner(cfg, device="cpu")
    rebuilt.model.load_state_dict(state, strict=True)
    after, _ = rebuilt.get_preds(dl=rebuilt.dls.test_dl(df))
    return torch.allclose(baseline, after, atol=atol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Path to the trusted local .pkl")
    ap.add_argument("repo_id", help="Target Hugging Face repo id")
    ap.add_argument("sample_images", help="Dir of sample PNGs for the fidelity check")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for the fidelity check (default 1e-5; "
                         "loosen, e.g. to 1e-4, only if fp16 numeric noise causes a "
                         "spurious failure).")
    args = ap.parse_args()

    learn = load_learner(args.model_path, cpu=True)
    architecture = recover_architecture(learn)
    is_multilabel = "MultiLabel" in str(learn.loss_func)

    with tempfile.TemporaryDirectory() as out:
        save_varkoder_model(learn, out, architecture=architecture,
                            is_multilabel=is_multilabel)
        if not verify_fidelity(learn, args.sample_images, out, atol=args.atol):
            raise SystemExit("Fidelity check FAILED — not pushing.")
        api = HfApi()
        for fname in (MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME):
            api.upload_file(path_or_fileobj=str(Path(out) / fname),
                            path_in_repo=fname, repo_id=args.repo_id)
    print("Pushed", MODEL_WEIGHTS_FILENAME, "and", MODEL_CONFIG_FILENAME)


if __name__ == "__main__":
    main()
