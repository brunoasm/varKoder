"""Weights-only (safetensors) serialization for varKoder models."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from fastai.vision.all import Normalize

MODEL_WEIGHTS_FILENAME = "varkoder_model.safetensors"
MODEL_CONFIG_FILENAME = "config.json"


def _extract_normalize(dls):
    """Return {'mean':[...], 'std':[...]} for a Normalize in the batch pipeline,
    or None. fastai stores mean/std as (1,C,1,1) tensors."""
    for t in dls.after_batch.fs:
        if isinstance(t, Normalize):
            return {"mean": t.mean.flatten().tolist(),
                    "std": t.std.flatten().tolist()}
    return None


def recover_architecture(learn):
    """Return the base architecture name of a learner (timm or custom)."""
    from varKoder.models.custom import Fiannaca2018Model, Arias2022Model
    if isinstance(learn.model, Fiannaca2018Model):
        return "fiannaca2018"
    if isinstance(learn.model, Arias2022Model):
        return "arias2022"
    return learn.model[0].model.default_cfg["architecture"]


def _fp32_contiguous_state_dict(model):
    return {k: v.detach().to(torch.float32).contiguous()
            for k, v in model.state_dict().items()}


def save_varkoder_model(learn, outdir, *, architecture, is_multilabel):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xb, _ = learn.dls.one_batch()
    input_size = list(xb.shape[1:])  # (C, H, W)
    label_names = list(learn.dls.vocab)

    state_dict = _fp32_contiguous_state_dict(learn.model)
    save_file(state_dict, str(outdir / MODEL_WEIGHTS_FILENAME))

    config = {
        "architecture": architecture,
        "label_names": label_names,
        "is_multilabel": bool(is_multilabel),
        "num_classes": len(label_names),
        "input_size": input_size,
        "normalize": _extract_normalize(learn.dls),
    }
    (outdir / MODEL_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
