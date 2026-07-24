"""Weights-only (safetensors) serialization for varKoder models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import save_file, load_file
from fastai.vision.all import Normalize, vision_learner, Learner
from fastai.losses import CrossEntropyLossFlat

from varKoder.core.config import CUSTOM_ARCHS
from varKoder.core.preprocessing import make_dataloaders
from varKoder.models.custom import instantiate_custom_model

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


def _dummy_df(label_names, input_size, tmpdir):
    """Two placeholder rows (one train, one valid) so a DataLoaders can be built
    without the original training data. Vocab is pinned separately."""
    c, h, w = input_size
    rows = []
    for i in range(2):
        arr = (np.zeros((h, w, 3))).astype("uint8")
        p = Path(tmpdir) / f"_dummy_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append({"path": str(p), "labels": label_names[0], "is_valid": i == 1})
    return pd.DataFrame(rows)


def build_learner(config, device="cpu"):
    import tempfile
    architecture = config["architecture"]
    label_names = config["label_names"]
    is_multilabel = config["is_multilabel"]
    input_size = tuple(config["input_size"])

    with tempfile.TemporaryDirectory() as tmp:
        df = _dummy_df(label_names, input_size, tmp)
        dls = make_dataloaders(df, architecture, is_multilabel, bs=2,
                               device=device, num_workers=0, vocab=label_names)

        if architecture in CUSTOM_ARCHS:
            model = instantiate_custom_model(architecture, len(label_names), input_size)
            learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
        else:
            # pretrained=False so no weights download; normalization is NOT added
            # by fastai in this mode, so we reapply it from config below.
            learn = vision_learner(dls, architecture, pretrained=False,
                                   normalize=False, loss_func=CrossEntropyLossFlat())

        norm = config.get("normalize")
        if norm is not None:
            # from_stats(cuda=True) puts stats on the DEFAULT device via to_device;
            # Normalize.encodes does (x - mean) with no device coercion, so stats
            # must sit on THIS learner's device. Build on CPU, then move explicitly.
            norm_tfm = Normalize.from_stats(norm["mean"], norm["std"], cuda=False)
            norm_tfm.mean = norm_tfm.mean.to(device)
            norm_tfm.std = norm_tfm.std.to(device)
            learn.dls.add_tfms([norm_tfm], "after_batch")
    learn.model = learn.model.to(device)
    return learn
