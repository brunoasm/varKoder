"""Weights-only (safetensors) serialization for varKoder models."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import save_file, load_file
from fastai.vision.all import Normalize, vision_learner, Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.learner import load_learner

from varKoder.core.config import CUSTOM_ARCHS
from varKoder.core.imaging import VarKodeImage
from varKoder.core.preprocessing import make_dataloaders
from varKoder.models.custom import instantiate_custom_model, new_custom_model

MODEL_WEIGHTS_FILENAME = "varkoder_model.safetensors"
MODEL_CONFIG_FILENAME = "config.json"

_PICKLE_WARNING = (
    "Loading a pickled (.pkl) model executes arbitrary code on load. This is "
    "deprecated and unsafe; re-export to safetensors with a recent varKoder."
)


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


def _probe_input_size(learn):
    """Return the post-transform input size (C, H, W) of a learner's pipeline.

    Prefers a real batch. A learner reloaded from a .pkl has no items --
    Learner.export() writes the pickle with empty train/valid datasets, so
    one_batch() raises -- and in that case one dummy image is routed through the
    same transform pipeline instead. The dummy is a VarKodeImage so that it
    satisfies the recorded input type of both the current pipeline and the older
    plain-PILImage one (VarKodeImage subclasses PILImage).
    """
    try:
        xb, _ = learn.dls.one_batch()
    except (ValueError, IndexError):
        dummy = VarKodeImage.create(np.zeros((8, 8, 3), dtype=np.uint8))
        xb = learn.dls.test_dl([dummy]).one_batch()[0]
    return list(xb.shape[1:])  # (C, H, W)


def save_varkoder_model(learn, outdir, *, architecture, is_multilabel):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_size = _probe_input_size(learn)
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


def build_learner(config, device="cpu", state_dict=None):
    """Rebuild a fastai Learner from a config, optionally loading weights.

    Pass `state_dict` whenever you have the weights: for the custom
    architectures it is the only reliable way to get their LazyLinear layers to
    the trained shapes, because an exported .pkl does not record the input
    resolution for them (their pipeline has no Resize to read it back from).
    Loading the state_dict materializes those layers directly from the weights.
    """
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
            if state_dict is not None:
                model = new_custom_model(architecture, len(label_names))
                model.load_state_dict(state_dict, strict=True)
            else:
                model = instantiate_custom_model(architecture, len(label_names), input_size)
            learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
        else:
            # pretrained=False so no weights download; normalization is NOT added
            # by fastai in this mode, so we reapply it from config below.
            learn = vision_learner(dls, architecture, pretrained=False,
                                   normalize=False, loss_func=CrossEntropyLossFlat())
            if state_dict is not None:
                learn.model.load_state_dict(state_dict, strict=True)

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


def _config_from_learner(learn, architecture=None):
    label_names = list(learn.dls.vocab)
    if architecture is None:
        architecture = recover_architecture(learn)
    # For architectures whose pipeline enforces a fixed size (timm archs with
    # default_cfg["fixed_input_size"], via the Resize added in make_dataloaders)
    # the probe recovers the true trained input_size. For resolution-flexible
    # architectures (no such Resize) the pipeline is a no-op on shape, so this
    # only reflects the dummy probe's size -- the true training resolution is
    # not recoverable from an exported learner alone. That is harmless for timm
    # archs, which pool adaptively; for the custom archs, pass the state_dict to
    # build_learner so their lazy layers materialize from the weights instead.
    return {
        "architecture": architecture,
        "label_names": label_names,
        "is_multilabel": "MultiLabel" in str(learn.loss_func),
        "num_classes": len(label_names),
        "input_size": _probe_input_size(learn),
        "normalize": _extract_normalize(learn.dls),
    }


def _read_local_dir(d):
    d = Path(d)
    config = json.loads((d / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(d / MODEL_WEIGHTS_FILENAME))
    return state, config


def resolve_model(source):
    """Resolve a model source to (state_dict, config).

    `source` may be: a local directory containing MODEL_WEIGHTS_FILENAME and
    MODEL_CONFIG_FILENAME (written by save_varkoder_model); a legacy .pkl
    file path (deprecated, unsafe -- emits UserWarning); or a Hugging Face
    repo id (downloads the two files, falling back to the legacy
    from_pretrained_fastai path, with the same warning, if they're absent).
    """
    p = Path(source)

    if p.is_dir():
        if (p / MODEL_WEIGHTS_FILENAME).exists() and (p / MODEL_CONFIG_FILENAME).exists():
            return _read_local_dir(p)
        raise ValueError(
            f"Model directory '{source}' must contain both "
            f"{MODEL_WEIGHTS_FILENAME} and {MODEL_CONFIG_FILENAME} (as written by "
            f"'varKoder train'). Point --model at such a directory, a .pkl file, "
            f"or a Hugging Face repo id."
        )

    if p.is_file() or str(source).endswith(".pkl"):
        if not p.is_file():
            raise ValueError(f"Model file '{source}' not found.")
        warnings.warn(_PICKLE_WARNING, UserWarning)
        learn = load_learner(source, cpu=True)
        return learn.model.state_dict(), _config_from_learner(learn)

    # Treat as a Hugging Face repo id.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        cfg_path = hf_hub_download(source, MODEL_CONFIG_FILENAME)
        wts_path = hf_hub_download(source, MODEL_WEIGHTS_FILENAME)
        config = json.loads(Path(cfg_path).read_text())
        state = load_file(wts_path)
        return state, config
    except EntryNotFoundError:
        # Repo exists but lacks the weights-only artifact: fall back to the
        # legacy pickled fastai model (deprecated, unsafe). This is the path the
        # current default model still takes, so give it its own error handling --
        # an exception raised in here is NOT caught by the sibling clause below.
        warnings.warn(_PICKLE_WARNING, UserWarning)
        try:
            from huggingface_hub import from_pretrained_fastai
            learn = from_pretrained_fastai(source)
            return learn.model.state_dict(), _config_from_learner(learn)
        except Exception as e:
            raise ValueError(
                f"Found a Hugging Face repo '{source}' but it has no "
                f"{MODEL_WEIGHTS_FILENAME} + {MODEL_CONFIG_FILENAME}, and its "
                f"legacy pickled model could not be loaded. "
                f"(underlying error: {type(e).__name__}: {e})"
            ) from e
    except Exception as e:
        raise ValueError(
            f"Unable to load model '{source}' as a local model directory "
            f"({MODEL_WEIGHTS_FILENAME} + {MODEL_CONFIG_FILENAME}), a .pkl file, "
            f"or a Hugging Face repo id. Please check the path or repo id. "
            f"(underlying error: {type(e).__name__}: {e})"
        ) from e
