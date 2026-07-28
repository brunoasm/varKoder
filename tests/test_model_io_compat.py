"""Cross-version model compatibility for weights-only loading.

A user upgrading to the safetensors release still has models produced by older
varKoder versions, and those are pickled fastai learners whose data pipeline is
baked into the pickle. These tests cover the pipeline vintages that exist in the
wild, plus re-exporting such a model to safetensors.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter,
    Learner, Resize, ResizeMethod, vision_learner,
)

from varKoder.core.imaging import VarKodeImage, SelectRandomFrame
from varKoder.core.model_io import resolve_model, build_learner, save_varkoder_model
from varKoder.models.custom import instantiate_custom_model


def _legacy_learner(df, labels, *, image_cls, with_frame_tfm, size=64):
    """A learner whose pipeline mimics a given varKoder vintage.

    image_cls=PILImage / with_frame_tfm=False  -> varKoder <= 1.6
    image_cls=VarKodeImage / with_frame_tfm=True -> varKoder 1.7.x (multiframe)
    """
    item_tfms = [SelectRandomFrame()] if with_frame_tfm else []
    item_tfms.append(Resize(size, method=ResizeMethod.Squish))
    dbl = DataBlock(
        blocks=(ImageBlock(cls=image_cls), CategoryBlock(vocab=labels, sort=False)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels"),
        item_tfms=item_tfms,
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    return vision_learner(dls, "resnet18", pretrained=False, normalize=True,
                          loss_func=CrossEntropyLossFlat())


def test_resolve_pkl_from_multiframe_pipeline(synthetic_images, tmp_path):
    """A 1.7.x-trained .pkl pickles VarKodeImage into its pipeline. Loading it
    must work: this is what every user who trained on 1.7.0/1.7.1 has."""
    df, labels = synthetic_images
    learn = _legacy_learner(df, labels, image_cls=VarKodeImage, with_frame_tfm=True)
    pkl = tmp_path / "trained_model.pkl"
    learn.export(pkl)

    with pytest.warns(UserWarning):
        state, cfg = resolve_model(str(pkl))
    assert cfg["label_names"] == labels
    assert cfg["input_size"] == [3, 64, 64]
    assert set(state.keys()) == set(learn.model.state_dict().keys())


def test_resolve_pkl_from_prehistoric_pipeline(synthetic_images, tmp_path):
    """A <=1.6-trained .pkl has a plain PILImage pipeline and must still load."""
    from fastai.vision.core import PILImage
    df, labels = synthetic_images
    learn = _legacy_learner(df, labels, image_cls=PILImage, with_frame_tfm=False)
    pkl = tmp_path / "trained_model.pkl"
    learn.export(pkl)

    with pytest.warns(UserWarning):
        state, cfg = resolve_model(str(pkl))
    assert cfg["label_names"] == labels
    assert cfg["input_size"] == [3, 64, 64]


def test_reexport_pkl_to_safetensors(synthetic_images, tmp_path):
    """Re-publishing an existing .pkl as safetensors (what xtra_scripts/push_to_hf.py
    does) must work on a learner loaded from disk, whose datasets export() emptied."""
    from fastai.learner import load_learner
    df, labels = synthetic_images
    learn = _legacy_learner(df, labels, image_cls=VarKodeImage, with_frame_tfm=True)
    pkl = tmp_path / "trained_model.pkl"
    learn.export(pkl)

    reloaded = load_learner(pkl, cpu=True)
    outdir = tmp_path / "weights_only"
    save_varkoder_model(reloaded, outdir, architecture="resnet18", is_multilabel=False)

    state, cfg = resolve_model(str(outdir))
    assert cfg["input_size"] == [3, 64, 64]
    assert cfg["label_names"] == labels


def test_custom_arch_pkl_roundtrips_through_weights_only(tmp_path):
    """Custom archs use LazyLinear and their pipeline has no Resize, so an
    exported .pkl does not record the trained resolution. Rebuilding must still
    work and predict identically: build_learner materializes the lazy layers
    from the weights when it is given the state_dict."""
    size = 32
    labels = ["alpha", "beta"]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        arr = (rng.random((size, size, 3)) * 255).astype("uint8")
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append({"path": str(p), "labels": labels[i % 2], "is_valid": i >= 3})
    df = pd.DataFrame(rows)

    dbl = DataBlock(
        blocks=(ImageBlock(cls=VarKodeImage), CategoryBlock(vocab=labels, sort=False)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels"),
        item_tfms=[SelectRandomFrame()],
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    model = instantiate_custom_model("fiannaca2018", len(labels), (1, size, size))
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())

    pkl = tmp_path / "custom.pkl"
    learn.export(pkl)

    with pytest.warns(UserWarning):
        state, cfg = resolve_model(str(pkl))

    # The trained resolution is genuinely not recoverable here...
    assert cfg["architecture"] == "fiannaca2018"
    # ...but the rebuild must still load the real weights strictly and agree
    # with the original model on real input.
    rebuilt = build_learner(cfg, device="cpu", state_dict=state)

    batch = torch.randn(2, 3, size, size)
    learn.model.eval()
    rebuilt.model.eval()
    with torch.no_grad():
        assert torch.allclose(learn.model(batch), rebuilt.model(batch), atol=1e-6)
