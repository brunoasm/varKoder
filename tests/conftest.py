import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter,
    vision_learner, Resize, ResizeMethod,
)
from fastai.losses import CrossEntropyLossFlat


@pytest.fixture
def synthetic_images(tmp_path):
    """Six 64x64 RGB PNGs across three single labels; 4 train / 2 valid."""
    rng = np.random.default_rng(0)
    labels = ["alpha", "beta", "gamma"]
    rows = []
    for i in range(6):
        arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append({"path": str(p), "labels": labels[i % 3], "is_valid": i >= 4})
    return pd.DataFrame(rows), labels


@pytest.fixture
def tiny_timm_learner(synthetic_images):
    """A CPU resnet18 vision_learner with random weights (no download)."""
    df, labels = synthetic_images
    dbl = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=labels)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels"),
        item_tfms=Resize(64, method=ResizeMethod.Squish),
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    learn = vision_learner(
        dls, "resnet18", pretrained=False, normalize=True,
        loss_func=CrossEntropyLossFlat(),
    )
    return learn
