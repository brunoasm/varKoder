import numpy as np
import pandas as pd
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
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


MULTIFRAME_LEVELS = [20, 130, 240]


@pytest.fixture
def multiframe_images(tmp_path):
    """Four 3-frame APNGs; frame k is a constant gray level MULTIFRAME_LEVELS[k].

    Frame 0 is the representative (largest-bp) frame, matching what
    ``varKoder image --stack`` writes. 3 train / 1 valid. Filenames and tEXt
    metadata follow the real varKoder convention so the fixture doubles as
    valid ``convert`` input.
    """
    labels = ["alpha", "beta"]
    frame_bps = [3000000, 1000000, 300000]  # descending: frame 0 = largest bp
    rows = []
    for i in range(4):
        frames = [
            Image.fromarray(np.full((64, 64), lv, dtype="uint8")).convert("RGB")
            for lv in MULTIFRAME_LEVELS
        ]
        label = labels[i % 2]
        # k=6 is deliberate: real cgr k-mer mapping is 2**k per side (verified
        # via get_kmer_mapping(6, "cgr")["x"].max()+1 == 64), matching this
        # fixture's existing 64x64 arrays exactly -- so a real `convert`
        # remap (Task 9) can run on these images with no dimension mismatch.
        p = tmp_path / f"sample{i}@stack+cgr+k6.apng"
        info = PngInfo()
        info.add_text("varkoderKeywords", label)
        info.add_text("varkoderMapping", "cgr")
        info.add_text("varkoderLowQualityFlag", "False")
        info.add_text("varkoderBaseFreqSd", "0.01")
        info.add_text("varkoderFrameSizes", ",".join(str(bp) for bp in frame_bps))
        info.add_text("varkoderFormatVersion", "2")
        frames[0].save(p, save_all=True, append_images=frames[1:], format="PNG", pnginfo=info)
        rows.append({"path": str(p), "labels": label, "is_valid": i >= 3})
    return pd.DataFrame(rows), labels, MULTIFRAME_LEVELS


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
