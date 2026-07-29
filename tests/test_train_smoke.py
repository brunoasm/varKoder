import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from varKoder.cli import setup_parser
from varKoder.commands.train import run_train_command
from varKoder.core.model_io import MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME
from varKoder.core.utils import format_bp_human_readable

pytestmark = pytest.mark.slow


def _make_train_images(indir):
    rng = np.random.default_rng(0)
    labels = ["alpha", "beta"]
    for i in range(8):
        lab = labels[i % 2]
        arr = (rng.random((32, 32, 3)) * 255).astype("uint8")
        p = indir / f"sample{i}@{format_bp_human_readable(500000 + i)}+cgr+k5.png"
        info = PngInfo()
        info.add_text("varkoderKeywords", lab)
        info.add_text("varkoderLowQualityFlag", "False")
        Image.fromarray(arr).save(p, pnginfo=info)


def _train_args(indir, outdir, *, epochs, resume=False):
    parser = setup_parser()
    argv = [
        "train", str(indir), str(outdir),
        "-c", "fiannaca2018",
        "-m", "none",
        "-e", str(epochs),
        "-z", "0",
        "-C",  # force cpu
        "-g",  # no-logging
        "-M",  # no-metrics
        "-S",  # single-label (simpler)
        "-B", "2",  # avoid a batch size of 1 (see module docstring note)
    ]
    if resume:
        argv.append("-u")
    else:
        argv.append("-x")
    return parser.parse_args(argv)


def test_one_epoch_smoke_writes_expected_outputs(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_train_images(indir)
    outdir = tmp_path / "out"

    run_train_command(_train_args(indir, outdir, epochs=1))

    assert (outdir / "trained_model.pkl").exists()
    assert (outdir / MODEL_WEIGHTS_FILENAME).exists()
    assert (outdir / MODEL_CONFIG_FILENAME).exists()
    assert (outdir / "labels.txt").exists()

    checkpoint_dir = outdir / "checkpoints"
    assert (checkpoint_dir / "last.pth").exists()
    progress = json.loads((checkpoint_dir / "progress.json").read_text())
    assert progress == {
        "architecture": "fiannaca2018",
        "phase": "unfrozen",
        "frozen_done": 0,
        "unfrozen_done": 1,
    }


def test_resume_continues_training_and_advances_progress(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_train_images(indir)
    outdir = tmp_path / "out"

    run_train_command(_train_args(indir, outdir, epochs=1))

    checkpoint_dir = outdir / "checkpoints"
    progress_before = json.loads((checkpoint_dir / "progress.json").read_text())
    assert progress_before["unfrozen_done"] == 1

    run_train_command(_train_args(indir, outdir, epochs=2, resume=True))

    progress_after = json.loads((checkpoint_dir / "progress.json").read_text())
    assert progress_after["unfrozen_done"] == 2
