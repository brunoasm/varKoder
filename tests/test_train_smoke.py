"""1-epoch training smoke tests for a custom architecture.

`-m none` is passed explicitly so these tests state, rather than rely on,
that no Hugging Face download happens. (Passing `-c fiannaca2018` alone
already opts out of the default pretrained model -- see
tests/test_train_pretrained_source.py -- but a training test should not be
where that regresses into a download.)

The other constraint: `--mix-augmentation None` combined with a custom
architecture crashes in fastai's callback wiring (`Learner(cbs=None)`), and
MixUp itself crashes on a batch size of 1. So `-B 2` (`--min-batch-size 2`)
is passed to keep the batch size above 1, while `--mix-augmentation` is left
at its default.
"""

import json
from unittest.mock import patch

import fastai.vision.all as fastai_vision
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
        "-B", "2",  # avoid a batch size of 1 (see module docstring)
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

    # Mock fit_one_cycle during resumed training to verify it requests only the
    # remaining epoch (1) rather than the full target (2), proving resume doesn't
    # silently retrain from scratch off the checkpointed weights.
    captured_calls = []
    original_fit_one_cycle = fastai_vision.Learner.fit_one_cycle

    def mock_fit_one_cycle(self, n_epoch, *args, **kwargs):
        captured_calls.append(n_epoch)
        return original_fit_one_cycle(self, n_epoch, *args, **kwargs)

    with patch.object(fastai_vision.Learner, "fit_one_cycle", mock_fit_one_cycle):
        run_train_command(_train_args(indir, outdir, epochs=2, resume=True))

    # The unfrozen phase should be called with n_epoch=1 (the remaining epoch to train)
    # Frozen phase is skipped (unfrozen_done=1, so remaining_frozen=0, phase="unfrozen")
    assert len(captured_calls) == 1, f"Expected one fit_one_cycle call, but got {len(captured_calls)}: {captured_calls}"
    assert captured_calls[0] == 1, f"Expected fit_one_cycle called with n_epoch=1 (remaining), but got n_epoch={captured_calls[0]}"

    progress_after = json.loads((checkpoint_dir / "progress.json").read_text())
    assert progress_after["unfrozen_done"] == 2
