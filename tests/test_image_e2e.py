import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from varKoder.cli import setup_parser
from varKoder.commands.image import run_image_command
from varKoder.core.utils import get_kmer_mapping, get_metadata_from_img_filename

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tiny_reads"

REQUIRED_TOOLS = ["dsk", "dsk2ascii", "fastp", "reformat.sh"]
_MISSING_TOOLS = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_TOOLS),
        reason=f"missing external tool(s) required by `image`: {_MISSING_TOOLS}",
    ),
]


def _run_image(tmp_path, extra_args=()):
    outdir = tmp_path / "images"
    int_dir = tmp_path / "intermediate"
    parser = setup_parser()
    args = parser.parse_args([
        "image", str(FIXTURE_ROOT),
        "-o", str(outdir),
        "-k", "5",
        "-p", "cgr",
        "-m", "1K",
        "-M", "10K",
        "-i", str(int_dir),
        "-f", str(tmp_path / "stats.csv"),  # else defaults to ./stats.csv in cwd
        "-x",
        "-R", "0",  # deterministic subsampling
        *extra_args,
    ])
    run_image_command(args, np.random.default_rng(args.seed))
    return outdir


def test_image_produces_expected_dimensions_and_parseable_name(tmp_path):
    outdir = _run_image(tmp_path)

    out_path = outdir / "sample1@04800+cgr+k5.png"
    assert out_path.is_file()

    expected_side = int(get_kmer_mapping(5, "cgr")["x"].max() + 1)
    with Image.open(out_path) as img:
        assert img.size == (expected_side, expected_side)

    meta = get_metadata_from_img_filename(out_path)
    assert meta["sample"] == "sample1"
    assert meta["bp"] == 4800
    assert meta["img_kmer_mapping"] == "cgr"
    assert meta["img_kmer_size"] == 5
    assert meta["multiframe"] is False


def test_image_stack_produces_multiframe_apng_with_metadata(tmp_path):
    outdir = _run_image(tmp_path, extra_args=["-S"])

    out_path = outdir / "sample1@stack+cgr+k5.apng"
    assert out_path.is_file()

    expected_side = int(get_kmer_mapping(5, "cgr")["x"].max() + 1)
    with Image.open(out_path) as img:
        assert img.size == (expected_side, expected_side)
        assert img.n_frames == 3
        assert img.info.get("varkoderFrameSizes") == "4800,2000,1000"
        assert img.info.get("varkoderFormatVersion") == "2"
        assert img.info.get("varkoderKeywords") == "TestTaxon"

    meta = get_metadata_from_img_filename(out_path)
    assert meta["sample"] == "sample1"
    assert meta["bp"] is None
    assert meta["multiframe"] is True

    # Largest-bp-first: frame sizes are already in descending order.
    frame_sizes = [int(x) for x in img.info["varkoderFrameSizes"].split(",")]
    assert frame_sizes == sorted(frame_sizes, reverse=True)
