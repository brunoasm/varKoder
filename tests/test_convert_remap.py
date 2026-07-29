import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from varKoder.commands.convert import run_convert_command
from varKoder.core.utils import get_kmer_mapping


def test_single_frame_remap_size_and_metadata(tmp_path):
    k = 5
    in_side = int(get_kmer_mapping(k, "cgr")["x"].max() + 1)
    out_side = int(get_kmer_mapping(k, "varKode")["x"].max() + 1)

    indir = tmp_path / "in"
    indir.mkdir()
    outdir = tmp_path / "out"

    src = indir / f"sample1@00500K+cgr+k{k}.png"
    info = PngInfo()
    info.add_text("varkoderKeywords", "alpha")
    info.add_text("varkoderMapping", "cgr")
    info.add_text("varkoderLowQualityFlag", "False")
    arr = np.random.default_rng(0).integers(0, 255, (in_side, in_side), dtype="uint8")
    Image.fromarray(arr).save(src, pnginfo=info)

    args = argparse.Namespace(
        input=str(indir), outdir=str(outdir), output_mapping="varKode",
        input_mapping=None, kmer_size=None, n_threads=1,
        sum_reverse_complements=False, overwrite=True,
    )
    run_convert_command(args)

    out_path = outdir / f"sample1@00500K+varKode+k{k}.png"
    assert out_path.is_file()

    with Image.open(out_path) as img:
        assert img.size == (out_side, out_side)
        assert img.info.get("varkoderMapping") == "varKode"
        assert img.info.get("varkoderKeywords") == "alpha"


def test_stack_remap_preserves_frame_count_order_and_naming(tmp_path, multiframe_images):
    df, labels, levels = multiframe_images

    indir = tmp_path / "in"
    indir.mkdir()
    outdir = tmp_path / "out"

    # Move just one fixture-generated sample into its own input directory.
    src = Path(df["path"].iloc[0])
    dest = indir / src.name
    shutil.move(str(src), dest)

    args = argparse.Namespace(
        input=str(indir), outdir=str(outdir), output_mapping="varKode",
        input_mapping=None, kmer_size=None, n_threads=1,
        sum_reverse_complements=False, overwrite=True,
    )
    run_convert_command(args)

    out_name = dest.name.replace("+cgr+", "+varKode+")
    out_path = outdir / out_name
    assert out_path.is_file()

    with Image.open(out_path) as img:
        assert getattr(img, "n_frames", 1) == len(levels)
        assert img.info.get("varkoderFrameSizes") == "3000000,1000000,300000"
        assert img.info.get("varkoderMapping") == "varKode"

        frame_means = []
        for i in range(img.n_frames):
            img.seek(i)
            frame_means.append(np.array(img.convert("L")).mean())

    # Frame order preserved: MULTIFRAME_LEVELS is [20, 130, 240] (ascending),
    # so the representative frame 0 is the *darkest*, not the brightest --
    # brightness has no inherent meaning here, it's just a per-frame constant
    # the fixture assigns in list order. The invariant remap must preserve is
    # that order, so the output means stay ascending too.
    assert frame_means == sorted(frame_means)
