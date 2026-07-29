import argparse
import shutil
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

        frame_levels = []
        for i in range(img.n_frames):
            img.seek(i)
            arr = np.array(img.convert("L"))
            frame_levels.append(int(np.unique(arr[arr > 0])[0]))

    # Frame order preserved: each output frame is a pure pixel permutation of
    # its input frame (the k-mer coordinate remap), so it has exactly one
    # non-zero pixel value -- the original frame's constant gray level. Check
    # those levels against the fixture's actual per-frame levels in order,
    # rather than just checking monotonicity: a regression that wrote the
    # same frame three times (e.g. append_images=[frames[0]] * n) would still
    # produce a monotonic (constant) sequence of means, but would fail this
    # exact-identity check.
    assert frame_levels == levels
