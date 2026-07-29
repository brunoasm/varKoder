import argparse
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
