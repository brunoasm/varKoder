import argparse

import numpy as np
from PIL import Image

from varKoder.commands.convert import ConvertCommand


def test_collect_image_files_keeps_arbitrarily_named_file_with_cli_overrides(tmp_path):
    # A file that does NOT parse as a varKoder name -- convert supports
    # remapping it anyway when the caller supplies explicit overrides.
    foreign = tmp_path / "photo.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype="uint8")).save(foreign)

    cmd = ConvertCommand.__new__(ConvertCommand)
    cmd.args = argparse.Namespace(
        input=str(tmp_path),
        outdir=str(tmp_path / "out"),
        input_mapping="cgr",
        kmer_size=7,
        output_mapping="varKode",
    )

    files = cmd._collect_image_files()

    assert len(files) == 1
    meta = files[0]
    assert meta["path"] == foreign
    assert meta["sample"] is None
    assert meta["img_kmer_mapping"] == "cgr"
    assert meta["img_kmer_size"] == 7
