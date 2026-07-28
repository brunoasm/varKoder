from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from varKoder.core.utils import (
    format_bp_human_readable,
    get_metadata_from_img_filename,
    get_varKoder_frame_sizes,
    iter_varKoder_images,
    parse_bp_human_readable,
)


@pytest.mark.parametrize(
    "bp",
    [123, 999, 1000, 1898, 99999 + 1, 500000, 1868000, 10000000, 100000000, 5000000000],
)
def test_bp_human_readable_round_trip(bp):
    assert parse_bp_human_readable(format_bp_human_readable(bp)) == bp


def test_bp_human_readable_rounds_to_four_sig_figs():
    # format_bp_human_readable keeps only 4 significant digits, so a value
    # needing a 5th digit of precision rounds up rather than round-tripping.
    assert format_bp_human_readable(99999) == "00100K"
    assert parse_bp_human_readable("00100K") == 100000


def test_parse_bp_human_readable_legacy_eight_digit_format():
    # v0.x wrote an 8-digit zero-padded count before the 'K' suffix.
    assert parse_bp_human_readable("00001868K") == 1868000


def test_parse_bp_human_readable_rejects_garbage():
    with pytest.raises(ValueError):
        parse_bp_human_readable("not-a-size")


def test_get_metadata_current_single_frame_format():
    name = f"sample1@{format_bp_human_readable(500000)}+cgr+k7.png"
    meta = get_metadata_from_img_filename(Path(name))
    assert meta == {
        "sample": "sample1",
        "bp": 500000,
        "img_kmer_mapping": "cgr",
        "img_kmer_size": 7,
        "path": Path(name),
        "multiframe": False,
    }


def test_get_metadata_legacy_v0x_single_frame_format():
    # v0.x names omit the mapping segment entirely; mapping defaults to 'varKode'.
    name = f"sample1@{format_bp_human_readable(500000)}+k7.png"
    meta = get_metadata_from_img_filename(Path(name))
    assert meta["sample"] == "sample1"
    assert meta["bp"] == 500000
    assert meta["img_kmer_mapping"] == "varKode"
    assert meta["img_kmer_size"] == 7
    assert meta["multiframe"] is False


def test_get_metadata_current_multiframe_format():
    meta = get_metadata_from_img_filename(Path("sample1@stack+cgr+k7.apng"))
    assert meta == {
        "sample": "sample1",
        "bp": None,
        "img_kmer_mapping": "cgr",
        "img_kmer_size": 7,
        "path": Path("sample1@stack+cgr+k7.apng"),
        "multiframe": True,
    }


def test_get_metadata_legacy_multiframe_format():
    meta = get_metadata_from_img_filename(Path("sample1@stack+k7.apng"))
    assert meta["img_kmer_mapping"] == "varKode"
    assert meta["multiframe"] is True


@pytest.mark.parametrize(
    "name",
    [
        "sample1@00500K+cgr+k7 2.png",     # iCloud/Dropbox sync conflict copy
        "sample_no_at_sign+cgr+k7.png",     # missing '@' sample/bp separator
        "sample1@00500K+cgr+k.png",         # empty k-mer size field
        "sample1@notanumber+cgr+k7.png",    # non-numeric bp field
        "notavarkode.png",                  # no '@', no '+' at all
        "sample1@a@00500K+cgr+k7.png",      # extra '@'
    ],
)
def test_get_metadata_rejects_malformed_names(name):
    with pytest.raises(ValueError):
        get_metadata_from_img_filename(Path(name))


def _save_png_with_text(path, **text_items):
    info = PngInfo()
    for k, v in text_items.items():
        info.add_text(k, v)
    Image.fromarray(np.zeros((4, 4, 3), dtype="uint8")).save(path, pnginfo=info)


def test_get_varKoder_frame_sizes_valid(tmp_path):
    p = tmp_path / "valid.png"
    _save_png_with_text(p, varkoderFrameSizes="10,20,30")
    assert get_varKoder_frame_sizes(p) == [10, 20, 30]


def test_get_varKoder_frame_sizes_absent(tmp_path):
    p = tmp_path / "absent.png"
    _save_png_with_text(p)  # no varkoderFrameSizes chunk at all
    assert get_varKoder_frame_sizes(p) == []


def test_get_varKoder_frame_sizes_malformed(tmp_path, capsys):
    p = tmp_path / "malformed.png"
    _save_png_with_text(p, varkoderFrameSizes="10,abc,30")
    assert get_varKoder_frame_sizes(p) == []
    assert "could not parse varkoderFrameSizes" in capsys.readouterr().err


def test_iter_varKoder_images_finds_png_and_apng_only(tmp_path):
    (tmp_path / "a@00500K+cgr+k7.png").touch()
    (tmp_path / "b@stack+cgr+k7.apng").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "a@00500K+cgr+k7.png.bak").touch()

    found = {p.name for p in iter_varKoder_images(tmp_path)}
    assert found == {"a@00500K+cgr+k7.png", "b@stack+cgr+k7.apng"}


def test_iter_varKoder_images_skips_unparseable_names_by_default(tmp_path, capsys):
    good_name = f"good@{format_bp_human_readable(500000)}+cgr+k7.png"
    (tmp_path / good_name).touch()
    (tmp_path / "sample1@00500K+cgr+k7 2.png").touch()  # sync conflict copy
    (tmp_path / "notavarkode.png").touch()

    found = {p.name for p in iter_varKoder_images(tmp_path)}
    assert found == {good_name}

    err = capsys.readouterr().err
    assert "ignored 2 file(s)" in err
    assert "sample1@00500K+cgr+k7 2.png" in err
    assert "notavarkode.png" in err


def test_iter_varKoder_images_skip_unparseable_false_yields_everything(tmp_path, capsys):
    good_name = f"good@{format_bp_human_readable(500000)}+cgr+k7.png"
    bad_name = "notavarkode.png"
    (tmp_path / good_name).touch()
    (tmp_path / bad_name).touch()

    found = {p.name for p in iter_varKoder_images(tmp_path, skip_unparseable=False)}
    assert found == {good_name, bad_name}
    assert capsys.readouterr().err == ""
