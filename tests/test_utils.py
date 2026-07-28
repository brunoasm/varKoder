import pytest

from varKoder.core.utils import (
    format_bp_human_readable,
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
