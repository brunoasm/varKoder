"""Which weights `varKoder train` starts from, given --architecture /
--pretrained-model / --random-weights.

These are pure decision tests: resolve_pretrained_source() returns the source to
fine-tune from, or None to build --architecture from scratch. Nothing is
downloaded or trained here, which is the point -- the bug this guards against
was a bare `train --architecture resnet18` silently downloading and fine-tuning
the published vision transformer instead.
"""

import pytest

from varKoder.commands.train import resolve_pretrained_source
from varKoder.core.config import DEFAULT_ARCHITECTURE, DEFAULT_MODEL


def test_bare_invocation_fine_tunes_the_default_model():
    assert resolve_pretrained_source(
        DEFAULT_ARCHITECTURE, DEFAULT_MODEL, False
    ) == DEFAULT_MODEL


@pytest.mark.parametrize("architecture", ["resnet18", "arias2022", "fiannaca2018"])
def test_requested_architecture_beats_the_default_model(architecture, capsys):
    assert resolve_pretrained_source(architecture, DEFAULT_MODEL, False) is None
    # And says so, rather than silently changing what gets trained.
    assert architecture in capsys.readouterr().err


def test_requested_model_is_used_when_architecture_is_left_alone():
    assert resolve_pretrained_source(
        DEFAULT_ARCHITECTURE, "some/repo", False
    ) == "some/repo"


def test_requesting_both_is_an_error():
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_pretrained_source("resnet18", "some/repo", False)


@pytest.mark.parametrize("pretrained_model", ["none", "None", "NONE", None, ""])
def test_none_opts_out_of_any_pretrained_model(pretrained_model):
    assert resolve_pretrained_source("resnet18", pretrained_model, False) is None
    assert resolve_pretrained_source(DEFAULT_ARCHITECTURE, pretrained_model, False) is None


def test_none_plus_architecture_is_not_a_conflict():
    # Both are "requested", but --pretrained-model none is a request for *no*
    # pretrained model, so there is nothing for --architecture to conflict with.
    assert resolve_pretrained_source("resnet18", "none", False) is None


def test_random_weights_opts_out_even_with_the_default_model():
    assert resolve_pretrained_source(DEFAULT_ARCHITECTURE, DEFAULT_MODEL, True) is None


def test_random_weights_still_reports_a_conflict():
    # --random-weights says how to initialize, not which of two mutually
    # exclusive architecture sources to believe; that is still a mistake.
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_pretrained_source("resnet18", "some/repo", True)
