import torch
from varKoder.models.custom import instantiate_custom_model


def test_fiannaca_materializes_and_forwards():
    m = instantiate_custom_model("fiannaca2018", num_classes=5, input_size=(3, 64, 64))
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.shape == (2, 5)
    # No uninitialized lazy params remain
    for p in m.parameters():
        assert not isinstance(p, torch.nn.parameter.UninitializedParameter)


def test_arias_materializes_and_forwards():
    m = instantiate_custom_model("arias2022", num_classes=3, input_size=(3, 64, 64))
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 3)


def test_unknown_arch_raises():
    import pytest
    with pytest.raises(Exception):
        instantiate_custom_model("nope", 2, (3, 64, 64))
