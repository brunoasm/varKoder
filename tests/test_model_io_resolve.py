import pytest
from varKoder.core.model_io import save_varkoder_model, resolve_model


def test_resolve_local_dir(tiny_timm_learner, tmp_path):
    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    state, cfg = resolve_model(str(tmp_path))
    assert cfg["architecture"] == "resnet18"
    assert set(state.keys()) == set(tiny_timm_learner.model.state_dict().keys())


def test_resolve_legacy_pkl_warns(tiny_timm_learner, tmp_path):
    pkl = tmp_path / "trained_model.pkl"
    tiny_timm_learner.export(pkl)
    with pytest.warns(UserWarning):
        state, cfg = resolve_model(str(pkl))
    assert "architecture" in cfg and "label_names" in cfg
