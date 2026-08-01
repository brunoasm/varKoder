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


def test_resolve_dir_missing_config_is_clear(tiny_timm_learner, tmp_path):
    # A directory with the weights file but no config.json should raise a
    # clear, actionable error rather than a bare FileNotFoundError.
    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    (tmp_path / "config.json").unlink()
    with pytest.raises(ValueError) as exc:
        resolve_model(str(tmp_path))
    msg = str(exc.value)
    assert "config.json" in msg and str(tmp_path) in msg


def test_resolve_bogus_source_is_clear():
    # A source that is not a local dir, not a .pkl, and not a valid HF repo id
    # should raise a clear error (offline: invalid repo-id format is rejected
    # before any network call).
    with pytest.raises(ValueError) as exc:
        resolve_model("not a valid model source!!")
    assert "Unable to load model" in str(exc.value)
