import warnings
from varKoder.commands.train import export_trained_model  # new helper
from varKoder.core.model_io import MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME


def test_export_writes_both_and_warns(tiny_timm_learner, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_trained_model(
            tiny_timm_learner, tmp_path,
            architecture="resnet18", is_multilabel=False,
        )
    assert (tmp_path / "trained_model.pkl").exists()
    assert (tmp_path / MODEL_WEIGHTS_FILENAME).exists()
    assert (tmp_path / MODEL_CONFIG_FILENAME).exists()
    assert (tmp_path / "labels.txt").exists()
    # FutureWarning specifically: DeprecationWarning is hidden by Python's default
    # filters outside __main__, so it would never reach a user running varKoder train.
    assert any(issubclass(w.category, FutureWarning) for w in caught)
