import json
import torch
from types import SimpleNamespace
from varKoder.core.model_io import save_varkoder_model
from varKoder.commands.query import QueryCommand


def test_query_loads_local_dir(tiny_timm_learner, synthetic_images, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    save_varkoder_model(tiny_timm_learner, model_dir,
                        architecture="resnet18", is_multilabel=False)

    args = SimpleNamespace(model=str(model_dir), max_batch_size=2)
    qc = QueryCommand.__new__(QueryCommand)  # bypass __init__
    qc.args = args
    qc.images_d = tmp_path  # no images needed for load_model path
    monkeypatch.setattr(
        "varKoder.commands.query.torch.backends.mps.is_built", lambda: False)
    learn = qc.load_model(n_images=1)
    assert learn.dls.vocab is not None
    assert qc.is_multilabel is False
