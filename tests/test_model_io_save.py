import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME,
)


def test_save_writes_files_and_config(tiny_timm_learner, tmp_path):
    save_varkoder_model(
        tiny_timm_learner, tmp_path, architecture="resnet18", is_multilabel=False,
    )
    weights = tmp_path / MODEL_WEIGHTS_FILENAME
    config = tmp_path / MODEL_CONFIG_FILENAME
    assert weights.exists() and config.exists()

    cfg = json.loads(config.read_text())
    assert cfg["architecture"] == "resnet18"
    assert cfg["is_multilabel"] is False
    assert cfg["label_names"] == list(tiny_timm_learner.dls.vocab)
    assert cfg["num_classes"] == len(tiny_timm_learner.dls.vocab)
    assert len(cfg["input_size"]) == 3
    assert "normalize" in cfg  # None here (fixture is pretrained=False)

    saved = load_file(str(weights))
    ref = tiny_timm_learner.model.state_dict()
    assert set(saved.keys()) == set(ref.keys())
    for k in ref:
        assert saved[k].dtype == torch.float32


def test_save_captures_normalization(tiny_timm_learner, tmp_path):
    from fastai.vision.all import Normalize
    tiny_timm_learner.dls.add_tfms(
        [Normalize.from_stats([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], cuda=False)],
        "after_batch")
    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    assert cfg["normalize"] is not None
    assert len(cfg["normalize"]["mean"]) == 3 and len(cfg["normalize"]["std"]) == 3
