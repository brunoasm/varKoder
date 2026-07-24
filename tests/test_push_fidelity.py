import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, build_learner, resolve_model, MODEL_CONFIG_FILENAME,
)


def test_saved_model_matches_source(tiny_timm_learner, synthetic_images, tmp_path):
    df, _ = synthetic_images
    dl = tiny_timm_learner.dls.test_dl(df)
    baseline, _ = tiny_timm_learner.get_preds(dl=dl)

    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    state, cfg = resolve_model(str(tmp_path))
    learn2 = build_learner(cfg, device="cpu")
    learn2.model.load_state_dict(state, strict=True)
    after, _ = learn2.get_preds(dl=learn2.dls.test_dl(df))

    assert torch.allclose(baseline, after, atol=1e-5)
