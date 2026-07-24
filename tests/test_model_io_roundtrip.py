import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, build_learner, recover_architecture,
    MODEL_CONFIG_FILENAME, MODEL_WEIGHTS_FILENAME,
)


def _preds(learn, df):
    dl = learn.dls.test_dl(df)
    p, _ = learn.get_preds(dl=dl)
    return p


def test_timm_roundtrip_fidelity(tiny_timm_learner, synthetic_images, tmp_path):
    df, _ = synthetic_images
    baseline = _preds(tiny_timm_learner, df)

    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))

    learn2 = build_learner(cfg, device="cpu")
    assert list(learn2.dls.vocab) == cfg["label_names"]
    learn2.model.load_state_dict(state, strict=True)
    after = _preds(learn2, df)

    assert torch.allclose(baseline, after, atol=1e-5)


def test_normalized_model_roundtrip_fidelity(tiny_timm_learner, synthetic_images, tmp_path):
    """A model trained WITH normalization must still round-trip exactly. This is
    the case fastai's pretrained=False path would silently drop, so it guards
    that build_learner reapplies normalization from config."""
    from fastai.vision.all import Normalize
    learn = tiny_timm_learner
    # cuda=False keeps stats on CPU so this runs on any machine (mps/cuda
    # available would otherwise put stats off-device and mismatch the CPU batch).
    learn.dls.add_tfms(
        [Normalize.from_stats([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], cuda=False)],
        "after_batch")
    df, _ = synthetic_images
    baseline = _preds(learn, df)

    save_varkoder_model(learn, tmp_path, architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    assert cfg["normalize"] is not None
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))

    learn2 = build_learner(cfg, device="cpu")
    assert list(learn2.dls.vocab) == cfg["label_names"]
    learn2.model.load_state_dict(state, strict=True)
    after = _preds(learn2, df)

    assert torch.allclose(baseline, after, atol=1e-5)


def test_custom_roundtrip_builds(synthetic_images, tmp_path):
    # config for a custom arch; weights come from a freshly built learner
    from varKoder.core.preprocessing import make_dataloaders
    from fastai.vision.all import Learner
    from fastai.losses import CrossEntropyLossFlat
    from varKoder.models.custom import instantiate_custom_model

    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=2, vocab=labels)
    model = instantiate_custom_model("fiannaca2018", len(labels), (1, 64, 64))
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
    baseline = _preds(learn, df)

    save_varkoder_model(learn, tmp_path, architecture="fiannaca2018", is_multilabel=False)

    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))
    learn2 = build_learner(cfg, device="cpu")
    assert list(learn2.dls.vocab) == cfg["label_names"]
    learn2.model.load_state_dict(state, strict=True)  # must not raise
    after = _preds(learn2, df)

    assert torch.allclose(baseline, after, atol=1e-5)


def test_recover_architecture_custom_models(synthetic_images):
    """Locks recover_architecture's custom-arch branch: a learner whose .model
    is a Fiannaca2018Model/Arias2022Model must report the matching arch name,
    not fall through to the timm default_cfg lookup."""
    from varKoder.models.custom import instantiate_custom_model
    df, labels = synthetic_images

    fiannaca_model = instantiate_custom_model("fiannaca2018", len(labels), (1, 64, 64))
    arias_model = instantiate_custom_model("arias2022", len(labels), (1, 64, 64))

    class _StandInLearner:
        def __init__(self, model):
            self.model = model

    assert recover_architecture(_StandInLearner(fiannaca_model)) == "fiannaca2018"
    assert recover_architecture(_StandInLearner(arias_model)) == "arias2022"
