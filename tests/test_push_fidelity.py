import pytest
import torch
from varKoder.core.model_io import (
    save_varkoder_model, build_learner, resolve_model,
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


@pytest.mark.parametrize("multilabel", [True, False])
def test_push_script_fidelity_gate_accepts_varkoder_losses(synthetic_images, tmp_path, multilabel):
    """The gate in xtra_scripts/push_to_hf.py must pass for a faithful conversion
    of a model trained the way varKoder trains: AsymmetricLossMultiLabel for
    multilabel and torch's CrossEntropyLoss for single-label without MixUp.
    Neither loss defines an activation, while the rebuilt learner's losses do, so
    comparing get_preds() output directly compares logits with probabilities."""
    import sys
    from pathlib import Path
    from fastai.learner import load_learner
    from fastai.vision.all import vision_learner
    from timm.loss import AsymmetricLossMultiLabel
    from torch.nn import CrossEntropyLoss
    from varKoder.core.preprocessing import make_dataloaders

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xtra_scripts"))
    from push_to_hf import verify_fidelity

    df, _ = synthetic_images
    loss = (AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=4, eps=1e-2, clip=0.1)
            if multilabel else CrossEntropyLoss())
    dls = make_dataloaders(df, "resnet18", multilabel, bs=2, device="cpu")
    learn = vision_learner(dls, "resnet18", pretrained=False, loss_func=loss)
    learn.export(tmp_path / "model.pkl")
    learn = load_learner(tmp_path / "model.pkl", cpu=True)

    out = tmp_path / "out"
    save_varkoder_model(learn, out, architecture="resnet18", is_multilabel=multilabel)
    image_dir = Path(df["path"].iloc[0]).parent
    assert verify_fidelity(learn, image_dir, out)
