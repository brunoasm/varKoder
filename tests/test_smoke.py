import torch


def test_fixtures_build(tiny_timm_learner, synthetic_images):
    df, labels = synthetic_images
    dl = tiny_timm_learner.dls.test_dl(df)
    preds, _ = tiny_timm_learner.get_preds(dl=dl)
    assert preds.shape == (len(df), len(labels))
    assert torch.isfinite(preds).all()
