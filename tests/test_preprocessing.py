import random

from varKoder.core.preprocessing import make_dataloaders
from varKoder.core.imaging import VarKodeImage, SelectRandomFrame


def _frame_levels(dl, epochs):
    """Mean pixel level of each single-image batch drawn over `epochs` passes.

    Frames are constant-valued, so a bs=1 batch mean identifies the frame.
    """
    seen = set()
    for _ in range(epochs):
        for xb, _yb in dl:
            seen.add(round(float(xb.mean()) * 255))
    return seen


def test_vocab_is_pinned(synthetic_images):
    df, labels = synthetic_images
    pinned = ["gamma", "beta", "alpha"]  # deliberately not sorted
    dls = make_dataloaders(df, "resnet18", is_multilabel=False, bs=2, vocab=pinned)
    assert list(dls.vocab) == pinned


def test_vocab_is_pinned_multilabel(synthetic_images):
    df, _ = synthetic_images
    # semicolon-joined multilabel column, reusing the same rows/paths
    df = df.copy()
    df["labels"] = ["alpha;beta", "beta;gamma", "alpha;gamma",
                    "alpha", "beta", "gamma"]
    pinned = ["gamma", "beta", "alpha"]  # deliberately not sorted
    dls = make_dataloaders(df, "resnet18", is_multilabel=True, bs=2, vocab=pinned)
    assert list(dls.vocab) == pinned


def test_custom_arch_has_no_resize(synthetic_images):
    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=2, vocab=labels)
    # custom archs use no item resize transform
    assert not any(type(t).__name__ == "Resize" for t in dls.after_item.fs)


def test_multiframe_images_load_as_varkode_image(multiframe_images):
    """The shared pipeline must use VarKodeImage, not plain PILImage: it is what
    stashes the frame count that SelectRandomFrame needs."""
    df, labels, levels = multiframe_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    img = dls.train.dataset[0][0]
    assert isinstance(img, VarKodeImage)
    assert img.varkoder_n_frames == len(levels)


def test_selectrandomframe_in_training_item_pipeline(multiframe_images):
    df, labels, _ = multiframe_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    assert any(isinstance(t, SelectRandomFrame) for t in dls.train.after_item.fs)


def test_training_draws_non_representative_frames(multiframe_images):
    """Regression guard: training on stacked (--stack) images must see every
    frame. A plain ImageBlock/PILImage pipeline always yields frame 0 and
    silently discards the rest of the sample's input sizes."""
    df, labels, levels = multiframe_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    random.seed(0)
    seen = _frame_levels(dls.train, epochs=40)
    assert seen == set(levels), f"frames drawn during training: {sorted(seen)}"


def test_validation_uses_representative_frame_only(multiframe_images):
    """SelectRandomFrame is split_idx=0, so validation -- and inference via
    test_dl, which uses validation semantics -- must always see frame 0."""
    df, labels, levels = multiframe_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    random.seed(0)
    assert _frame_levels(dls.valid, epochs=20) == {levels[0]}


def test_inference_test_dl_uses_representative_frame(multiframe_images):
    """The query path builds a test_dl from the same pipeline; it must be
    deterministic on the representative frame."""
    df, labels, levels = multiframe_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    random.seed(0)
    test_dl = dls.test_dl(df[["path"]])
    seen = set()
    for _ in range(20):
        for (xb,) in test_dl:
            seen.add(round(float(xb.mean()) * 255))
    assert seen == {levels[0]}


def test_single_frame_images_unaffected(synthetic_images):
    """Legacy single-frame PNGs must behave exactly as before: n_frames == 1
    makes SelectRandomFrame a no-op."""
    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=1, vocab=labels)
    img = dls.train.dataset[0][0]
    assert img.varkoder_n_frames == 1
