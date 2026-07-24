from varKoder.core.preprocessing import make_dataloaders


def test_vocab_is_pinned(synthetic_images):
    df, labels = synthetic_images
    pinned = ["gamma", "beta", "alpha"]  # deliberately not sorted
    dls = make_dataloaders(df, "resnet18", is_multilabel=False, bs=2, vocab=pinned)
    assert list(dls.vocab) == pinned


def test_custom_arch_has_no_resize(synthetic_images):
    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=2, vocab=labels)
    # custom archs use no item resize transform
    assert not any(type(t).__name__ == "Resize" for t in dls.after_item.fs)
