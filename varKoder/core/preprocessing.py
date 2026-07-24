"""Shared fastai DataLoaders construction, used by training and by
weights-only inference so preprocessing is identical in both paths."""

from PIL.Image import Resampling
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, MultiCategoryBlock, ColReader,
    ColSplitter, RandomSplitter, Resize, ResizeMethod, aug_transforms,
    RandomErasing, default_device,
)
from timm import create_model

from varKoder.core.config import CUSTOM_ARCHS


def make_dataloaders(df, architecture, is_multilabel, *, bs, device="cpu",
                     num_workers=0, max_lighting=0, p_lighting=0,
                     random_erasing=False, vocab=None):
    item_transforms = None
    if architecture not in CUSTOM_ARCHS:
        default_cfg = create_model(architecture, pretrained=False).default_cfg
        if default_cfg.get("fixed_input_size"):
            item_transforms = Resize(
                size=default_cfg["input_size"][1:],
                method=ResizeMethod.Squish,
                resamples=(Resampling.BOX, Resampling.BOX),
            )

    transforms = aug_transforms(
        do_flip=False, max_rotate=0, max_zoom=1, max_lighting=max_lighting,
        max_warp=0, p_affine=0, p_lighting=p_lighting,
    )
    if random_erasing:
        transforms.append(RandomErasing())

    if is_multilabel:
        cat_block = MultiCategoryBlock(vocab=vocab) if vocab is not None else MultiCategoryBlock
        blocks = (ImageBlock, cat_block)
        get_y = ColReader("labels", label_delim=";")
    else:
        # CategoryBlock defaults to sort=True even when a vocab is supplied,
        # which would silently re-sort a pinned vocab alphabetically. Passing
        # sort=False preserves the exact order given in `vocab`.
        cat_block = CategoryBlock(vocab=vocab, sort=False) if vocab is not None else CategoryBlock
        blocks = (ImageBlock, cat_block)
        get_y = ColReader("labels")

    splitter = ColSplitter() if "is_valid" in df.columns else RandomSplitter()

    dbl = DataBlock(
        blocks=blocks, splitter=splitter, get_x=ColReader("path"), get_y=get_y,
        item_tfms=item_transforms, batch_tfms=transforms,
    )
    return dbl.dataloaders(df, bs=bs, device=device, num_workers=num_workers)
