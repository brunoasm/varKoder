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
from varKoder.core.imaging import VarKodeImage, SelectRandomFrame
from varKoder.core.utils import eprint


def make_dataloaders(df, architecture, is_multilabel, *, bs, device="cpu",
                     num_workers=0, max_lighting=0, p_lighting=0,
                     random_erasing=False, vocab=None, valid_pct=0.2,
                     verbose=False):
    # Randomly draw one frame per multi-frame sample each epoch (training only; a no-op
    # for legacy single-frame images). Must run before any Resize/ToTensor so a frame is
    # chosen while the item is still a VarKodeImage. SelectRandomFrame has split_idx=0,
    # so validation and inference (test_dl, which uses validation semantics) always see
    # the representative frame 0.
    item_transforms = [SelectRandomFrame()]
    if architecture not in CUSTOM_ARCHS:
        default_cfg = create_model(architecture, pretrained=False).default_cfg
        if default_cfg.get("fixed_input_size"):
            item_transforms.append(Resize(
                size=default_cfg["input_size"][1:],
                method=ResizeMethod.Squish,
                resamples=(Resampling.BOX, Resampling.BOX),
            ))
            if verbose:
                eprint(
                    "Model architecture",
                    architecture,
                    "requires image resizing to",
                    str(default_cfg["input_size"][1:]),
                )
                eprint("This will be done automatically.")

    transforms = aug_transforms(
        do_flip=False, max_rotate=0, max_zoom=1, max_lighting=max_lighting,
        max_warp=0, p_affine=0, p_lighting=p_lighting,
    )
    if random_erasing:
        transforms.append(RandomErasing())

    if is_multilabel:
        cat_block = MultiCategoryBlock(vocab=vocab) if vocab is not None else MultiCategoryBlock
        blocks = (ImageBlock(cls=VarKodeImage), cat_block)
        get_y = ColReader("labels", label_delim=";")
    else:
        # CategoryBlock defaults to sort=True even when a vocab is supplied,
        # which would silently re-sort a pinned vocab alphabetically. Passing
        # sort=False preserves the exact order given in `vocab`.
        cat_block = CategoryBlock(vocab=vocab, sort=False) if vocab is not None else CategoryBlock
        blocks = (ImageBlock(cls=VarKodeImage), cat_block)
        get_y = ColReader("labels")

    splitter = ColSplitter() if "is_valid" in df.columns else RandomSplitter(valid_pct=valid_pct)

    dbl = DataBlock(
        blocks=blocks, splitter=splitter, get_x=ColReader("path"), get_y=get_y,
        item_tfms=item_transforms, batch_tfms=transforms,
    )
    return dbl.dataloaders(df, bs=bs, device=device, num_workers=num_workers)
