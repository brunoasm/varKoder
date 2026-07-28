#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image loading and frame-selection for multi-frame (APNG) varKode images.

This module defines the fastai image type and augmentation transform used to train
and query on multi-frame varKode images, where a single file holds one frame per
input-size (bp amount) of a sample.

Both classes are defined at module level (never as closures) because ``learn.export()``
pickles them by module + qualname; they must be importable under the exact same path at
train and query time.

Conventions
-----------
* Frames are stored largest-bp-first, so **frame 0 is the representative frame**
  (the largest amount of data). This is also what a legacy reader
  (``Image.open(...).convert("RGB")``) sees, so old models/readers degrade gracefully.
* Legacy single-frame PNGs have ``n_frames == 1``; the random-frame transform is a
  no-op on them, so their behavior is identical to before this feature existed.
"""

import random

from PIL import Image
from fastai.vision.core import PILImage
from fastai.vision.augment import RandTransform


class VarKodeImage(PILImage):
    """
    A varKode image that may hold multiple frames (one per input size).

    Loading inherits ``PILImage``'s behavior: it opens the file at frame 0 (the
    representative, largest-bp frame) and converts to RGB (fastai's
    ``PILBase._open_args = {'mode': 'RGB'}``), matching the 3-channel input the model
    expects. ``create`` is overridden only to stash the source path and frame count so
    ``SelectRandomFrame`` can re-seek to a random frame during training.
    """

    @classmethod
    def create(cls, fn, **kwargs):
        res = super().create(fn, **kwargs)  # frame 0, converted to RGB
        res.varkoder_path = str(fn)
        try:
            with Image.open(fn) as im:
                res.varkoder_n_frames = getattr(im, "n_frames", 1)
        except Exception:
            res.varkoder_n_frames = 1
        return res


class SelectRandomFrame(RandTransform):
    """
    Training-only transform that picks a random frame from a multi-frame varKode image.

    Runs on the training set only (``split_idx = 0``), so validation and query
    (``test_dl``, which uses validation semantics) always see the representative frame 0.
    It re-opens the file per call — rather than seeking on a shared handle — so it is safe
    under ``num_workers > 0``. For single-frame images (legacy PNGs) it is a no-op.
    """

    split_idx = 0

    def encodes(self, x: VarKodeImage):
        n = getattr(x, "varkoder_n_frames", 1)
        if n <= 1:
            return x
        idx = random.randrange(n)
        with Image.open(x.varkoder_path) as im:
            im.seek(idx)
            frame = im.convert("RGB")  # match the 3-channel input the model expects
        res = VarKodeImage(frame)
        res.varkoder_path = x.varkoder_path
        res.varkoder_n_frames = n
        return res
