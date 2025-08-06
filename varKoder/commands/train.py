#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train command module for varKoder.

This module contains functionality for training neural network models
on varKode images for DNA barcode classification.
"""

import os
import warnings
import torch
import pandas as pd
import numpy as np
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from math import log
import tempfile
import shutil

from fastai.vision.all import (
    aug_transforms, vision_learner, CategoryBlock, ImageBlock, 
    MultiCategoryBlock, DataBlock, ColReader, ColSplitter,
    Learner, cnn_learner, accuracy, error_rate, Resize, ResizeMethod
)
from fastai.callback.mixup import MixUp, CutMix
from fastai.torch_core import set_seed, default_device, defaults
from fastai.learner import load_learner
from fastai.losses import CrossEntropyLossFlat
from fastai.callback.core import Callback, CancelValidException
from fastai.metrics import accuracy, accuracy_multi, PrecisionMulti, RecallMulti, RocAuc
from fastai.distributed import to_parallel, detach_parallel

from torch.nn import CrossEntropyLoss, Module, Sequential, Linear, Flatten, LazyLinear, ReLU, Dropout, Conv1d, MaxPool1d
from timm.loss import AsymmetricLossMultiLabel
from timm import create_model

from varKoder.core.config import (
    LABELS_SEP, CUSTOM_ARCHS
)
from varKoder.core.utils import (
    eprint, get_metadata_from_img_filename, get_varKoder_labels, 
    get_varKoder_qual
)

from PIL.Image import Resampling

# Define classes for custom models
class Arias2022Head(Module):
    def __init__(self, n_classes):
        super(Arias2022Head, self).__init__()
        self.head = Sequential(Linear(64, n_classes))
    def forward(self, x):
        return self.head(x)

class Arias2022Body(Module):
    def __init__(self):
        super(Arias2022Body, self).__init__()
        self.body = Sequential(
                Flatten(), #reshape to 1D array
                LazyLinear(512),
                ReLU(),
                Dropout(0.5),
                Linear(512, 64),
                ReLU(),
                Dropout(0.5))

    def forward(self, x):
        x = x[:, 0, :, :] #keep only one channel
        x = self.body(x)
        return x

class Fiannaca2018Head(Module):
    def __init__(self,n_classes):
        super(Fiannaca2018Head, self).__init__()
        self.head = Sequential(Linear(500, n_classes))

    def forward(self, x):
        return self.head(x)

class Fiannaca2018Body(Module):
    def __init__(self):
        super(Fiannaca2018Body, self).__init__()
        self.flatten = Flatten()
        self.body = Sequential(
            Conv1d(1, 5, kernel_size=5),  # First convolutional layer
            ReLU(),
            MaxPool1d(kernel_size=2),  # Pooling layer

            Conv1d(5, 10, kernel_size=5),  # Second convolutional layer
            ReLU(),
            MaxPool1d(kernel_size=2),  # Pooling layer

            Flatten(),
            LazyLinear(500),  # Adjust the size based on the output of previous layers
            ReLU())

    def forward(self, x):
        x = x[:, 0, :, :] #keep only one channel
        x = self.flatten(x)
        x = x.unsqueeze(1)
        x = self.body(x)
        return x

class Fiannaca2018Model(Module):
    def __init__(self,n_classes):
        super(Fiannaca2018Model, self).__init__()
        self.model = Sequential(Fiannaca2018Body(),Fiannaca2018Head(n_classes))

    def forward(self, x):
        x = self.model(x)
        return x

class Arias2022Model(Module):
    def __init__(self,n_classes):
        super(Arias2022Model, self).__init__()
        self.model = Sequential(Arias2022Body(),Arias2022Head(n_classes))

    def forward(self, x):
        x = self.model(x)
        return x

def build_custom_model(architecture, dls):
    """
    Build a custom model architecture for training.
    
    Args:
        architecture: Model architecture name
        dls: DataLoaders object
        
    Returns:
        Custom model
    """
    if architecture == 'arias2022':
        custom_model = Arias2022Model(len(dls.vocab))
    elif architecture == 'fiannaca2018':
        custom_model = Fiannaca2018Model(len(dls.vocab))
    else:
        raise Exception('Custom models must be one of: fiannaca2018 arias2022')

    # Initialize LazyLinear with dummy batch
    xb, yb = dls.one_batch()
    input_image_size = xb.shape[-2:]  
    dummy_batch = torch.randn((1, 1, input_image_size[0], input_image_size[1]))  
    custom_model(dummy_batch)

    return custom_model

class SkipValidationCallback(Callback):
    """Callback to skip validation during training."""
    def before_validate(self):
        raise CancelValidException

def train_nn(
    df,
    architecture,
    valid_pct=0.2,
    max_bs=64,
    min_bs=1,
    base_lr=1e-3,
    model_state_dict=None,
    epochs=30,
    freeze_epochs=0,
    normalize=True,
    callbacks=None,
    max_lighting=0,
    p_lighting=0,
    pretrained=False,
    loss_fn=CrossEntropyLoss(),
    is_multilabel=False,
    metrics_threshold=0.7,
    gamma_neg=4,
    verbose=True,
    num_workers=0,
    no_metrics=False,
    force_cpu=False
):
    """
    Train a neural network model on varKode images.
    
    Args:
        df: DataFrame with image paths and labels
        architecture: Model architecture name
        valid_pct: Validation set percentage
        max_bs: Maximum batch size
        min_bs: Minimum batch size
        base_lr: Base learning rate
        model_state_dict: Pretrained model state dictionary
        epochs: Number of epochs to train
        freeze_epochs: Number of epochs to train with frozen layers
        normalize: Whether to normalize images
        callbacks: Training callbacks
        max_lighting: Maximum lighting augmentation
        p_lighting: Probability of lighting augmentation
        pretrained: Whether to use pretrained weights
        loss_fn: Loss function
        is_multilabel: Whether this is a multilabel classification task
        metrics_threshold: Threshold for multilabel metrics
        gamma_neg: Negative sample downweighting parameter
        verbose: Whether to show verbose output
        num_workers: Number of data loader workers
        no_metrics: Whether to skip metrics computation
        force_cpu: Whether to force CPU usage instead of GPU
        
    Returns:
        Trained model
    """
    # If forcing CPU usage, set FastAI defaults to ensure consistency
    if force_cpu:
        defaults.device = torch.device('cpu')
        # Disable GPU backends to prevent any GPU usage
        torch.backends.cuda.enabled = False
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'enabled'):
            torch.backends.mps.enabled = False
    
    # If skipping validation metrics, add NoValidation callback
    if no_metrics:
        if isinstance(callbacks, list):
            callbacks.append(SkipValidationCallback())
        else:
            callbacks = [callbacks, SkipValidationCallback()]

    # Find a batch size that is a power of 2 and splits the dataset in about 10 batches
    batch_size = 2 ** round(log(df[~df["is_valid"]].shape[0] / 10, 2))
    batch_size = min(batch_size, max_bs)
    batch_size = max(batch_size, min_bs)

    # Set kind of splitter for DataBlock
    if "is_valid" in df.columns:
        sptr = ColSplitter()
    else:
        sptr = RandomSplitter(valid_pct=valid_pct)

    # Check if item resizing is necessary
    item_transforms = None
    if architecture not in CUSTOM_ARCHS:
        default_cfg = create_model(architecture, pretrained=False).default_cfg
        if "fixed_input_size" in default_cfg.keys() and default_cfg["fixed_input_size"]:
            item_transforms = Resize(
                size=default_cfg["input_size"][1:],
                method=ResizeMethod.Squish,
                resamples=(Resampling.BOX, Resampling.BOX),
            )
            eprint(
                "Model architecture",
                architecture,
                "requires image resizing to",
                str(default_cfg["input_size"][1:]),
            )
            eprint("This will be done automatically.")
            

    # Set batch transforms
    transforms = aug_transforms(
        do_flip=False,
        max_rotate=0,
        max_zoom=1,
        max_lighting=max_lighting,
        max_warp=0,
        p_affine=0,
        p_lighting=p_lighting,
    )

    # Set DataBlock
    if is_multilabel:
        blocks = (ImageBlock, MultiCategoryBlock)
        get_y = ColReader("labels", label_delim=";")
    else:
        blocks = (ImageBlock, CategoryBlock)
        get_y = ColReader("labels")

    dbl = DataBlock(
        blocks=blocks,
        splitter=sptr,
        get_x=ColReader("path"),
        get_y=get_y,
        item_tfms=item_transforms,
        batch_tfms=transforms,
    )

    # Create data loaders with calculated batch size and appropriate device
    device = torch.device('cpu') if force_cpu else default_device()
    dls = dbl.dataloaders(df, 
        bs=batch_size, 
        device=device, 
        num_workers=num_workers)

    # Create learner
    if is_multilabel:
        # Find all labels that are not 'low_quality:True'
        labels = [i for i, x in enumerate(dls.vocab) if x != "low_quality:True"]
        # Define metrics
        precision = PrecisionMulti(labels=labels, average="micro", thresh=metrics_threshold)
        recall = RecallMulti(labels=labels, average="micro", thresh=metrics_threshold)
        auc = RocAuc(average="micro")
        metrics = [auc, precision, recall]
    else:
        metrics = accuracy

    if architecture in CUSTOM_ARCHS:
        # Build model
        custom_model = build_custom_model(architecture, dls)
        
        # Ensure custom model is on correct device
        if force_cpu:
            custom_model = custom_model.cpu()
        
        learn = Learner(dls, 
                        custom_model, 
                        metrics=metrics, 
                        cbs=callbacks,
                        loss_func=loss_fn
                       )
        
    else:
        learn = vision_learner(dls,
                               architecture,
                               metrics=metrics,
                               normalize=normalize,
                               pretrained=pretrained,
                               cbs=callbacks,
                               loss_func=loss_fn,
                            )
    
    # Only use FP16 if not forcing CPU (FP16 is typically GPU-only)
    if not force_cpu:
        learn = learn.to_fp16()
    
    # Ensure learner components are on correct device when forcing CPU
    if force_cpu:
        learn.model = learn.model.cpu()
        if hasattr(learn, 'dls'):
            learn.dls.device = device
   
    # If there a pretrained model body weights, replace them
    if model_state_dict:
        old_state_dict = learn.state_dict()
        new_state_dict = {
            k: v.to(device) if hasattr(v, 'to') else v  # Ensure tensors are on correct device
            for k, v in model_state_dict.items()
            if k in old_state_dict and old_state_dict[k].size() == v.size()
        }
        learn.model.load_state_dict(new_state_dict, strict=False)
        
        # Ensure model is on correct device after loading state dict
        if force_cpu:
            learn.model = learn.model.cpu()

    # Check for multiple GPUs and parallelize if available (unless CPU is forced)
    is_parallel = (not force_cpu and torch.backends.cuda.is_built() 
                  and torch.cuda.device_count() > 1)
    if is_parallel:
        learn.to_parallel()

    # Train the model with or without verbose output
    training_context = learn.no_bar() if not verbose else contextlib.nullcontext()
    logging_context = learn.no_logging() if not verbose else contextlib.nullcontext()

    with training_context, logging_context:
        learn.fine_tune(epochs=epochs, freeze_epochs=freeze_epochs, base_lr=base_lr)

    # Detach parallelization if it was used
    if is_parallel:
        learn.detach_parallel()

    # Remove skip validation callback if used
    learn.remove_cb(SkipValidationCallback)

    return learn

class TrainCommand:
    """
    Class for handling the train command functionality in varKoder.
    
    This class implements methods to train neural network models on varKode images.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize TrainCommand with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        
        # Check if output directory exists
        if not args.overwrite:
            if Path(args.outdir).exists():
                raise Exception(
                    "Output directory exists, use --overwrite if you want to overwrite it."
                )
    
    def collect_images(self) -> pd.DataFrame:
        """
        Collect and process image files for training.
        
        Returns:
            DataFrame with image information
        """
        eprint("Collecting image files for training...")
        
        # Collect all image files
        image_files = []
        f_counter = 0
        for f in Path(self.args.input).rglob("*.png"):
            image_files.append(get_metadata_from_img_filename(f))
            f_counter += 1
            if f_counter % 1000 == 0:
                eprint(f"\rFound {f_counter} image files", end='', flush=True)
        eprint(f"\rFound {f_counter} image files", flush=True)
        
        # If using label table
        if self.args.label_table_path:
            n_image_files = pd.DataFrame(image_files).merge(
                pd.read_csv(self.args.label_table_path)[
                     ["sample", "labels"]
                    #["sample", "labels", "possible_low_quality"]
                ],
                on="sample",
                how="inner",
            )
            excluded_samples = set([x["sample"] for x in image_files]) - set(n_image_files["sample"])
            eprint(len(excluded_samples),"samples excluded due to absence in provided label table.")
            if self.args.verbose:
                eprint('Samples excluded:\n','\n'.join(excluded_samples))
            image_files = n_image_files
        else:
            # Get labels from image metadata
            image_files = pd.DataFrame(image_files).assign(
                labels=lambda x: x["path"].apply(
                    lambda y: ";".join(get_varKoder_labels(y))
                ),
                possible_low_quality=lambda x: x["path"].apply(get_varKoder_qual),
            )
        
        return image_files
    
    def prepare_validation_split(self, image_files: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare validation split for training.
        
        Args:
            image_files: DataFrame with image information
            
        Returns:
            DataFrame with validation split information
        """
        # If a specific validation set was defined, let's use it
        if self.args.validation_set:
            eprint("Splitting validation set as defined by user.")
            try:  # Try to treat as a path first
                with open(self.args.validation_set, "r") as valsamps:
                    validation_samples = valsamps.readline().strip().split(",")
            except:  # If not, try to treat as a list
                validation_samples = self.args.validation_set.split(",")
        else:
            eprint(
                "Splitting validation set randomly. Fraction of samples per label combination held as validation:",
                str(self.args.validation_set_fraction),
            )
            validation_samples = (
                image_files[["sample", "labels"]]
                .assign(
                    labels=lambda x: x["labels"].apply(
                        lambda y: ";".join(sorted([z for z in y.split(";")]))
                    )
                )
                .drop_duplicates()
                .groupby("labels")
                .sample(frac=self.args.validation_set_fraction)
                .loc[:, "sample"]
            )
        
        # Mark validation samples
        image_files = image_files.assign(
            is_valid=image_files["sample"].isin(validation_samples),
            labels=lambda x: x["labels"].apply(
                lambda y: ";".join(sorted([z for z in y.split(";")]))
            ),
        )
        
        return image_files
    
    def check_label_types(self, image_files: pd.DataFrame) -> None:
        """
        Check label types and warn about mismatches.
        
        Args:
            image_files: DataFrame with image information
        """
        if self.args.single_label:
            eprint("Single label model requested.")
            if (image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "Some samples contain more than one label. These will be concatenated. Maybe you want a multilabel model instead?",
                    stacklevel=2,
                )
        else:
            eprint("Multilabel model requested.")
            if not (image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "No sample contains more than one label. Maybe you want a single label model instead?",
                    stacklevel=2,
                )
    
    def run(self) -> None:
        """
        Run the train command.
        """
        eprint("Starting train command.")
        
        # 1. Collect image files
        image_files = self.collect_images()
        
        # 2. Prepare validation split
        image_files = self.prepare_validation_split(image_files)
        
        # 3. Check label types
        self.check_label_types(image_files)
        
        # 4. Set up model and training parameters
        eprint("Setting up neural network model for training.")
        
        callback = {"MixUp": MixUp, "CutMix": CutMix, "None": None}[
            self.args.mix_augmentation
        ]
        
        # 5. Check for pretrained model
        model_state_dict = None
        
        if self.args.cpu:
            eprint("CPU forced by user. Using CPU for processing.")
            load_on_cpu = True
        elif torch.backends.mps.is_built() or (torch.backends.cuda.is_built() 
                                               and torch.cuda.device_count()):
            eprint("GPU available. Will try to use GPU for processing.")
            load_on_cpu = False
        else:
            load_on_cpu = True
            eprint("GPU not available. Using CPU for processing.")
        
        if self.args.pretrained_model:
            eprint("Loading pretrained model from file:", str(self.args.pretrained_model))
            past_learn = load_learner(self.args.pretrained_model, cpu=load_on_cpu)
            model_state_dict = past_learn.model.state_dict()
            pretrained = False
            del past_learn
        
        elif not self.args.random_weights and self.args.architecture not in CUSTOM_ARCHS:
            pretrained = True
            eprint("Starting model with pretrained weights from timm library.")
            eprint("Model architecture:", self.args.architecture)
        
        else:
            pretrained = False
            eprint("Starting model with random weights.")
            eprint("Model architecture:", self.args.architecture)
        
        # 6. Set loss function
        if self.args.mix_augmentation == "None" and self.args.single_label:
            loss = CrossEntropyLoss()
        elif self.args.single_label:
            loss = CrossEntropyLossFlat()
        else:
            loss = AsymmetricLossMultiLabel(
              gamma_pos=0, 
              gamma_neg=self.args.negative_downweighting, 
              eps=1e-2, 
              clip=0.1)
        
        # 7. Print training information
        eprint(
            "Start training for",
            self.args.freeze_epochs,
            "epochs with frozen model body weights followed by",
            self.args.epochs,
            "epochs with unfrozen weights and learning rate of",
            self.args.base_learning_rate,
        )
        
        # 8. Set additional parameters for multilabel training
        extra_params = {}
        if not self.args.single_label:
            extra_params = {
                "metrics_threshold": self.args.threshold,
            }
        
        # 9. Train model
        learn = train_nn(
            df=image_files,
            architecture=self.args.architecture,
            valid_pct=self.args.validation_set_fraction,
            max_bs=self.args.max_batch_size,
            min_bs=self.args.min_batch_size,
            base_lr=self.args.base_learning_rate,
            epochs=self.args.epochs,
            freeze_epochs=self.args.freeze_epochs,
            normalize=True,
            pretrained=pretrained,
            callbacks=callback,
            max_lighting=self.args.max_lighting,
            p_lighting=self.args.p_lighting,
            loss_fn=loss,
            model_state_dict=model_state_dict,
            verbose=not self.args.no_logging,
            is_multilabel=not self.args.single_label,
            num_workers=self.args.num_workers,
            no_metrics=self.args.no_metrics,
            force_cpu=self.args.cpu,
            **extra_params
        )
        
        # 10. Save results
        outdir = Path(self.args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        learn.export(outdir / "trained_model.pkl")
        with open(outdir / "labels.txt", "w") as outfile:
            outfile.write("\n".join(learn.dls.vocab))
        image_files.to_csv(outdir / "input_data.csv", index=False)
        
        eprint("Model, labels, and data table saved to directory", str(outdir))


def run_train_command(args: Any) -> None:
    """
    Run the train command with the given arguments.
    
    This is the main entry point for the train command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
    """
    train_cmd = TrainCommand(args)
    train_cmd.run()