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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from fastai.vision.all import (
    aug_transforms, vision_learner, CategoryBlock, ImageBlock, 
    MultiCategoryBlock, DataBlock, ColReader, ColSplitter,
    Learner, cnn_learner, accuracy, error_rate
)
from fastai.callback.mixup import MixUp, CutMix
from fastai.torch_core import set_seed
from fastai.learner import load_learner
from fastai.losses import CrossEntropyLossFlat
from torch.nn import CrossEntropyLoss
from timm.loss import AsymmetricLossMultiLabel

from varKoder.core.config import (
    LABELS_SEP, CUSTOM_ARCHS
)
from varKoder.core.utils import (
    eprint, get_metadata_from_img_filename, get_varKoder_labels, 
    get_varKoder_qual
)


def train_nn(
    df,
    architecture,
    valid_pct=0.2,
    max_bs=64,
    base_lr=1e-2,
    epochs=30,
    freeze_epochs=0,
    normalize=True,
    pretrained=True,
    callbacks=None,
    metrics_threshold=0.7,
    max_lighting=0.25,
    p_lighting=0.75,
    loss_fn=None,
    model_state_dict=None,
    verbose=True,
    is_multilabel=True,
    num_workers=0,
    no_metrics=False,
):
    """
    Train a neural network model on varKode images.
    
    Args:
        df: DataFrame with image paths and labels
        architecture: Model architecture to use
        valid_pct: Percentage of data to use for validation
        max_bs: Maximum batch size
        base_lr: Base learning rate
        epochs: Number of epochs to train
        freeze_epochs: Number of epochs to train with frozen weights
        normalize: Whether to normalize input images
        pretrained: Whether to use pretrained weights
        callbacks: Training callbacks to use
        metrics_threshold: Threshold for metrics calculation
        max_lighting: Maximum lighting augmentation
        p_lighting: Probability of lighting augmentation
        loss_fn: Loss function to use
        model_state_dict: State dict from a pretrained model
        verbose: Whether to show training progress
        is_multilabel: Whether this is a multilabel classification task
        num_workers: Number of workers for data loading
        no_metrics: Whether to skip metrics calculation
        
    Returns:
        Trained fastai Learner object
    """
    # Label handling
    if is_multilabel:
        # Get unique labels from all samples
        labels = []
        for ls in df["labels"].values:
            labels.extend(ls.split(";"))
        labels = sorted(list(set(labels)))
        
        # Create vocabulary lookup
        vocab = labels
        
        # Set up blocks for multilabel classification
        blocks = (ImageBlock, MultiCategoryBlock(vocab=vocab))
        
        # Define label getter function
        item_tfms = [
            lambda x: np.array(
                [1 if l in x.split(";") else 0 for l in vocab], dtype=np.float32
            )
        ]
        
        # Set up metrics for multilabel
        if not no_metrics:
            from varKoder.models.metrics import F1ScoreMulti, PrecisionMulti, RecallMulti
            metrics = [
                F1ScoreMulti(thresh=metrics_threshold),
                PrecisionMulti(thresh=metrics_threshold),
                RecallMulti(thresh=metrics_threshold),
            ]
        else:
            metrics = []
            
    else:
        # For single label classification
        blocks = (ImageBlock, CategoryBlock)
        item_tfms = None
        metrics = [accuracy, error_rate] if not no_metrics else []
    
    # Set up data loaders
    dblock = DataBlock(
        blocks=blocks,
        get_x=ColReader("path"),
        get_y=ColReader("labels") if not item_tfms else lambda x: x["labels"],
        item_tfms=item_tfms,
        batch_tfms=[
            aug_transforms(
                do_flip=False,
                max_rotate=0,
                max_zoom=1,
                max_lighting=max_lighting,
                max_warp=0,
                p_affine=0,
                p_lighting=p_lighting,
            )
        ],
        splitter=ColSplitter(),
    )
    
    dls = dblock.dataloaders(df, bs=max_bs, num_workers=num_workers)
    
    # Determine appropriate architecture
    if architecture in ["fiannaca2018", "arias2022"]:
        from varKoder.models.architectures import fiannaca2018_model, arias2022_model
        
        if architecture == "fiannaca2018":
            model = fiannaca2018_model(len(dls.vocab), is_multilabel=is_multilabel)
        else:
            model = arias2022_model(len(dls.vocab), is_multilabel=is_multilabel)
            
        learn = Learner(
            dls=dls,
            model=model,
            metrics=metrics,
            loss_func=loss_fn,
            cbs=callbacks() if callbacks else None,
        )
        
    elif "hf-hub" in architecture:
        # For HuggingFace hub models
        from fastkaggle.vision.all import get_hf_model
        hf_model = architecture.split(":", 1)[1]
        
        model = get_hf_model(
            hf_model, 
            pretrained=pretrained, 
            num_classes=len(dls.vocab) if is_multilabel else len(dls.vocab)
        )
        
        learn = Learner(
            dls=dls,
            model=model,
            metrics=metrics,
            loss_func=loss_fn,
            cbs=callbacks() if callbacks else None,
        )
        
    else:
        # For timm models
        learn = vision_learner(
            dls,
            architecture,
            metrics=metrics,
            normalize=normalize,
            loss_func=loss_fn,
            pretrained=pretrained,
            cbs=callbacks() if callbacks else None,
        )
    
    # Load state dict if provided
    if model_state_dict is not None:
        learn.model.load_state_dict(model_state_dict)
    
    # Train the model
    with learn.no_bar() if not verbose else learn.no_mbar():
        if freeze_epochs > 0:
            learn.freeze()
            learn.fit_one_cycle(freeze_epochs, base_lr)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, base_lr)
    
    return learn


class TrainCommand:
    """
    Class for handling the train command functionality in varKoder.
    
    This class implements methods to train neural network models on 
    varKode images for species identification and other classification tasks.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize TrainCommand with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        
        # Check if output directory exists
        if not args.overwrite and Path(args.outdir).exists():
            raise Exception("Output directory exists, use --overwrite if you want to overwrite it.")
            
        # Set up variables
        self.image_files = None
        self.validation_samples = None
        self.model_state_dict = None
        self.pretrained = False
        self.load_on_cpu = True
        self.callback = None
        self.trans = None
        self.loss = None
        
    def _process_image_files(self) -> None:
        """
        Process image files found in the input directory.
        Creates a DataFrame with metadata for all image files.
        """
        eprint("Starting train command.")
        
        # Create a data table for all images
        image_files = list()
        f_counter = 0
        for f in Path(self.args.input).rglob("*.png"):
            image_files.append(get_metadata_from_img_filename(f))
            f_counter += 1
            if f_counter % 1000 == 0:
                eprint(f"\rFound {f_counter} image files", end='', flush=True)
        eprint(f"\rFound {f_counter} image files", flush=True)
        
        # Process image files based on label source
        if self.args.label_table_path:
            n_image_files = pd.DataFrame(image_files).merge(
                pd.read_csv(self.args.label_table_path)[
                    ["sample", "labels"]
                ],
                on="sample",
                how="inner",
            )
            excluded_samples = set([x["sample"] for x in image_files]) - set(n_image_files["sample"])
            eprint(len(excluded_samples), "samples excluded due to absence in provided label table.")
            if self.args.verbose:
                eprint('Samples excluded:\n', '\n'.join(excluded_samples))
            image_files = n_image_files
        else:
            image_files = pd.DataFrame(image_files).assign(
                labels=lambda x: x["path"].apply(
                    lambda y: ";".join(get_varKoder_labels(y))
                ),
                possible_low_quality=lambda x: x["path"].apply(get_varKoder_qual),
            )
            
        self.image_files = image_files
    
    def _determine_validation_set(self) -> None:
        """
        Determine the validation set based on command arguments.
        Sets the validation_samples attribute and updates image_files.
        """
        # Determine validation set
        if self.args.validation_set:
            eprint("Splitting validation set as defined by user.")
            try:  # try to treat as a path first
                with open(self.args.validation_set, "r") as valsamps:
                    self.validation_samples = valsamps.readline().strip().split(",")
            except:  # if not, try to treat as a list
                self.validation_samples = self.args.validation_set.split(",")
        else:
            eprint(
                "Splitting validation set randomly. Fraction of samples per label combination held as validation:",
                str(self.args.validation_set_fraction),
            )
            self.validation_samples = (
                self.image_files[["sample", "labels"]]
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
        
        # Mark validation samples in the dataset
        self.image_files = self.image_files.assign(
            is_valid=self.image_files["sample"].isin(self.validation_samples),
            labels=lambda x: x["labels"].apply(
                lambda y: ";".join(sorted([z for z in y.split(";")]))
            ),
        )
    
    def _setup_training_parameters(self) -> None:
        """
        Set up neural network training parameters based on command arguments.
        """
        eprint("Setting up neural network model for training.")
        
        # Set up callback
        self.callback = {"MixUp": MixUp, "CutMix": CutMix, "None": None}[
            self.args.mix_augmentation
        ]
        
        # Set up transforms
        self.trans = aug_transforms(
            do_flip=False,
            max_rotate=0,
            max_zoom=1,
            max_lighting=self.args.max_lighting,
            max_warp=0,
            p_affine=0,
            p_lighting=self.args.p_lighting,
        )
        
        # Check GPU availability
        if torch.backends.mps.is_built() or (
            torch.backends.cuda.is_built() and torch.cuda.device_count()
        ):
            eprint("GPU available. Will try to use GPU for processing.")
            self.load_on_cpu = False
        else:
            self.load_on_cpu = True
            eprint("GPU not available. Using CPU for processing.")
        
        # Handle pretrained model if provided
        if self.args.pretrained_model:
            eprint("Loading pretrained model from file:", str(self.args.pretrained_model))
            past_learn = load_learner(self.args.pretrained_model, cpu=self.load_on_cpu)
            self.model_state_dict = past_learn.model.state_dict()
            self.pretrained = False
            del past_learn
        elif not self.args.random_weights and self.args.architecture not in CUSTOM_ARCHS:
            self.pretrained = True
            eprint("Starting model with pretrained weights from timm library.")
            eprint("Model architecture:", self.args.architecture)
        else:
            self.pretrained = False
            eprint("Starting model with random weights.")
            eprint("Model architecture:", self.args.architecture)
    
    def _validate_label_types(self) -> None:
        """
        Validate label types and warn if there seems to be a mismatch.
        """
        # Check for label types and warn if there seems to be a mismatch
        if self.args.single_label:
            eprint("Single label model requested.")
            if (self.image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "Some samples contain more than one label. These will be concatenated. "
                    "Maybe you want a multilabel model instead?",
                    stacklevel=2,
                )
        else:
            eprint("Multilabel model requested.")
            if not (self.image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "No sample contains more than one label. Maybe you want a single label model instead?",
                    stacklevel=2,
                )
    
    def _setup_loss_function(self) -> None:
        """
        Set up the appropriate loss function based on command arguments.
        """
        # Set loss function based on args.mix_augmentation
        if self.args.mix_augmentation == "None" and self.args.single_label:
            self.loss = CrossEntropyLoss()
        elif self.args.single_label:
            self.loss = CrossEntropyLossFlat()
        else:
            self.loss = AsymmetricLossMultiLabel(
                gamma_pos=0,
                gamma_neg=self.args.negative_downweighting,
                eps=1e-2,
                clip=0.1
            )
    
    def _train_model(self) -> None:
        """
        Train the model with the determined parameters.
        """
        # Print training information
        eprint(
            "Start training for",
            self.args.freeze_epochs,
            "epochs with frozen model body weights followed by",
            self.args.epochs,
            "epochs with unfrozen weights and learning rate of",
            self.args.base_learning_rate,
        )
        
        # Additional parameters for multilabel training
        extra_params = {}
        if not self.args.single_label:
            extra_params = {
                "metrics_threshold": self.args.threshold,
            }
        
        # Call training function
        self.learn = train_nn(
            df=self.image_files,
            architecture=self.args.architecture,
            valid_pct=self.args.validation_set_fraction,
            max_bs=self.args.max_batch_size,
            base_lr=self.args.base_learning_rate,
            epochs=self.args.epochs,
            freeze_epochs=self.args.freeze_epochs,
            normalize=True,
            pretrained=self.pretrained,
            callbacks=self.callback,
            max_lighting=self.args.max_lighting,
            p_lighting=self.args.p_lighting,
            loss_fn=self.loss,
            model_state_dict=self.model_state_dict,
            verbose=not self.args.no_logging,
            is_multilabel=not self.args.single_label,
            num_workers=self.args.num_workers,
            no_metrics=self.args.no_metrics,
            **extra_params
        )
    
    def _save_results(self) -> None:
        """
        Save the trained model, labels, and data table.
        """
        # Save results
        outdir = Path(self.args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        self.learn.export(outdir / "trained_model.pkl")
        with open(outdir / "labels.txt", "w") as outfile:
            outfile.write("\n".join(self.learn.dls.vocab))
        self.image_files.to_csv(outdir / "input_data.csv", index=False)
        
        eprint("Model, labels, and data table saved to directory", str(outdir))
    
    def run(self) -> None:
        """
        Run the train command.
        """
        # Process image files
        self._process_image_files()
        
        # Determine validation set
        self._determine_validation_set()
        
        # Set up training parameters
        self._setup_training_parameters()
        
        # Validate label types
        self._validate_label_types()
        
        # Set up loss function
        self._setup_loss_function()
        
        # Train model
        self._train_model()
        
        # Save results
        self._save_results()


def run_train_command(args: Any) -> None:
    """
    Run the train command with the given arguments.
    
    This is the main entry point for the train command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
    """
    train_cmd = TrainCommand(args)
    train_cmd.run()