#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train command module for varKoder.

This module contains functionality for training neural network models
on varKode images for DNA barcode classification.
"""

import os
import json
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
    vision_learner, Learner, cnn_learner, accuracy, error_rate
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

from varKoder.core.config import (
    LABELS_SEP, CUSTOM_ARCHS
)
from varKoder.core.utils import (
    eprint, get_metadata_from_img_filename, get_varKoder_labels,
    get_varKoder_qual
)
from varKoder.core.preprocessing import make_dataloaders
from varKoder.core.model_io import save_varkoder_model, recover_architecture, resolve_model
from varKoder.models.custom import (
    Fiannaca2018Model, Arias2022Model, instantiate_custom_model,
)


def export_trained_model(learn, outdir, *, architecture, is_multilabel):
    """Write both the legacy pkl (deprecated) and the safetensors artifact."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings.warn(
        "Exporting trained_model.pkl is deprecated and will be removed in a "
        "future release; use the safetensors artifact (varkoder_model.safetensors "
        "+ config.json).",
        DeprecationWarning, stacklevel=2,
    )
    # config must store the base timm name, never the "hf-hub:" form
    if architecture.startswith("hf-hub:"):
        architecture = recover_architecture(learn)
    learn.export(outdir / "trained_model.pkl")
    save_varkoder_model(learn, outdir, architecture=architecture,
                        is_multilabel=is_multilabel)
    with open(outdir / "labels.txt", "w") as f:
        f.write("\n".join(learn.dls.vocab))

def build_custom_model(architecture, dls):
    xb, _ = dls.one_batch()
    input_image_size = xb.shape[-2:]
    return instantiate_custom_model(
        architecture, len(dls.vocab), (1, input_image_size[0], input_image_size[1])
    )

class SkipValidationCallback(Callback):
    """Callback to skip validation during training."""
    def before_validate(self):
        raise CancelValidException

class CheckpointCallback(Callback):
    """Save model weights and training progress after every epoch.

    Writes ``last.pth`` (model weights) and ``progress.json`` (which phase and how many
    epochs of each phase have completed) into ``checkpoint_dir`` so an interrupted run can
    be resumed. Runs late (``order=99``) so metrics are recorded before the snapshot.
    """
    order = 99

    def __init__(self, checkpoint_dir, architecture, phase, frozen_base,
                 unfrozen_base, freeze_epochs):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.architecture = architecture
        self.phase = phase
        self.frozen_base = frozen_base
        self.unfrozen_base = unfrozen_base
        self.freeze_epochs = freeze_epochs

    def after_epoch(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        done_this = self.epoch + 1  # epochs completed within the current fit call

        if self.phase == "frozen":
            frozen_done = self.frozen_base + done_this
            unfrozen_done = self.unfrozen_base
        else:
            frozen_done = self.freeze_epochs
            unfrozen_done = self.unfrozen_base + done_this

        # Save the underlying model weights (unwrap DataParallel if present)
        model = self.learn.model
        model = model.module if hasattr(model, "module") else model
        tmp_path = self.checkpoint_dir / "last.pth.tmp"
        torch.save(model.state_dict(), tmp_path)
        tmp_path.replace(self.checkpoint_dir / "last.pth")

        progress = {
            "architecture": self.architecture,
            "phase": self.phase,
            "frozen_done": frozen_done,
            "unfrozen_done": unfrozen_done,
        }
        with open(self.checkpoint_dir / "progress.json", "w") as f:
            json.dump(progress, f)

def run_fine_tune(learn, epochs, freeze_epochs, base_lr, checkpoint_dir=None,
                  checkpoint_architecture=None, resume_progress=None,
                  lr_mult=100, pct_start=0.3, div=5.0):
    """Replicate fastai's ``Learner.fine_tune`` with per-epoch checkpointing and resume.

    Mirrors the two-phase schedule of ``fine_tune`` (frozen ``fit_one_cycle`` followed by an
    unfrozen ``fit_one_cycle`` with discriminative learning rates). When ``resume_progress``
    is supplied, epochs already completed in each phase are skipped. fastai has no mid-cycle
    resume, so the one-cycle schedule restarts for the remaining epochs of an interrupted
    phase.
    """
    frozen_done = 0
    unfrozen_done = 0
    phase = "frozen"
    if resume_progress:
        frozen_done = resume_progress.get("frozen_done", 0)
        unfrozen_done = resume_progress.get("unfrozen_done", 0)
        phase = resume_progress.get("phase", "frozen")

    # Frozen phase
    remaining_frozen = freeze_epochs - frozen_done
    if phase == "frozen" and remaining_frozen > 0:
        learn.freeze()
        cbs = None
        if checkpoint_dir is not None:
            cbs = CheckpointCallback(checkpoint_dir, checkpoint_architecture, "frozen",
                                     frozen_done, unfrozen_done, freeze_epochs)
        learn.fit_one_cycle(remaining_frozen, slice(base_lr), pct_start=0.99, cbs=cbs)

    # Unfrozen phase
    unfrozen_lr = base_lr / 2
    remaining_unfrozen = epochs - unfrozen_done
    if remaining_unfrozen > 0:
        learn.unfreeze()
        cbs = None
        if checkpoint_dir is not None:
            cbs = CheckpointCallback(checkpoint_dir, checkpoint_architecture, "unfrozen",
                                     freeze_epochs, unfrozen_done, freeze_epochs)
        learn.fit_one_cycle(remaining_unfrozen,
                            slice(unfrozen_lr / lr_mult, unfrozen_lr),
                            pct_start=pct_start, div=div, cbs=cbs)

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
    force_cpu=False,
    random_erasing=False,
    checkpoint_dir=None,
    resume_progress=None
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
        random_erasing: Whether to apply RandomErasing augmentation
        
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

    # Create data loaders with calculated batch size and appropriate device
    device = torch.device('cpu') if force_cpu else default_device()
    dls = make_dataloaders(
        df, architecture, is_multilabel, bs=batch_size, device=device,
        num_workers=num_workers, max_lighting=max_lighting, p_lighting=p_lighting,
        random_erasing=random_erasing,
    )

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

    # Recover the effective base timm architecture from the built model so that a resumed
    # run can rebuild it offline (a plain timm arch name builds without contacting the Hub,
    # whereas an "hf-hub:" name would trigger a config download).
    checkpoint_architecture = architecture
    try:
        checkpoint_architecture = learn.model[0].model.default_cfg["architecture"]
    except Exception:
        pass

    # Train the model with or without verbose output
    training_context = learn.no_bar() if not verbose else contextlib.nullcontext()
    logging_context = learn.no_logging() if not verbose else contextlib.nullcontext()

    with training_context, logging_context:
        run_fine_tune(
            learn,
            epochs=epochs,
            freeze_epochs=freeze_epochs,
            base_lr=base_lr,
            checkpoint_dir=checkpoint_dir,
            checkpoint_architecture=checkpoint_architecture,
            resume_progress=resume_progress,
        )

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

        self.checkpoint_dir = Path(args.outdir) / "checkpoints"
        self.resuming = False

        # If resuming, the output directory must already contain a checkpoint
        if getattr(args, "resume", False):
            progress_file = self.checkpoint_dir / "progress.json"
            if not progress_file.exists():
                raise Exception(
                    f"--resume was requested but no checkpoint was found at {progress_file}."
                )
            self.resuming = True
        # Otherwise, do not overwrite an existing output directory unless asked
        elif not args.overwrite:
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

        # Determine whether to load/train on CPU or GPU
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

        model_state_dict = None
        resume_progress = None
        pretrained = False

        if self.resuming:
            # Resume from checkpoint: reuse the exact split and architecture recorded at the
            # start of the interrupted run so the rebuilt model matches the saved weights.
            resume_progress = json.loads((self.checkpoint_dir / "progress.json").read_text())
            train_architecture = resume_progress["architecture"]
            eprint(
                "Resuming training from checkpoint.",
                "Frozen epochs completed:", resume_progress.get("frozen_done", 0),
                "- Unfrozen epochs completed:", resume_progress.get("unfrozen_done", 0),
            )
            image_files = pd.read_csv(self.checkpoint_dir / "input_data.csv")
            model_state_dict = torch.load(
                self.checkpoint_dir / "last.pth", map_location="cpu"
            )
        else:
            # 1. Collect image files
            image_files = self.collect_images()

            # 2. Prepare validation split
            image_files = self.prepare_validation_split(image_files)

            # 3. Check label types
            self.check_label_types(image_files)

            train_architecture = self.args.architecture

            use_pretrained = (
                self.args.pretrained_model
                and str(self.args.pretrained_model).lower() != "none"
                and not self.args.random_weights
            )
            if use_pretrained:
                eprint("Loading pretrained model from:", str(self.args.pretrained_model))
                pre_state, pre_config = resolve_model(self.args.pretrained_model)
                model_state_dict = pre_state
                train_architecture = pre_config["architecture"]
                pretrained = False

            elif not self.args.random_weights and self.args.architecture not in CUSTOM_ARCHS:
                pretrained = True
                eprint("Starting model with pretrained weights from timm library.")
                eprint("Model architecture:", self.args.architecture)

            else:
                pretrained = False
                eprint("Starting model with random weights.")
                eprint("Model architecture:", self.args.architecture)

            # Persist the split before training so an interrupted run can be resumed with
            # the same train/validation partition (and therefore the same label vocabulary).
            # Clear any stale checkpoint from a previous run first so a later --resume cannot
            # mix old progress with this run.
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            image_files.to_csv(self.checkpoint_dir / "input_data.csv", index=False)

        # 4. Set up model and training parameters
        eprint("Setting up neural network model for training.")

        callback = {"MixUp": MixUp, "CutMix": CutMix, "None": None}[
            self.args.mix_augmentation
        ]

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
            architecture=train_architecture,
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
            random_erasing=self.args.random_erasing,
            checkpoint_dir=self.checkpoint_dir,
            resume_progress=resume_progress,
            **extra_params
        )
        
        # 10. Save results
        outdir = Path(self.args.outdir)
        export_trained_model(
            learn, outdir,
            architecture=train_architecture,
            is_multilabel=not self.args.single_label,
        )
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