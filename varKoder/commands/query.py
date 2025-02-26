#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Query command module for varKoder.

This module contains functionality for querying DNA sequences against a trained model.
It handles the preprocessing of reads, k-mer counting, image generation, and prediction.
"""

import os
import multiprocessing
import tempfile
import shutil
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

from varKoder.core.config import (
    LABELS_SEP, BP_KMER_SEP, SAMPLE_BP_SEP, QUAL_THRESH, 
    MAPPING_CHOICES, DEFAULT_KMER_SIZE, DEFAULT_KMER_MAPPING
)
from varKoder.core.utils import (
    eprint, get_kmer_mapping, process_input, get_varKoder_labels,
    get_varKoder_qual, get_varKoder_freqsd, get_metadata_from_img_filename
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from fastai.learner import load_learner
from huggingface_hub import from_pretrained_fastai


class QueryCommand:
    """
    Class for handling the query command functionality in varKoder.
    
    This class implements methods to process DNA sequences, generate images,
    and query a trained model for predictions.
    """
    
    def __init__(self, args: Any, np_rng: np.random.Generator) -> None:
        """
        Initialize QueryCommand with command line arguments and random number generator.
        
        Args:
            args: Parsed command line arguments
            np_rng: NumPy random number generator
        """
        self.args = args
        self.np_rng = np_rng
        self.all_stats = defaultdict(OrderedDict)
        
        # Validate kmer size if needed
        if not args.images:
            if args.kmer_size not in range(5, 9 + 1):
                raise ValueError("kmer size must be between 5 and 9")
                
        # Check if output directory exists and if predictions already exist
        if not args.overwrite:
            if Path(args.outdir, "predictions.csv").is_file():
                raise Exception(
                    "Output predictions file exists, use --overwrite if you want to overwrite it."
                )
        
        # Set up paths
        if not args.images:
            # If no directory provided for intermediate results, create a temporary one
            try:
                self.inter_dir = Path(args.int_folder)
            except TypeError:
                self.inter_dir = Path(tempfile.mkdtemp(prefix="barcoding_"))
                
            # Set up directory to save images if requested
            if args.keep_images:
                self.images_d = Path(args.outdir) / "query_images"
                self.images_d.mkdir(parents=True, exist_ok=True)
            elif args.int_folder:
                self.images_d = Path(tempfile.mkdtemp(prefix="barcoding_img_"))
            else:
                self.images_d = self.inter_dir / "images"
        else:
            # If input is already images, just set the images directory to the input
            self.images_d = Path(args.input)
            
        # Set up kmer mapping if needed
        if not args.images:
            self.kmer_mapping = get_kmer_mapping(args.kmer_size, args.kmer_mapping)
            
        # Initialize stats
        self.stats_path = Path(args.stats_file)
        if self.stats_path.exists():
            self.all_stats.update(
                pd.read_csv(self.stats_path, index_col=[0], dtype={0:str}, low_memory=False).to_dict(orient="index")
            )
    
    def process_samples(self, condensed_files: pd.DataFrame) -> None:
        """
        Process samples to generate images.
        
        This method is called only when the input is raw sequences, not images.
        
        Args:
            condensed_files: DataFrame with file information
        """
        from varKoder.commands.image import (
            run_clean2img, run_clean2img_wrapper, process_stats
        )
        
        # Determine subfolder structure
        n_records = condensed_files.shape[0]
        subfolder_levels = math.floor(math.log(n_records/1000, 16))
        
        # Prepare arguments for run_clean2img function
        args_for_multiprocessing = [
            (
                tup,
                self.kmer_mapping,
                self.args,
                self.np_rng,
                self.inter_dir,
                self.all_stats,
                self.stats_path,
                self.images_d,
                subfolder_levels
            )
            for tup in condensed_files.iterrows()
        ]
        
        # Single-threaded execution
        if self.args.n_threads == 1:
            for arg_tuple in args_for_multiprocessing:
                stats = run_clean2img(*arg_tuple)
                process_stats(
                    stats,
                    condensed_files,
                    self.args,
                    self.stats_path,
                    self.images_d,
                    self.all_stats,
                    QUAL_THRESH,
                    LABELS_SEP,
                )
        
        # Multi-threaded execution
        else:
            with multiprocessing.Pool(processes=int(self.args.n_threads)) as pool:
                for stats in pool.imap_unordered(
                    run_clean2img_wrapper, args_for_multiprocessing
                ):
                    process_stats(
                        stats,
                        condensed_files,
                        self.args,
                        self.stats_path,
                        self.images_d,
                        self.all_stats,
                        QUAL_THRESH,
                        LABELS_SEP,
                    )
    
    def extract_image_metadata(self, img_paths: List[Path]) -> Dict[str, List]:
        """
        Extract metadata from image files.
        
        Args:
            img_paths: List of paths to image files
            
        Returns:
            dict: Dictionary of metadata extracted from images
        """
        # Initialize metadata lists
        actual_labels = []
        qual_flags = []
        freq_sds = []
        sample_ids = []
        query_bp = []
        query_klen = []
        query_mapping = []
        
        # Extract metadata from each image
        for p in img_paths:
            try:
                labs = ";".join(get_varKoder_labels(p))
            except (AttributeError, TypeError):
                labs = np.nan

            try:
                qual_flag = get_varKoder_qual(p)
            except (AttributeError, TypeError):
                qual_flag = np.nan

            try:
                freq_sd = get_varKoder_freqsd(p)
            except (AttributeError, TypeError):
                freq_sd = np.nan

            img_metadata = get_metadata_from_img_filename(p)

            sample_ids.append(img_metadata['sample'])
            query_bp.append(img_metadata['bp'])
            query_klen.append(img_metadata['img_kmer_size'])
            query_mapping.append(img_metadata['img_kmer_mapping'])
            actual_labels.append(labs)
            qual_flags.append(qual_flag)
            freq_sds.append(freq_sd)
        
        # Return combined metadata
        return {
            "varKode_image_path": img_paths,
            "sample_id": sample_ids,
            "query_basepairs": query_bp,
            "query_kmer_len": query_klen,
            "query_mapping": query_mapping,
            "trained_model_path": str(self.args.model),
            "actual_labels": actual_labels,
            "possible_low_quality": qual_flags,
            "basefrequency_sd": freq_sds,
        }
    
    def load_model(self) -> Any:
        """
        Load the trained model for querying.
        
        Returns:
            The loaded model
        """
        n_images = len([img for img in self.images_d.rglob("*.png")])
        try:
            if n_images >= 128:
                eprint(n_images, "images in the input, will try to use GPU for prediction.")
                learn = load_learner(self.args.model, cpu=False)
            else:
                eprint(n_images, "images in the input, will use CPU for prediction.")
                learn = load_learner(self.args.model, cpu=True)
        except FileNotFoundError:
            eprint('Model', self.args.model, "not found locally, trying Hugging Face hub.")
            try: 
                learn = from_pretrained_fastai(self.args.model)
            except:
                raise Exception('Unable to load model', self.args.model, "locally or from Hugging Face Hub, please check")
        
        return learn
    
    def make_predictions(self, model: Any, img_paths: List[Path]) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            model: Trained model
            img_paths: List of paths to image files
            
        Returns:
            DataFrame: Predictions results
        """
        # Extract metadata from images
        common_data = self.extract_image_metadata(img_paths)
        
        # Create dataloader for prediction
        df = pd.DataFrame({"path": img_paths})
        query_dl = model.dls.test_dl(df, bs=self.args.max_batch_size)
        
        # Make predictions based on model type
        if "MultiLabel" in str(model.loss_func):
            eprint(
                "This is a multilabel classification model, each input may have 0 or more predictions."
            )
            pp, _ = model.get_preds(dl=query_dl, act=nn.Sigmoid())
            above_threshold = pp >= self.args.threshold
            vocab = model.dls.vocab
            predicted_labels = [
                ";".join([vocab[idx] for idx, val in enumerate(row) if val])
                for row in above_threshold
            ]
        
            output_df = pd.DataFrame({
                **common_data,
                "prediction_type": "Multilabel",
                "prediction_threshold": self.args.threshold,
                "predicted_labels": predicted_labels,
            })
        
        else:
            eprint(
                "This is a single label classification model, each input may have only one prediction."
            )
            pp, _ = model.get_preds(dl=query_dl)
        
            best_ps, best_idx = torch.max(pp, dim=1)
            best_labels = model.dls.vocab[best_idx]
        
            output_df = pd.DataFrame({
                **common_data,
                "prediction_type": "Single label",
                "best_pred_label": best_labels,
                "best_pred_prob": best_ps,
            })
        
        # Add raw probabilities if requested
        if self.args.include_probs:
            output_df = pd.concat([output_df, pd.DataFrame(pp, columns=model.dls.vocab)], axis=1)
        
        return output_df, pp
    
    def run(self) -> None:
        """
        Run the query command.
        """
        eprint("Running varKoder query command")
        
        # If input is raw sequences, process them to generate images
        if not self.args.images:
            eprint("Kmer size:", str(self.args.kmer_size))
            eprint("Processing reads and preparing images")
            eprint("Reading input data")
            
            inpath = Path(self.args.input)
            condensed_files = process_input(
                inpath, 
                is_query=True,
                no_pairs=self.args.no_pairs
            )
            
            if condensed_files.shape[0] == 0:
                raise Exception("No files found in input. Please check.")
            
            # Generate images from sequences
            self.process_samples(condensed_files)
            
            eprint("All images done, saved in", str(self.images_d))
        
        # Find all PNG images in the images directory
        img_paths = [img for img in self.images_d.rglob("*.png")]
        if not img_paths:
            raise Exception("No images found for querying. Please check input.")
        
        # Load the model
        model = self.load_model()
        
        # Make predictions
        output_df, _ = self.make_predictions(model, img_paths)
        
        # Save predictions
        outdir = Path(self.args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(outdir / "predictions.csv", index=False)
        
        # Clean up temporary directory if created
        if not self.args.images and not self.args.int_folder and self.inter_dir.is_dir():
            shutil.rmtree(self.inter_dir)
        
        eprint(f"Predictions saved to {outdir / 'predictions.csv'}")


def run_query_command(args: Any, np_rng: np.random.Generator) -> None:
    """
    Run the query command with the given arguments.
    
    This is the main entry point for the query command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
        np_rng: NumPy random number generator
    """
    query_cmd = QueryCommand(args, np_rng)
    query_cmd.run()