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

from varKoder.commands.image import (
    clean_reads, split_fastq, count_kmers, make_image, 
    run_clean2img, run_clean2img_wrapper
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
        
        # Check if kmer size provided is supported
        if args.kmer_size not in range(5, 9 + 1):
            raise ValueError("kmer size must be between 5 and 9")
        
        # Set up intermediate directory
        try:
            self.inter_dir = Path(args.int_folder)
        except TypeError:
            self.inter_dir = Path(tempfile.mkdtemp(prefix="barcoding_"))
            
        # Check if output directory exists
        if not args.overwrite:
            if Path(args.outdir, "predictions.csv").is_file():
                raise Exception(
                    "Output predictions file exists, use --overwrite if you want to overwrite it."
                )
        
        # Set directory to save images
        if args.images:
            self.images_d = Path(args.input)
        elif args.keep_images:
            self.images_d = Path(args.outdir) / "query_images"
            self.images_d.mkdir(parents=True, exist_ok=True)
        elif args.int_folder:
            self.images_d = Path(tempfile.mkdtemp(prefix="barcoding_img_"))
        else:
            self.images_d = self.inter_dir / "images"
        
        # Set up kmer mapping
        self.kmer_mapping = get_kmer_mapping(args.kmer_size, args.kmer_mapping)
    
    def prepare_images(self) -> List[Path]:
        """
        Prepare images for querying.
        
        If the input is raw reads, process them into images.
        If the input is already images, just collect them.
        
        Returns:
            List of image paths
        """
        if self.args.images:
            # Input is already images, just collect them
            return [img for img in self.images_d.rglob("*.png")]
        
        # Check if input directory contains PNG files
        inpath = Path(self.args.input)
        png_files = list(inpath.glob("*.png"))
        
        # If PNG files are found, suggest using the --images flag
        if png_files:
            eprint("ERROR: Found PNG files in input directory.")
            eprint("If your input directory contains pre-generated images, use the --images flag:")
            eprint("    varKoder query --images " + str(inpath) + " " + str(self.args.outdir))
            raise Exception("Input directory contains PNG files. Use --images flag for pre-generated images.")
        
        # Input is raw reads, process them into images
        eprint("Processing reads and preparing images")
        eprint("Reading input data")
        
        # Parse input and create a table relating reads files to samples and taxa
        try:
            condensed_files = process_input(
                inpath, 
                is_query=True, 
                no_pairs=getattr(self.args, 'no_pairs', False)
            )
            
            if condensed_files.shape[0] == 0:
                raise Exception("No files found in input. Please check.")
            
            # Process samples to generate images
            # Prepare arguments for run_clean2img function
            args_for_multiprocessing = [
                (
                    tup,
                    self.kmer_mapping,
                    self.args,
                    self.np_rng,
                    self.inter_dir,
                    self.all_stats,
                    Path(self.args.stats_file),
                    self.images_d,
                    0  # No subfolder levels for query
                )
                for tup in condensed_files.iterrows()
            ]
            
            # Single-threaded execution
            if self.args.n_threads == 1:
                for arg_tuple in args_for_multiprocessing:
                    run_clean2img(*arg_tuple)
            
            # Multi-threaded execution
            else:
                with multiprocessing.Pool(processes=int(self.args.n_threads)) as pool:
                    for _ in pool.imap_unordered(
                        run_clean2img_wrapper, args_for_multiprocessing
                    ):
                        pass
            
            eprint("All images prepared, saved in", str(self.images_d))
            
            return [img for img in self.images_d.rglob("*.png")]
        
        except KeyError as e:
            if str(e) == "'labels'":
                eprint("Error processing input directory. Check if it contains images or if it has the expected structure.")
                eprint("If your directory contains images, use the --images flag:")
                eprint("    varKoder query --images " + str(inpath) + " " + str(self.args.outdir))
                raise Exception("Input format error. If using pre-generated images, use --images flag.") from e
            else:
                raise
    
    def load_model(self) -> Any:
        """
        Load the model for inference.
        
        Returns:
            Loaded model
        """
        n_images = len([img for img in self.images_d.rglob("*.png")])
        
        try:
            if n_images >= 128:
                eprint(n_images, "images in the input, will try to use GPU for prediction.")
                model = load_learner(self.args.model, cpu=False)
            else:
                eprint(n_images, "images in the input, will use CPU for prediction.")
                model = load_learner(self.args.model, cpu=True)
        except FileNotFoundError:
            eprint('Model', self.args.model, "not found locally, trying Hugging Face hub.")
            try: 
                model = from_pretrained_fastai(self.args.model)
            except Exception as e:
                raise Exception('Unable to load model', self.args.model, "locally or from Hugging Face Hub, please check")
        
        return model
    
    def run(self) -> None:
        """
        Run the query command.
        """
        # Prepare images (either process raw reads or collect existing images)
        img_paths = self.prepare_images()
        
        if not img_paths:
            raise Exception("No images found to query. Please check your input.")
        
        # Extract metadata from images
        actual_labels = []
        qual_flags = []
        freq_sds = []
        sample_ids = []
        query_bp = []
        query_klen = []
        query_mapping = []
        
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

            img_metadatada = get_metadata_from_img_filename(p)

            
            sample_ids.append(img_metadatada['sample'])
            query_bp.append(img_metadatada['bp'])
            query_klen.append(img_metadatada['img_kmer_size'])
            query_mapping.append(img_metadatada['img_kmer_mapping'])
            actual_labels.append(labs)
            qual_flags.append(qual_flag)
            freq_sds.append(freq_sd)
        
        # Start output dataframe
        common_data = {
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
        
        # Load model
        learn = self.load_model()
        
        # Create data loader for inference
        df = pd.DataFrame({"path": img_paths})
        query_dl = learn.dls.test_dl(df, bs=self.args.max_batch_size)
        
        # Make predictions
        if "MultiLabel" in str(learn.loss_func):
            eprint(
                "This is a multilabel classification model, each input may have 0 or more predictions."
            )
            pp, _ = learn.get_preds(dl=query_dl, act=nn.Sigmoid())
            above_threshold = pp >= self.args.threshold
            vocab = learn.dls.vocab
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
                "This is a single label classification model, each input may will have only one prediction."
            )
            pp, _ = learn.get_preds(dl=query_dl)
        
            best_ps, best_idx = torch.max(pp, dim=1)
            best_labels = [learn.dls.vocab[i] for i in best_idx]
        
            output_df = pd.DataFrame({
                **common_data,
                "prediction_type": "Single label",
                "best_pred_label": best_labels,
                "best_pred_prob": best_ps.tolist(),
            })
        
        # Add probabilities if requested
        if self.args.include_probs:
            prob_df = pd.DataFrame(pp.numpy(), columns=learn.dls.vocab)
            output_df = pd.concat([output_df, prob_df], axis=1)
        
        # Save results
        outdir = Path(self.args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(outdir / "predictions.csv", index=False)
        
        eprint("Predictions saved to", str(outdir / "predictions.csv"))
        
        # Clean up temporary directory if created
        if not self.args.int_folder and not self.args.keep_images and self.inter_dir.is_dir():
            shutil.rmtree(self.inter_dir)


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