#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for varKoder.

This module defines the command-line interface structure for the varKoder tool,
including argument parsing and the main execution flow.
"""

import argparse
import sys
import tempfile
import shutil
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict, defaultdict

from varKoder.core.config import (
    VERSION, LABELS_SEP, BP_KMER_SEP, SAMPLE_BP_SEP, QUAL_THRESH, 
    MAPPING_CHOICES, DEFAULT_KMER_SIZE, DEFAULT_KMER_MAPPING,
    DEFAULT_THREADS, DEFAULT_CPUS_PER_THREAD, DEFAULT_OUTDIR,
    DEFAULT_STATS_FILE, DEFAULT_MIN_BP, DEFAULT_MAX_BP,
    DEFAULT_TRIM_BP, DEFAULT_THRESHOLD, DEFAULT_VALIDATION_SET_FRACTION,
    DEFAULT_ARCHITECTURE, DEFAULT_MAX_BATCH_SIZE, DEFAULT_MIN_BATCH_SIZE,
    DEFAULT_BASE_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_FREEZE_EPOCHS, 
    DEFAULT_NEGATIVE_DOWNWEIGHTING, DEFAULT_MIX_AUGMENTATION, 
    DEFAULT_P_LIGHTING, DEFAULT_MAX_LIGHTING, DEFAULT_MODEL
)
from varKoder.core.utils import (
    eprint, set_seed, process_input, get_kmer_mapping
)


def setup_parser():
    """
    Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser object
    """
    # Create top-level parser with common arguments
    main_parser = argparse.ArgumentParser(
        description="varKoder: using neural networks for DNA barcoding based on variation in whole-genome kmer frequencies",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Add version argument to main parser
    main_parser.add_argument("--version", action='version', version=f'varKoder {VERSION}', help="show varKoder version and exit")
    
    subparsers = main_parser.add_subparsers(required=False, dest="command")

    parent_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parent_parser.add_argument("-R", "--seed", help="random seed.", type=int)
    parent_parser.add_argument(
        "-x", "--overwrite", help="overwrite existing results.", action="store_true"
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        help="verbose output to stderr to help in debugging.",
        action="store_true",
        default=False,
    )
    parent_parser.add_argument("-vv", "--version", action='version', version=f'%(prog)s {VERSION}', help="prints varKoder version installed")
    
    # Create parser for image command
    parser_img = subparsers.add_parser(
        "image",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Preprocess reads and prepare images for Neural Network training.",
    )
    parser_img.add_argument(
        "input",
        help="path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.",
    )
    parser_img.add_argument(
        "-k", "--kmer-size", help="size of kmers to count (5–9)", type=int, default=DEFAULT_KMER_SIZE
    )
    parser_img.add_argument(
        "-p", "--kmer-mapping", help="method to map kmers. See online documentation for an explanation.", 
        type=str, default=DEFAULT_KMER_MAPPING, choices=MAPPING_CHOICES
    )
    
    parser_img.add_argument(
        "-n",
        "--n-threads",
        help="number of samples to preprocess in parallel.",
        default=DEFAULT_THREADS,
        type=int,
    )
    parser_img.add_argument(
        "-c",
        "--cpus-per-thread",
        help="number of cpus to use for preprocessing each sample.",
        default=DEFAULT_CPUS_PER_THREAD,
        type=int,
    )
    parser_img.add_argument(
        "-o",
        "--outdir",
        help="path to folder where to write final images.",
        default=DEFAULT_OUTDIR,
    )
    parser_img.add_argument(
        "-f",
        "--stats-file",
        help="path to file where sample statistics will be saved.",
        default=DEFAULT_STATS_FILE,
    )
    parser_img.add_argument(
        "-i",
        "--int-folder",
        help="folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used.",
    )
    parser_img.add_argument(
        "-m",
        "--min-bp",
        type=str,
        help="minimum number of post-cleaning basepairs to make an image. Samples below this threshold will be discarded",
        default=DEFAULT_MIN_BP,
    )
    parser_img.add_argument(
        "-M",
        "--max-bp",
        help="maximum number of post-cleaning basepairs to make an image. Use '0' to use all of the available data.",
        default=DEFAULT_MAX_BP
    )
    parser_img.add_argument(
        "-t",
        "--label-table",
        help="output a table with labels associated with each image, in addition to including them in the EXIF data.",
        action="store_true",
    )
    parser_img.add_argument(
        "-a",
        "--no-adapter",
        help="do not attempt to remove adapters from raw reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-D",
        "--no-deduplicate",
        help="do not attempt to remove duplicates in raw reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-r",
        "--no-merge",
        help="do not attempt to merge paired reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-X",
        "--no-image",
        help="clean and split raw reads, but do not generate image.",
        action="store_true",
    )
    parser_img.add_argument(
        "-T",
        "--trim-bp",
        help="number of base pairs to trim from the start and end of each read, separated by comma.",
        default=DEFAULT_TRIM_BP
    )

    # Create parser for train command
    parser_train = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train a Neural Network based on provided images.",
    )
    parser_train.add_argument("input", help="path to the folder with input images.")
    parser_train.add_argument(
        "outdir", help="path to the folder where trained model will be stored."
    )
    parser_train.add_argument(
        "-n",
        "--num-workers",
        help="number of subprocess for data loading. See https://docs.fast.ai/data.load.html#dataloader",
        default=0,
        type=int,
    )
    parser_train.add_argument(
        "-t",
        "--label-table-path",
        help="path to csv table with labels for each sample. By default, varKoder will instead read labels from the image file metadata.",
    )
    parser_train.add_argument(
        "-S",
        "--single-label",
        help="Train as a single-label image classification model instead of multilabel. If multiple labels are provided, they will be concatenated to a single label.",
        action="store_true",
    )
    parser_train.add_argument(
        "-d",
        "--threshold",
        help="Threshold to calculate precision and recall during for the validation set. Ignored if using --single-label",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    parser_train.add_argument(
        "-V",
        "--validation-set",
        help="comma-separated list of sample IDs to be included in the validation set, or path to a text file with such a list. If not provided, a random validation set will be created. Turns off --validation-set-fraction",
    )
    parser_train.add_argument(
        "-f",
        "--validation-set-fraction",
        help="fraction of samples to be held as a random validation set. Will be ignored if --validation-set is provided.",
        type=float,
        default=DEFAULT_VALIDATION_SET_FRACTION,
    )
    parser_train.add_argument(
        "-c",
        "--architecture",
        help="model architecture. Options include all those supported by timm library plus 'arias2022' and 'fiannaca2018'. See documentation for more info. ",
        default=DEFAULT_ARCHITECTURE,
    )
    parser_train.add_argument(
        "-m",
        "--pretrained-model",
        help="optional pickle file with pretrained model to update with new images. Turns off --architecture if used.",
    )
    parser_train.add_argument(
        "-b",
        "--max-batch-size",
        help="maximum batch size when using GPU for training.",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )
    parser_train.add_argument(
        "-B",
        "--min-batch-size",
        help="minimum batch size for training.",
        type=int,
        default=DEFAULT_MIN_BATCH_SIZE,
    )
    parser_train.add_argument(
        "-C",
        "--cpu",
        help="force use CPU for training instead of GPU.",
        action="store_true",
        default=False,
    )
    parser_train.add_argument(
        "-r",
        "--base-learning-rate",
        help="base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates.",
        type=float,
        default=DEFAULT_BASE_LEARNING_RATE,
    )
    parser_train.add_argument(
        "-e",
        "--epochs",
        help="number of epochs to train. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser_train.add_argument(
        "-z",
        "--freeze-epochs",
        help="number of freeze epochs to train. Recommended if using a pretrained model. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune",
        type=int,
        default=DEFAULT_FREEZE_EPOCHS,
    )
    parser_train.add_argument(
        "-w",
        "--random-weights",
        help="start training with random weights. By default, pretrained model weights are downloaded from timm. See https://github.com/rwightman/pytorch-image-models.",
        action="store_true",
    )
    parser_train.add_argument(
        "-i",
        "--negative_downweighting",
        type=float,
        default=DEFAULT_NEGATIVE_DOWNWEIGHTING,
        help="Parameter controlling strength of loss downweighting for negative samples. See gamma(negative) parameter in https://arxiv.org/abs/2009.14119. Ignored if used with --single-label.",
    )
    parser_train.add_argument(
        "-X",
        "--mix-augmentation",
        help="apply MixUp or CutMix augmentation. See https://docs.fast.ai/callback.mixup.html",
        choices=["CutMix", "MixUp", "None"],
        default=DEFAULT_MIX_AUGMENTATION,
    )
    parser_train.add_argument(
        "-s",
        "--label-smoothing",
        help="turn on Label Smoothing. Only used with --single-label. See https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb",
        action="store_true",
        default=False,
    )
    parser_train.add_argument(
        "-p",
        "--p-lighting",
        help="probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.augment.html#aug_transforms",
        type=float,
        default=DEFAULT_P_LIGHTING,
    )
    parser_train.add_argument(
        "-l",
        "--max-lighting",
        help="maximum scale of lighting transform. See https://docs.fast.ai/vision.augment.html#aug_transforms",
        type=float,
        default=DEFAULT_MAX_LIGHTING,
    )
    parser_train.add_argument(
        "-g",
        "--no-logging",
        help="hide fastai progress bar and logging during training.",
        action="store_true",
        default=False,
    )
    parser_train.add_argument(
        "-M",
        "--no-metrics",
        help="skip calculation of validation loss and metrics during training.",
        action="store_true",
        default=False,
    )

    # Create parser for query command
    parser_query = subparsers.add_parser(
        "query",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Query a fastq file agains a trained neural network. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and query this image. The response is written to stdout in json format",
    )

    parser_query.add_argument(
        "input", help="path to folder with fastq files to be queried."
    )
    parser_query.add_argument(
        "outdir", help="path to the folder where results will be saved."
    )
    parser_query.add_argument(
        "-l",
        "--model", 
        help="path pickle file with exported trained model or name of HuggingFace hub model",
        default=DEFAULT_MODEL
    )
    parser_query.add_argument(
        "-1",
        "--no-pairs",
        help="prevents varKoder query from considering folder structure in input to find read pairs. Each fastq file will be treated as a separate sample",
        action="store_true",
    )
    parser_query.add_argument(
        "-I",
        "--images",
        help="input folder contains processed images instead of raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-k", "--kmer-size", help="size of kmers to count (5–9)", type=int, default=DEFAULT_KMER_SIZE
    )
    parser_query.add_argument(
        "-p", "--kmer-mapping", help="method to map kmers. See online documentation for an explanation.", 
        type=str, default=DEFAULT_KMER_MAPPING, choices=MAPPING_CHOICES
    )
    
    parser_query.add_argument(
        "-n",
        "--n-threads",
        help="number of samples to preprocess in parallel.",
        default=DEFAULT_THREADS,
        type=int,
    )
    parser_query.add_argument(
        "-c",
        "--cpus-per-thread",
        help="number of cpus to use for preprocessing each sample.",
        default=DEFAULT_CPUS_PER_THREAD,
        type=int,
    )
    parser_query.add_argument(
        "-f",
        "--stats-file",
        help="path to file where sample statistics will be saved.",
        default=DEFAULT_STATS_FILE,
    )
    parser_query.add_argument(
        "-d",
        "--threshold",
        help="confidence threshold to make a prediction.",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    parser_query.add_argument(
        "-i",
        "--int-folder",
        help="folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used and deleted when done.",
    )
    parser_query.add_argument(
        "-m",
        "--keep-images",
        help="whether barcode images should be saved to a directory named 'query_images'.",
        action="store_true",
    )
    parser_query.add_argument(
        "-P",
        "--include-probs",
        help="whether probabilities for each label should be included in the output.",
        action="store_true",
    )
    parser_query.add_argument(
        "-a",
        "--no-adapter",
        help="do not attempt to remove adapters from raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-r",
        "--no-merge",
        help="do not attempt to merge paired reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-D",
        "--no-deduplicate",
        help="do not attempt to remove duplicates in raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-T",
        "--trim-bp",
        help="number of base pairs to trim from the start and end of each read, separated by comma.",
        default=DEFAULT_TRIM_BP
    )
    parser_query.add_argument(
        "-M",
        "--max-bp",
        help="number of post-cleaning basepairs to use for making image. Use '0' to use all of the available data.",
        default=DEFAULT_MAX_BP
    )
    parser_query.add_argument(
        "-b",
        "--max-batch-size",
        help="maximum batch size when using GPU for predictions.",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )

    # Create parser for convert command
    parser_cvt = subparsers.add_parser(
        "convert",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Convert images between different kmer mappings.",
    )
    parser_cvt.add_argument(
        "-n",
        "--n-threads",
        help="number of threads to process images in parallel.",
        default=DEFAULT_THREADS,
        type=int,
    )
    parser_cvt.add_argument(
        "-k","--kmer-size", help="size of kmers used to produce original images. Will be inferred from file names if omitted.", 
        type=int, default=DEFAULT_KMER_SIZE, choices=[5, 6, 7, 8, 9]
    )
    parser_cvt.add_argument(
        "-p","--input-mapping", help="kmer mapping of input images. Will be inferred from file names if omitted.", 
        choices=MAPPING_CHOICES
    )
    parser_cvt.add_argument(
        "-r","--sum-reverse-complements", help="When converting from CGR to varKode, add together counts from canonical kmers and their reverse complements.", 
        action='store_true'
    )

    parser_cvt.add_argument(
        "output_mapping", help="kmer mapping of output images.", choices=MAPPING_CHOICES
    )
    parser_cvt.add_argument(
        "input", help="path to folder with png images files to be converted."
    )
    parser_cvt.add_argument(
        "outdir", help="path to the folder where results will be saved."
    )
    
    return main_parser


def main():
    """
    Main entry point for the varKoder CLI.
    """
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # If no command is provided, show help and exit
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Parse max_bp (for compatibility with previous versions of the code)
    if hasattr(args, 'max_bp'):
        if args.max_bp == '0':
            args.max_bp = None
            eprint("All of the data will be used to produce images.")
        else:
            eprint(f"Images will be limited to --max-bp: {args.max_bp}b")

    # Check if input directory exists
    if not Path(args.input).exists():
        raise Exception("Input path", args.input, "does not exist. Please check.")

    # Set random seed
    try:
        set_seed(args.seed)
        np_rng = np.random.default_rng(args.seed)
    except TypeError:
        np_rng = np.random.default_rng()

    # Import command-specific modules only when needed
    if args.command == "image":
        from varKoder.commands.image import run_image_command
        run_image_command(args, np_rng)
    elif args.command == "train":
        from varKoder.commands.train import run_train_command
        run_train_command(args)
    elif args.command == "query":
        from varKoder.commands.query import run_query_command
        run_query_command(args, np_rng)
    elif args.command == "convert":
        from varKoder.commands.convert import run_convert_command
        run_convert_command(args)
    
    eprint("DONE")


if __name__ == "__main__":
    main()