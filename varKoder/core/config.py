#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration constants and settings for varKoder.

This module contains all the constants and configuration values used throughout
the varKoder application, including default parameters, file-naming conventions,
and quality thresholds.
"""

from importlib.metadata import version

# Version information
VERSION = version("varKoder")

# File naming conventions
LABEL_SAMPLE_SEP = "+"
LABELS_SEP = ";"
BP_KMER_SEP = "+"
SAMPLE_BP_SEP = "@"

# Quality thresholds
QUAL_THRESH = 0.01

# K-mer mapping options
MAPPING_CHOICES = ['varKode', 'cgr']

# Custom architecture choices
CUSTOM_ARCHS = ['fiannaca2018', 'arias2022']

# Default parameters
DEFAULT_KMER_SIZE = 7
DEFAULT_KMER_MAPPING = 'cgr'  # Original default was 'cgr'
DEFAULT_THRESHOLD = 0.7
DEFAULT_THREADS = 1
DEFAULT_CPUS_PER_THREAD = 1
DEFAULT_MIN_BP = "500K"
DEFAULT_MAX_BP = "200M"
DEFAULT_TRIM_BP = "10,10"
DEFAULT_VALIDATION_SET_FRACTION = 0.2
DEFAULT_BASE_LEARNING_RATE = 5e-3
DEFAULT_EPOCHS = 30
DEFAULT_FREEZE_EPOCHS = 0
DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_NEGATIVE_DOWNWEIGHTING = 4
DEFAULT_P_LIGHTING = 0.75
DEFAULT_MAX_LIGHTING = 0.25
DEFAULT_MIX_AUGMENTATION = "MixUp"
DEFAULT_ARCHITECTURE = "hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA"
DEFAULT_MODEL = "brunoasm/vit_large_patch32_224.NCBI_SRA"

# Output file names
DEFAULT_OUTDIR = "images"
DEFAULT_STATS_FILE = "stats.csv"

# External commands
FASTP_CMD = "fastp"
DSK_CMD = "dsk"
DSK2ASCII_CMD = "dsk2ascii"
REFORMAT_CMD = "reformat.sh"
PIGZ_CMD = "pigz"