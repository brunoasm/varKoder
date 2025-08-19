#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for varKoder.

This module contains general utility functions used throughout the varKoder
application, including file handling, error reporting, and common operations.
"""

import sys
import os
import re
import math
import hashlib
import tempfile
import shutil
import subprocess
import contextlib
import traceback
import humanfriendly
import numpy as np
import pandas as pd
from importlib.resources import files
from pathlib import Path
from collections import OrderedDict, defaultdict
from functools import partial
from io import StringIO
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from varKoder.core.config import (
    LABEL_SAMPLE_SEP, LABELS_SEP, BP_KMER_SEP, 
    SAMPLE_BP_SEP, QUAL_THRESH, MAPPING_CHOICES
)


def eprint(*args, **kwargs):
    """
    Print to stderr for better error and status reporting.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments to pass to print function
    """
    print(*args, file=sys.stderr, **kwargs)


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed value
    """
    import torch
    import random
    import numpy as np
    from fastai.torch_core import set_seed as fastai_set_seed
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        fastai_set_seed(seed)


def get_varKoder_labels(img_path):
    """
    Extract labels from a varKoder image's metadata.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        List of labels
    """
    return [x for x in Image.open(img_path).info.get("varkoderKeywords").split(";")]


def get_varKoder_qual(img_path):
    """
    Extract quality flag from a varKoder image's metadata.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Boolean quality flag
    """
    return bool(Image.open(img_path).info.get("varkoderLowQualityFlag"))


def get_varKoder_freqsd(img_path):
    """
    Extract base frequency standard deviation from a varKoder image's metadata.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Float representing base frequency standard deviation
    """
    return float(Image.open(img_path).info.get("varkoderBaseFreqSd"))


def get_varKoder_mapping(img_path):
    """
    Extract k-mer mapping method from a varKoder image's metadata.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        String representing the mapping method
    """
    return str(Image.open(img_path).info.get("varkoderMapping"))


def format_bp_human_readable(bp_count):
    """
    Format base pair count in human-readable format with exactly 5 digits and optional suffix.
    Always uses 5 digits, padding with zeros when needed.
    
    Examples:
        1868000 -> "01868K"
        100000567 -> "00100G" 
        1898 -> "01898"
        5000000000 -> "05000M"
        123 -> "00123"
    
    Args:
        bp_count: Base pair count as integer
        
    Returns:
        String in format like "01868K", "00100G", "01898", "05000M"
    """
    # Round to 4 significant digits using scientific notation with 3 decimal places
    scientific = f"{bp_count:.3e}"
    
    # Parse scientific notation
    if 'e+' in scientific:
        mantissa_str, exp_str = scientific.split('e+')
        exponent = int(exp_str)
    elif 'e-' in scientific:
        mantissa_str, exp_str = scientific.split('e-')  
        exponent = -int(exp_str)
    else:
        return f"{bp_count:04d}"
    
    mantissa = float(mantissa_str)
    
    # Convert back to integer (rounded to 4 significant digits)
    rounded_number = int(mantissa * (10 ** exponent))
    
    # Convert to string and apply suffix replacements
    num_str = str(rounded_number)
    
    # Apply replacements from largest to smallest to avoid double replacement
    # But ensure the result always has exactly 4 digits before the suffix
    
    # Try each suffix and check if result would have <= 5 digits
    replacements = [
        ('000000000000000000', 'E', 18),  # exabytes (10^18)
        ('000000000000000', 'P', 15),     # petabytes (10^15)
        ('000000000000', 'T', 12),        # trillions (10^12)
        ('000000000', 'G', 9),            # billions (10^9)  
        ('000000', 'M', 6),               # millions (10^6)
        ('000', 'K', 3)                   # thousands (10^3)
    ]
    
    result = num_str
    for suffix_zeros, suffix_letter, zero_count in replacements:
        if num_str.endswith(suffix_zeros) and len(num_str) > zero_count:
            potential_result = num_str[:-zero_count] + suffix_letter
            numeric_part = potential_result[:-1]
            
            # Only apply this replacement if it results in <= 5 digits
            if len(numeric_part) <= 5:
                result = potential_result
                break
    
    # Ensure exactly 5 digits by padding with zeros if needed
    if result[-1].isdigit():
        # No suffix, pad the whole number to 5 digits
        return f"{result:>05s}"
    else:
        # Has suffix, pad the numeric part to exactly 5 digits
        suffix = result[-1]
        numeric_part = result[:-1]
        return f"{numeric_part:>05s}{suffix}"


def parse_bp_human_readable(bp_str):
    """
    Parse human-readable base pair format back to integer.
    Supports both new format (e.g., "1868K", "100M", "1898") 
    and legacy 8-digit format (e.g., "00001868K") for backwards compatibility.
    Uses humanfriendly for robust parsing.
    
    Args:
        bp_str: String like "1868K", "100M", "1898", or "00001868K"
        
    Returns:
        Base pair count as integer
    """
    bp_str = bp_str.strip()
    
    # Check if this is the old 8-digit format (8 digits followed by K)
    if len(bp_str) == 9 and bp_str[:-1].isdigit() and bp_str.endswith('K'):
        return int(bp_str[:-1]) * 1000
    
    # Use humanfriendly to parse both plain numbers and sizes with suffixes
    try:
        return int(humanfriendly.parse_size(bp_str))
    except Exception as e:
        raise ValueError(f"Invalid base pair format: {bp_str}") from e


def get_metadata_from_img_filename(img_path):
    """
    Extract metadata from a varKoder image filename.
    Supports both old 8-digit format and new human-readable format.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Dictionary with metadata
    """
    sample_name, split2 = Path(img_path).name.removesuffix('.png').split(SAMPLE_BP_SEP)
    try:
        n_bp, img_kmer_mapping, img_kmer_size = split2.split(BP_KMER_SEP)
    except ValueError:  # backwards compatible with varKoder v0.X
        n_bp, img_kmer_size = split2.split(BP_KMER_SEP)
        img_kmer_mapping = 'varKode'
    
    # Use the new parsing function that handles both old and new formats
    n_bp = parse_bp_human_readable(n_bp)
    img_kmer_size = int(img_kmer_size[1:])

    return {
        'sample': sample_name,
        'bp': n_bp,
        'img_kmer_mapping': img_kmer_mapping,
        'img_kmer_size': img_kmer_size,
        'path': Path(img_path)
    }


def get_kmer_mapping(kmer_size=7, method='varKode'):
    """
    Get k-mer mapping table for a given k-mer size and method.
    
    Args:
        kmer_size: Size of k-mers (5-9)
        method: Mapping method ('varKode' or 'cgr')
        
    Returns:
        DataFrame with k-mer mapping
    """
    if method == 'varKode':
        map_path = files("varKoder").joinpath(f"kmer_mapping/{kmer_size}mer_mapping.parquet")
        kmer_mapping = pd.read_parquet(map_path).set_index("kmer")
    elif method == 'cgr':
        kmer_mapping = get_cgr(kmer_size)
    else:
        raise Exception('method must be "varKode" or "cgr"')

    return kmer_mapping


def get_cgr(kmer_size):
    """
    Generate Chaos Game Representation coordinates for k-mers.
    
    Args:
        kmer_size: Size of k-mers
        
    Returns:
        DataFrame with CGR coordinates
    """
    # Following corners from Jeffrey, using cartesian coords
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)

    nucleotides = np.array([0, 1, 2, 3], dtype=np.uint8)  # A, C, G, T
    all_sequences = np.array(np.meshgrid(*[nucleotides]*kmer_size)).T.reshape(-1, kmer_size)

    rev_sequences = all_sequences[:, ::-1]  # Reverse the sequences
    rev_complement = 3 - rev_sequences      # Compute reverse complement

    # Calculate CGR coordinates using vectorized operations
    coords = np.full((all_sequences.shape[0], 2), 0.5)  # Start coordinates (x, y) at the center
    for i in range(kmer_size):
        coords = (coords + corners[all_sequences[:, i]]) / 2

    # Convert integer sequences back to string for display
    # Join sequences and reverse complements to the mapping
    nucleotide_map = np.array(['A', 'C', 'G', 'T'])
    df = pd.concat([
        pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1]
        }, index=[''.join(nucleotide_map[seq]) for seq in all_sequences]),
        pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1]
        }, index=[''.join(nucleotide_map[seq]) for seq in rev_complement])
    ])

    # Replace coords with integer numbers
    sq_side = len(df['x'].drop_duplicates())
    df['x'] = (sq_side*(df['x'] - df['x'].min())).astype(int)
    df['y'] = (sq_side*(df['y'] - df['y'].min())).astype(int)

    return df


def read_stats(stats_path):
    """
    Read statistics from a CSV file.
    
    Args:
        stats_path: Path to the CSV file
        
    Returns:
        Dictionary with statistics
    """
    return pd.read_csv(stats_path, index_col=[0], dtype={0: str}, low_memory=False).to_dict(
        orient="index"
    )


def stats_to_csv(all_stats, stats_path):
    """
    Save statistics dictionary to a CSV file.
    
    Args:
        all_stats: Dictionary with statistics
        stats_path: Path to the output CSV file
        
    Returns:
        DataFrame with statistics
    """
    df = pd.DataFrame.from_dict(all_stats, orient="index").rename_axis(index=["sample"])
    df.to_csv(stats_path)
    return df.reset_index()


def rc(seq):
    """
    Get the reverse complement of a DNA sequence.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Reverse complement string
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reverse_complement = "".join(complement[base] for base in reversed(seq))
    return reverse_complement


def is_fastq_file(filename):
    """
    Check if a file is a FASTQ file based on its extension.
    
    Args:
        filename: Filename string or Path object
        
    Returns:
        Boolean indicating if the file is a FASTQ file
    """
    name = str(filename)
    return (name.endswith("fq") or 
            name.endswith("fastq") or 
            name.endswith("fq.gz") or 
            name.endswith("fastq.gz"))

def is_fasta_file(filename):
    """
    Check if a file is a FASTA file based on its extension.
    
    Args:
        filename: Path to the file
        
    Returns:
        bool: True if file appears to be FASTA format
    """
    filename_str = str(filename).lower()
    fasta_extensions = ['.fasta', '.fa', '.fas', '.fna']
    
    # Check if filename ends with any fasta extension (accounting for possible .gz)
    for ext in fasta_extensions:
        if filename_str.endswith(ext):
            return True
        # Also check for compressed versions
        if filename_str.endswith(ext + '.gz'):
            return True
    return False


def process_input(inpath, is_query=False, no_pairs=False):
    """
    Process input file or directory and return a table with files.
    
    Args:
        inpath: Path to input file or directory
        is_query: Boolean indicating if this is a query operation
        no_pairs: Boolean indicating if paired reads should be treated separately
        
    Returns:
        DataFrame with file information
    """
    # First, check if input is a folder
    # If it is, make input table from the folder
    if inpath.is_dir() and not is_query:
        eprint(f"Analyzing input directory {inpath}")
        files_records = list()

        seen_samples = set()
        for taxon in inpath.iterdir():
            if taxon.is_dir():
                for sample in taxon.iterdir():
                    if sample.is_dir():
                        if sample.name in seen_samples:
                            raise Exception(
                                f"Duplicate sample name '{sample.name}' detected. Each sample must have a unique name across all taxa."
                            )
                        seen_samples.add(sample.name)
                        for fl in sample.iterdir():
                            # Check if file is a sequence file (fastq or fasta)
                            if is_fastq_file(fl.name) or is_fasta_file(fl.name):
                                files_records.append(
                                    {
                                        "labels": (taxon.name,),
                                        "sample": sample.name,
                                        "files": taxon / sample.name / fl.name,
                                    }
                                )
                            else:
                                tmp_fl_path = taxon / sample.name / fl.name
                                eprint(f"Warning: File '{tmp_fl_path}' is not recognized as a sequence file and will be ignored.")
        try:
            files_table = (
                pd.DataFrame(files_records)
                .groupby(["labels", "sample"])
                .agg(list)
                .reset_index()
            )
        except KeyError as e:
            if str(e) == "'labels'":
                raise Exception("Folder input could not be parsed. Check the github documentation for input format.") from e
            else:
                raise

        if not files_table.shape[0]:
            raise Exception("Folder detected, but no records read. Check format.")

    elif is_query:

        eprint("Analyzing input directory {inpath}")

        files_records = list()
        # Start by checking if input directory contains directories
        contains_dir = any(f.is_dir() or 
                           (f.is_symlink() and Path(os.readlink(f)).is_dir()) 
                           for f in inpath.iterdir())
        # If there are no subdirectories, or no_pairs is True treat each fastq as a single sample. Otherwise, use each directory for a sample
        if not contains_dir:
            for i, fl in enumerate(inpath.rglob('*')):
                if is_fastq_file(fl.name) or is_fasta_file(fl.name):
                    files_records.append(
                        {
                            "labels": ("query",),
                            "sample": fl.name.split(".")[0],
                            "files": fl,
                        }
                    )

        else:
            for sample in inpath.iterdir():
                if sample.resolve().is_dir():
                    for fl in sample.iterdir():
                        if is_fastq_file(fl.name) or is_fasta_file(fl.name):
                            files_records.append(
                                {
                                    "labels": ("query",),
                                    "sample": sample.name,
                                    "files": sample / fl.name,
                                }
                            )

        # Create DataFrame from records and group directly - consistent with original implementation
        files_table = (
            pd.DataFrame(files_records)
            .groupby(["labels", "sample"])
            .agg(list)
            .reset_index()
        )

        if not files_table.shape[0]:
            raise Exception("Folder detected, but no records read. Check format.")

    # If it isn't a folder, read csv table input
    else:
        eprint("Analyzing input table {inpath}")
        files_table = pd.read_csv(inpath)
        for colname in ["labels", "sample", "files"]:
            if colname not in files_table.columns:
                raise Exception("Input csv file missing column: " + colname)
        else:
            files_table = files_table.assign(
                labels=lambda x: x["labels"].str.split(";")
            ).assign(
                files=lambda x: x["files"].apply(
                    lambda y: [str(Path(inpath.parent, z)) for z in y.split(";")]
                )
            )

    files_table["sample"] = files_table["sample"].astype(str)

    files_table = (
        files_table.loc[:, ["labels", "sample", "files"]]
        .groupby("sample")
        .agg("sum")
        .map(lambda x: sorted(set(x)))
        .reset_index()
    )

    return files_table
