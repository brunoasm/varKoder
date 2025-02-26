#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image command module for varKoder.

This module contains functionality for generating varKode images from DNA sequences.
It handles the preprocessing of reads, k-mer counting, and image generation.
"""

import os
import multiprocessing
import tempfile
import shutil
import math
import humanfriendly
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

from varKoder.core.config import (
    LABELS_SEP, BP_KMER_SEP, SAMPLE_BP_SEP, QUAL_THRESH, 
    MAPPING_CHOICES, DEFAULT_KMER_SIZE, DEFAULT_KMER_MAPPING,
    FASTP_CMD, DSK_CMD, DSK2ASCII_CMD, REFORMAT_CMD, PIGZ_CMD
)
from varKoder.core.utils import (
    eprint, get_kmer_mapping, process_input, stats_to_csv, read_stats
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import subprocess
import os


def get_basefrequency_sd(fastq_path):
    """
    Calculate standard deviation of base frequencies in a FASTQ file.
    
    Args:
        fastq_path (Path): Path to FASTQ file
        
    Returns:
        float: Standard deviation of base frequencies
    """
    # Count frequency of bases
    base_counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    lines_to_analyze = min(4 * 100000, sum(1 for _ in open(fastq_path)))
    
    with open(fastq_path, "r") as f:
        for i, line in enumerate(f):
            if i >= lines_to_analyze:
                break
            if i % 4 == 1:  # Sequence line
                for base in line.strip():
                    if base in base_counts:
                        base_counts[base] += 1
    
    total = sum(base_counts.values())
    if total == 0:
        return 0
    
    # Calculate frequencies
    frequencies = [count / total for count in base_counts.values()]
    
    # Calculate standard deviation
    mean = sum(frequencies) / len(frequencies)
    variance = sum((x - mean) ** 2 for x in frequencies) / len(frequencies)
    return math.sqrt(variance)


def clean_reads(sample_files, sample_name, args, inter_dir):
    """
    Clean and preprocess sequencing reads.
    
    Args:
        sample_files (list): List of input FASTQ files
        sample_name (str): Sample name
        args: Command line arguments
        inter_dir (Path): Intermediate output directory
        
    Returns:
        tuple: (cleaned_fastq_path, n_reads, n_bp, failed_samples)
    """
    try:
        inter_dir.mkdir(parents=True, exist_ok=True)
    except:
        raise Exception(f"Error creating directory {inter_dir}")
    
    min_bp = humanfriendly.parse_size(args.min_bp) if args.min_bp else 0
    maxbp = humanfriendly.parse_size(args.max_bp) if args.max_bp else None
    
    failed_samples = []
    n_pairs = len(sample_files) // 2
    out_fastq = inter_dir / f"{sample_name}.fastq"

    fastp_args = []
    if args.no_adapter:
        fastp_args.append("--disable_adapter_trimming")
    
    if args.no_merge:
        fastp_args.append("--disable_adapter_trimming")
    
    if n_pairs > 0:
        eprint(f"Sample {sample_name} has {n_pairs} pairs. Cleaning...")
        
        # Process paired-end reads
        fastp_cmd_list = [
            FASTP_CMD,
            "--in1", str(sample_files[0]),
            "--in2", str(sample_files[1]),
            "--out1", str(inter_dir / f"{sample_name}_1.fq"),
            "--out2", str(inter_dir / f"{sample_name}_2.fq"),
            "--html", str(inter_dir / f"{sample_name}.html"),
            "--json", str(inter_dir / f"{sample_name}.json"),
            "--thread", str(args.cpus_per_thread),
            *fastp_args
        ]
        
        subprocess.run(fastp_cmd_list, check=True, capture_output=not args.verbose)
        
        # Merge cleaned read pairs if requested
        if args.no_merge:
            # If not merging, concatenate the cleaned files
            with open(out_fastq, "wb") as outfile:
                for infile in [
                    inter_dir / f"{sample_name}_1.fq",
                    inter_dir / f"{sample_name}_2.fq"
                ]:
                    with open(infile, "rb") as readfile:
                        outfile.write(readfile.read())
        else:
            # Merge overlapping read pairs
            subprocess.run(
                [
                    REFORMAT_CMD,
                    f"in1={inter_dir / f'{sample_name}_1.fq'}",
                    f"in2={inter_dir / f'{sample_name}_2.fq'}",
                    f"out={out_fastq}",
                    "overwrite=true",
                    "t=1"
                ],
                check=True,
                capture_output=not args.verbose
            )
            
    else:
        # Process single-end reads
        eprint(f"Sample {sample_name} has {len(sample_files)} single-end file(s). Cleaning...")
        
        fastp_cmd_list = [
            FASTP_CMD,
            "--in1", str(sample_files[0]),
            "--out1", str(out_fastq),
            "--html", str(inter_dir / f"{sample_name}.html"),
            "--json", str(inter_dir / f"{sample_name}.json"),
            "--thread", str(args.cpus_per_thread),
            *fastp_args
        ]
        
        subprocess.run(fastp_cmd_list, check=True, capture_output=not args.verbose)
    
    # If we need to deduplicate
    if not args.no_deduplicate:
        eprint(f"Deduplicating {sample_name}...")
        dedup_fastq = inter_dir / f"{sample_name}.dedup.fastq"
        
        subprocess.run(
            [
                REFORMAT_CMD,
                f"in={out_fastq}",
                f"out={dedup_fastq}",
                "dedupe",
                "overwrite=true",
                "t=1"
            ],
            check=True,
            capture_output=not args.verbose
        )
        
        out_fastq = dedup_fastq
    
    # Check if we have any reads
    try:
        n_reads, n_bp = 0, 0
        with open(out_fastq, "r") as f:
            for i, line in enumerate(f):
                if i % 4 == 1:  # Sequence line
                    n_reads += 1
                    n_bp += len(line.strip())
    except:
        eprint(f"Error reading output FASTQ file for {sample_name}. Skipping sample.")
        failed_samples.append(sample_name)
        return None, 0, 0, failed_samples
    
    if n_bp < min_bp:
        eprint(f"Sample {sample_name} has less than {args.min_bp} bp after cleaning. Skipping.")
        failed_samples.append(sample_name)
        return None, n_reads, n_bp, failed_samples
    
    eprint(f"Done cleaning {sample_name}. Got {n_reads} reads ({n_bp} bp).")
    return out_fastq, n_reads, n_bp, failed_samples


def split_fastq(fastq_path, max_bp, stats):
    """
    Split a FASTQ file to the desired size.
    
    Args:
        fastq_path (Path): Path to input FASTQ file
        max_bp (int): Maximum number of base pairs to include
        stats (dict): Statistics dictionary
        
    Returns:
        Path: Path to split FASTQ file
    """
    if max_bp is None or stats["n_bp"] <= max_bp:
        return fastq_path
    
    eprint(f"Reducing {fastq_path.name} to {max_bp} bp...")
    
    # Calculate fraction to keep
    fraction = max_bp / stats["n_bp"]
    out_path = fastq_path.parent / f"{fastq_path.stem}.split.fastq"
    
    subprocess.run(
        [
            REFORMAT_CMD,
            f"in={fastq_path}",
            f"out={out_path}",
            f"samplerate={fraction}",
            "overwrite=true",
            "t=1"
        ],
        check=True,
        capture_output=True
    )
    
    return out_path


def count_kmers(fastq_path, kmer_size, stats, args, inter_dir):
    """
    Count k-mers in a FASTQ file.
    
    Args:
        fastq_path (Path): Path to input FASTQ file
        kmer_size (int): K-mer size
        stats (dict): Statistics dictionary
        args: Command line arguments
        inter_dir (Path): Intermediate output directory
        
    Returns:
        tuple: (Path to k-mer counts file, Updated stats dict, Failed sample bool)
    """
    eprint(f"Counting {kmer_size}-mers for {fastq_path.stem}...")
    
    # Set up paths
    dsk_output = inter_dir / f"{fastq_path.stem}.h5"
    counts_file = inter_dir / f"{fastq_path.stem}.counts"
    
    trim_5, trim_3 = [int(x) for x in args.trim_bp.split(",")]
    trim_args = []
    if trim_5 > 0 or trim_3 > 0:
        trim_args.extend(["-trimLeft", str(trim_5), "-trimRight", str(trim_3)])
    
    # Count k-mers using DSK
    try:
        subprocess.run(
            [
                DSK_CMD,
                "-file", str(fastq_path),
                "-kmer-size", str(kmer_size),
                "-abundance-min", "1",
                "-verbose", "0" if not args.verbose else "1",
                "-out", str(dsk_output),
                *trim_args
            ],
            check=True,
            capture_output=not args.verbose
        )
        
        # Convert DSK output to text format
        subprocess.run(
            [
                DSK2ASCII_CMD,
                "-file", str(dsk_output),
                "-out", str(counts_file),
                "-verbose", "0" if not args.verbose else "1"
            ],
            check=True,
            capture_output=not args.verbose
        )
        
        # Calculate base frequency standard deviation
        if "basefreq_sd" not in stats:
            stats["basefreq_sd"] = get_basefrequency_sd(fastq_path)
        
        return counts_file, stats, False
        
    except Exception as e:
        eprint(f"Error counting k-mers for {fastq_path.stem}: {str(e)}")
        return None, stats, True


def make_image(kmer_counts_file, kmer_mapping, sample_name, n_bp, outdir, kmer_size, kmer_mapping_method, labels=None, basefreq_sd=None):
    """
    Create a varKode image from k-mer counts.
    
    Args:
        kmer_counts_file (Path): Path to k-mer counts file
        kmer_mapping (DataFrame): K-mer mapping table
        sample_name (str): Sample name
        n_bp (int): Number of base pairs
        outdir (Path): Output directory
        kmer_size (int): K-mer size
        kmer_mapping_method (str): K-mer mapping method
        labels (list): List of labels
        basefreq_sd (float): Base frequency standard deviation
        
    Returns:
        Path: Path to generated image
    """
    # Load k-mer counts
    kmer_counts = pd.read_csv(
        kmer_counts_file, 
        sep=" ", 
        names=["kmer", "count"], 
        dtype={"kmer": str, "count": np.int32}
    ).set_index("kmer")
    
    # Create empty image matrix
    side = 2**(kmer_size-1)
    img_matrix = np.zeros((side, side), dtype=np.uint8)
    
    # Fill image matrix with counts
    for kmer, row in kmer_mapping.iterrows():
        if kmer in kmer_counts.index:
            img_matrix[row.y, row.x] = min(255, kmer_counts.loc[kmer, "count"])
    
    # Create image
    img = Image.fromarray(img_matrix)
    
    # Create metadata
    # Format BP to use K or M notation
    if n_bp < 1000000:
        bp_str = f"{int(n_bp/1000)}K"
    else:
        bp_str = f"{int(n_bp/1000000)}M"
    
    # Create image filename
    img_filename = f"{sample_name}{SAMPLE_BP_SEP}{bp_str}{BP_KMER_SEP}{kmer_mapping_method}{BP_KMER_SEP}k{kmer_size}.png"
    
    # Add metadata to image
    metadata = PngInfo()
    if labels:
        metadata.add_text("varkoderKeywords", LABELS_SEP.join(labels))
    
    if basefreq_sd is not None:
        metadata.add_text("varkoderBaseFreqSd", str(basefreq_sd))
        if basefreq_sd > QUAL_THRESH:
            metadata.add_text("varkoderLowQualityFlag", "1")
        else:
            metadata.add_text("varkoderLowQualityFlag", "0")
    
    metadata.add_text("varkoderMapping", kmer_mapping_method)
    metadata.add_text("varkoderBp", str(n_bp))
    metadata.add_text("varkoderKmerSize", str(kmer_size))
    
    # Save image
    outdir.mkdir(parents=True, exist_ok=True)
    img_path = outdir / img_filename
    img.save(img_path, pnginfo=metadata)
    
    return img_path


def run_clean2img(row_tuple, kmer_mapping, args, np_rng, inter_dir, all_stats, stats_path, images_d, subfolder_levels=0):
    """
    Process a single sample from raw reads to image.
    
    Args:
        row_tuple (tuple): Row tuple from DataFrame
        kmer_mapping (DataFrame): K-mer mapping table
        args: Command line arguments
        np_rng: NumPy random number generator
        inter_dir (Path): Intermediate directory
        all_stats (dict): All statistics
        stats_path (Path): Path to statistics file
        images_d (Path): Output images directory
        subfolder_levels (int): Number of subfolder levels
        
    Returns:
        dict: Sample statistics
    """
    idx, row = row_tuple
    sample_name = row["sample"]
    sample_dir = inter_dir / sample_name
    
    if sample_name in all_stats:
        eprint(f"Sample {sample_name} already processed, skipping")
        return all_stats[sample_name]
    
    labels = sorted(list(row["labels"]))
    eprint(f"Processing {sample_name} ({', '.join(labels)})")
    
    # Build stats structure
    stats = OrderedDict(
        {
            "labels": ";".join(labels),
            "files": ";".join([str(f) for f in row["files"]]),
        }
    )
    
    # Clean reads
    cleaned_fastq, n_reads, n_bp, failed_samples = clean_reads(
        row["files"], sample_name, args, sample_dir
    )
    
    stats["n_reads"] = n_reads
    stats["n_bp"] = n_bp
    
    # Check if cleaning failed
    if cleaned_fastq is None:
        stats["success"] = False
        stats["failed_stage"] = "cleaning"
        all_stats[sample_name] = stats
        stats_to_csv(all_stats, stats_path)
        return stats
    
    # Split FASTQ if needed
    maxbp = humanfriendly.parse_size(args.max_bp) if args.max_bp else None
    split_fastq_path = split_fastq(cleaned_fastq, maxbp, stats)
    
    # Count k-mers
    counts_file, stats, failed = count_kmers(
        split_fastq_path, args.kmer_size, stats, args, sample_dir
    )
    
    if failed:
        stats["success"] = False
        stats["failed_stage"] = "kmers"
        all_stats[sample_name] = stats
        stats_to_csv(all_stats, stats_path)
        return stats
    
    # Generate image if requested
    if not args.no_image:
        # Create output directory structure
        if subfolder_levels > 0:
            # Generate subfolder name from the hash of the sample name
            hash_obj = hashlib.md5(sample_name.encode())
            hash_hex = hash_obj.hexdigest()
            subfolder = Path("/".join([hash_hex[i:i+2] for i in range(0, subfolder_levels*2, 2)]))
            img_dir = images_d / subfolder
        else:
            img_dir = images_d
        
        img_path = make_image(
            counts_file,
            kmer_mapping,
            sample_name,
            stats["n_bp"],
            img_dir,
            args.kmer_size,
            args.kmer_mapping,
            labels,
            stats.get("basefreq_sd")
        )
        
        stats["image"] = str(img_path)
    
    stats["success"] = True
    all_stats[sample_name] = stats
    stats_to_csv(all_stats, stats_path)
    return stats


def run_clean2img_wrapper(args_tuple):
    """
    Wrapper function for run_clean2img to use with multiprocessing.
    
    Args:
        args_tuple (tuple): Arguments tuple
        
    Returns:
        dict: Sample statistics
    """
    return run_clean2img(*args_tuple)


def process_stats(stats, condensed_files, args, stats_path, images_d, all_stats, qual_thresh, labels_sep):
    """
    Process and update statistics.
    
    Args:
        stats (dict): Sample statistics
        condensed_files (DataFrame): Files table
        args: Command line arguments
        stats_path (Path): Path to statistics file
        images_d (Path): Output images directory
        all_stats (dict): All statistics
        qual_thresh (float): Quality threshold
        labels_sep (str): Label separator
    """
    if args.label_table and "success" in stats and stats["success"] and not args.no_image:
        labels_file = images_d / "labels.csv"
        labels_table = pd.DataFrame(
            columns=["sample", "image", "labels", "bad_qual", "basefreq_sd"]
        )
        
        if labels_file.exists():
            labels_table = pd.read_csv(labels_file)
        
        if stats["sample"] not in labels_table["sample"].values:
            new_row = pd.DataFrame(
                {
                    "sample": [stats["sample"]],
                    "image": [stats["image"]],
                    "labels": [stats["labels"]],
                    "bad_qual": [bool(float(stats.get("basefreq_sd", 0)) > qual_thresh)],
                    "basefreq_sd": [float(stats.get("basefreq_sd", 0))],
                }
            )
            labels_table = pd.concat([labels_table, new_row], ignore_index=True)
            labels_table.to_csv(labels_file, index=False)


class ImageCommand:
    """
    Class for handling the image command functionality in varKoder.
    
    This class implements methods to process DNA sequences and generate
    varKode or CGR images.
    """
    
    def __init__(self, args: Any, np_rng: np.random.Generator) -> None:
        """
        Initialize ImageCommand with command line arguments and random number generator.
        
        Args:
            args: Parsed command line arguments
            np_rng: NumPy random number generator
        """
        self.args = args
        self.np_rng = np_rng
        self.all_stats = defaultdict(OrderedDict)
        
        # Validate input parameters
        if args.kmer_size not in range(5, 9 + 1):
            raise ValueError("kmer size must be between 5 and 9")
            
        # Set up intermediate directory
        try:
            self.inter_dir = Path(args.int_folder)
        except TypeError:
            self.inter_dir = Path(tempfile.mkdtemp(prefix="barcoding_"))
            
        # Check if output directory exists
        if not args.overwrite and Path(args.outdir).exists():
            raise Exception("Output directory exists, use --overwrite if you want to overwrite it.")
        else:
            if Path(args.outdir).is_dir():
                shutil.rmtree(Path(args.outdir))
                
        # Set directory to save images
        self.images_d = Path(args.outdir)
        
        # Set up kmer mapping
        self.kmer_mapping = get_kmer_mapping(args.kmer_size, args.kmer_mapping)
        
        # Initialize stats
        self.stats_path = Path(args.stats_file)
        if self.stats_path.exists():
            self.all_stats.update(read_stats(self.stats_path))
    
    def process_samples(self, condensed_files: pd.DataFrame) -> None:
        """
        Process all samples to generate images.
        
        Args:
            condensed_files: DataFrame with file information
        """
        # Check if we need multiple subfolder levels
        # This will ensure we have about 1000 samples per subfolder
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
    
    def run(self) -> None:
        """
        Run the image command.
        """
        eprint("varKoder")
        eprint("Kmer size:", str(self.args.kmer_size))
        eprint("Processing reads and preparing images")
        eprint("Reading input data")
        
        # Parse input and create a table relating reads files to samples and taxa
        inpath = Path(self.args.input)
        condensed_files = process_input(inpath, is_query=False)
        
        if condensed_files.shape[0] == 0:
            raise Exception("No files found in input. Please check.")
        
        # Process samples and generate images
        self.process_samples(condensed_files)
        
        eprint("All images done, saved in", str(self.images_d))
        
        # Clean up temporary directory if created
        if not self.args.int_folder and self.inter_dir.is_dir():
            shutil.rmtree(self.inter_dir)


def run_image_command(args: Any, np_rng: np.random.Generator) -> None:
    """
    Run the image command with the given arguments.
    
    This is the main entry point for the image command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
        np_rng: NumPy random number generator
    """
    image_cmd = ImageCommand(args, np_rng)
    image_cmd.run()