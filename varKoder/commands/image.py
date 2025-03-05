#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image command module for varKoder.

This module contains functionality for generating varKode images from DNA sequences.
It handles the preprocessing of reads, k-mer counting, and image generation.
"""

import os
import re
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
from io import StringIO
from tqdm import tqdm
from functools import partial
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
import json
import hashlib
import gzip
import itertools
import subprocess
import traceback

from varKoder.core.config import (
    LABELS_SEP, BP_KMER_SEP, SAMPLE_BP_SEP, QUAL_THRESH, 
    MAPPING_CHOICES, DEFAULT_KMER_SIZE, DEFAULT_KMER_MAPPING,
    FASTP_CMD, DSK_CMD, DSK2ASCII_CMD, REFORMAT_CMD, PIGZ_CMD,
    LABEL_SAMPLE_SEP
)
from varKoder.core.utils import (
    eprint, get_kmer_mapping, process_input, stats_to_csv, read_stats
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from concurrent.futures import ThreadPoolExecutor

# ORIGINAL IMPLEMENTATION OF FUNCTIONS FROM functions.py
def get_basefrequency_sd(file_list):
    """
    This function reads the json file produced by fastp within the clean_reads function
    and returns the standard deviation in base frequencies from positions 1-40 in forward reads.
    This is expected to be 0 for high-quality samples but increases in low-quality ones.
    
    Args:
        file_list: List of fastp JSON files
        
    Returns:
        float: Standard deviation of base frequencies
    """
    base_sd = []

    for f in file_list:
        js = json.load(open(f, "r"))
        try:
            content = np.array(
                [
                    x
                    for k, x in js["merged_and_filtered"]["content_curves"].items()
                    if k in ["A", "T", "C", "G"]
                ]
            )
            base_sd.append(np.std(content[:, 5:40], axis=1).mean())
        except KeyError:
            pass
        try:
            content = np.array(
                [
                    x
                    for k, x in js["read1_after_filtering"]["content_curves"].items()
                    if k in ["A", "T", "C", "G"]
                ]
            )
            base_sd.append(np.std(content[:, 5:40], axis=1).mean())
        except KeyError:
            pass

        return np.mean(base_sd)

def estimate_read_lengths(filename, sample_size=10000):
    """
    Estimate average read length by sampling the first sample_size reads from a file.
    
    Args:
        filename: Path to the FASTQ file (can be gzipped)
        sample_size: Number of reads to sample for estimation
    
    Returns:
        float: Average read length
        int: Total number of reads sampled
    """
    
    total_length = 0
    reads_counted = 0
    
    opener = gzip.open if str(filename).endswith('gz') else open
    with opener(filename, 'rb') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # Sequence line
                total_length += len(line.strip())
                reads_counted += 1
            if reads_counted >= sample_size:
                break
    
    return total_length / reads_counted if reads_counted > 0 else 0, reads_counted

def count_total_reads(filename):
    """Count total number of reads in a FASTQ file using system tools for efficiency."""
    
    # Convert to absolute path
    abs_path = str(Path(filename).resolve())
    
    try:
        if str(filename).endswith('gz'):
            # Try using gunzip -c instead of zcat for better compatibility
            cmd = f"gunzip -c '{abs_path}' | wc -l"
        else:
            cmd = f"wc -l '{abs_path}'"
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=True  # This will raise an exception if the command fails
        )
        
        line_count = int(result.stdout.strip().split()[0])  # Handle potential leading spaces
        read_count = line_count // 4
        
        
        if read_count == 0:
            # Double check using Python if system command returns 0
            with gzip.open(abs_path, 'rb') if filename.endswith('gz') else open(abs_path, 'rb') as f:
                line_count = sum(1 for _ in f)
                read_count = line_count // 4
                eprint(f"Python count found {read_count} reads")
        
        return read_count
        
    except Exception as e:
        eprint(f"Error counting reads in {filename}: {str(e)}")
        eprint("Falling back to Python-based counting...")
        
        try:
            # Fallback to pure Python counting if system command fails
            with gzip.open(abs_path, 'rb') if str(filename).endswith('gz') else open(abs_path, 'rb') as f:
                line_count = sum(1 for _ in f)
                return line_count // 4
        except Exception as e:
            eprint(f"Failed to count reads: {str(e)}")
            raise

def calculate_reads_needed(files_info, max_bp):
    """
    Calculate how many reads to take from each file to achieve target coverage.
    If max_bp is None, returns all available reads from each file.
    
    Args:
        files_info: Dictionary containing file information including average read lengths
        max_bp: Target base pair count. If None, all available reads will be returned
    
    Returns:
        Dictionary with number of reads to take from each file
    """
    reads_to_take = {}
    
    # Early return case: when max_bp is None, return all available reads
    if max_bp is None:
        # Process unpaired files first
        for file_info in files_info['unpaired']:
            reads_to_take[file_info['file']] = file_info['total_reads']
            
        # Process paired files (R1 and R2)
        for r1_info, r2_info in zip(files_info['R1'], files_info['R2']):
            # For paired files, take all reads available while ensuring we don't take
            # more reads than are available in either file of the pair
            max_possible_pairs = min(r1_info['total_reads'], r2_info['total_reads'])
            reads_to_take[r1_info['file']] = max_possible_pairs
            reads_to_take[r2_info['file']] = max_possible_pairs
            
        return reads_to_take
    
    # If max_bp is a number, calculate reads to take
    remaining_bp = 5 * max_bp  # We want 5x coverage
    
    # First allocate from unpaired files
    for file_info in files_info['unpaired']:
        bp_per_read = file_info['avg_length']
        max_possible_reads = file_info['total_reads']
        reads_needed = min(
            max_possible_reads,
            int(remaining_bp / bp_per_read)
        )
        reads_to_take[file_info['file']] = reads_needed
        remaining_bp -= reads_needed * bp_per_read
    
    # Then allocate from paired files (R1 and R2)
    if remaining_bp > 0:
        for r1_info, r2_info in zip(files_info['R1'], files_info['R2']):
            bp_per_pair = r1_info['avg_length'] + r2_info['avg_length']
            max_possible_pairs = min(r1_info['total_reads'], r2_info['total_reads'])
            pairs_needed = min(
                max_possible_pairs,
                int(remaining_bp / bp_per_pair)
            )
            reads_to_take[r1_info['file']] = pairs_needed
            reads_to_take[r2_info['file']] = pairs_needed
            remaining_bp -= pairs_needed * bp_per_pair
    
    return reads_to_take

def extract_reads(filename, num_reads, output_file):
    """
    Extract a specified number of reads from a FASTQ file using pure Python.
    This function handles both gzipped and plain text files efficiently.
    
    Args:
        filename: Path to input FASTQ file (can be gzipped)
        num_reads: Number of reads to extract
        output_file: Path to output file where reads will be written
    """
    
    # Convert to absolute paths for clarity and consistency
    abs_input = str(Path(filename).resolve())
    abs_output = str(Path(output_file).resolve())
    
    # Calculate number of lines to extract (4 lines per read in FASTQ format)
    num_lines = num_reads * 4
    
    try:
        # Open input file with appropriate opener (gzip or regular)
        opener = gzip.open if str(filename).endswith('gz') else open
        with opener(abs_input, 'rb') as infile, open(abs_output, 'ab') as outfile:
            # Use itertools.islice for memory-efficient iteration
            for line in itertools.islice(infile, num_lines):
                # Write each line to the output file
                # Note: we're preserving the original line endings here
                outfile.write(line)
        
        # Verify the output file was created and contains data
        if not Path(output_file).exists():
            raise Exception(f"Failed to create output file: {output_file}")
        
        file_size = Path(output_file).stat().st_size
        if file_size == 0:
            raise Exception(f"Output file is empty: {output_file}")
            
        
    except Exception as e:
        eprint(f"Error during read extraction: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling function

def concatenate_reads(reads, basename, max_bp, work_dir, verbose):
    """
    Main function to concatenate reads with optimized sampling approach.
    Returns an integer count of total base pairs.
    """
    # First, gather information about all input files
    files_info = {'unpaired': [], 'R1': [], 'R2': []}
    
    for category in ['unpaired', 'R1', 'R2']:
        for readf in sorted(reads[category]):
            # Round the average length to the nearest integer since we can't have fractional base pairs
            avg_length, _ = estimate_read_lengths(readf)
            avg_length = round(avg_length)  # Convert to integer
            total_reads = count_total_reads(readf)
            if verbose: eprint(f"File: {readf}")
            if verbose: eprint(f"Average read length (rounded): {avg_length}")
            if verbose: eprint(f"Total reads: {total_reads}")
            
            files_info[category].append({
                'file': readf,
                'avg_length': avg_length,  # Now storing as integer
                'total_reads': total_reads
            })
    
    # Calculate how many reads we need from each file
    reads_to_take = calculate_reads_needed(files_info, max_bp)
    if verbose: eprint(f"Reads to take from each file: {reads_to_take}")
    
    # Create output files
    output_files = {
        'unpaired': Path(work_dir) / f"{basename}_unpaired.fq",
        'R1': Path(work_dir) / f"{basename}_R1.fq",
        'R2': Path(work_dir) / f"{basename}_R2.fq"
    }
    
    # Clear output files if they exist
    for outfile in output_files.values():
        outfile.unlink(missing_ok=True)
    
    # Extract and concatenate reads
    total_bp = 0  # This will now be an integer
    for category in ['unpaired', 'R1', 'R2']:
        for file_info in files_info[category]:
            readf = file_info['file']
            if readf in reads_to_take:
                num_reads = reads_to_take[readf]
                if num_reads > 0:
                    extract_reads(readf, num_reads, output_files[category])
                    # Calculate exact integer number of base pairs
                    total_bp += num_reads * file_info['avg_length']
    
    return total_bp

def clean_reads(
    infiles,
    outpath,
    cut_adapters=True,
    merge_reads=True,
    deduplicate=True,
    trim_bp=(0,0),
    max_bp=None,
    threads=1,
    overwrite=False,
    verbose=False,
):
    """
    Cleans illumina reads and saves results as a single merged fastq file to outpath.
    Cleaning includes:
    1 - adapter removal
    2 - deduplication
    3 - merging
    
    Args:
        infiles: List of input FASTQ files
        outpath: Path to output FASTQ file
        cut_adapters: Whether to remove adapters
        merge_reads: Whether to merge paired-end reads
        deduplicate: Whether to deduplicate reads
        trim_bp: Number of base pairs to trim from start and end
        max_bp: Maximum number of base pairs to use
        threads: Number of threads to use
        overwrite: Whether to overwrite existing files
        verbose: Whether to print verbose output
        
    Returns:
        OrderedDict: Statistics dictionary
    """
    start_time = pd.Timestamp.now()

    # let's print a message to the user
    basename = Path(outpath).name.removesuffix("".join(Path(outpath).suffixes))

    if not overwrite and outpath.is_file():
        eprint("Skipping cleaning for", basename + ":", "File exists.")
        return OrderedDict()

    eprint("Finding and concatenating files for", basename)

    # let's start separating forward, reverse and unpaired reads
    re_pair = {
        "R1": re.compile(r"(?<=[_R\.])1(?=[_\.])"),
        "R2": re.compile(r"(?<=[_R\.])2(?=[_\.])"),
    }

    reads = {
        "R1": [str(x) for x in infiles if re_pair["R1"].search(str(x)) is not None],
        "R2": [str(x) for x in infiles if re_pair["R2"].search(str(x)) is not None],
    }
    reads["unpaired"] = [
        str(x) for x in infiles if str(x) not in reads["R1"] + reads["R2"]
    ]

    # sometimes we will have sequences with names that look paired but they do not have a pair, so let's check for this:
    for r in reads["R1"]:
        if re_pair["R1"].sub("2", r) not in reads["R2"]:
            reads["unpaired"].append(r)
            del reads["R1"][reads["R1"].index(r)]
    for r in reads["R2"]:
        if re_pair["R2"].sub("1", r) not in reads["R1"]:
            reads["unpaired"].append(r)
            del reads["R2"][reads["R2"].index(r)]

    eprint("Concatenating input files:", reads)

    # from now on we will manipulate files in a temporary folder that will be deleted at the end of this function
    work_dir = tempfile.mkdtemp(prefix="barcoding_clean_" + basename)

    retained_bp = concatenate_reads(reads, basename, max_bp, work_dir, verbose)
    

    act_list = []
    extra_command = []
    # add arguments depending on adapter option
    if cut_adapters:
        act_list.append("remove adapters")
        extra_command.extend(["--detect_adapter_for_pe"])
    else:
        extra_command.extend(["--disable_adapter_trimming"])

    # add arguments depending on merge option
    if merge_reads:
        extra_command.extend(["--merge", "--include_unmerged"])
        act_list.append("merge reads")

    # add arguments depending on dedup option
    if deduplicate:
        act_list.append("remove duplicates")
        extra_command.extend(["--dedup", "--dup_calc_accuracy", "1"])
    else:
        extra_command.extend(["--dont_eval_duplication"])
    eprint(f"Preprocessing {basename}: {', '.join(act_list)}. Trimming (front,tail): {trim_bp}")



    # now we can run fastp to remove adapters, duplicates and merge paired reads
    if (Path(work_dir) / (basename + "_R1.fq")).is_file():
        # let's build the call to the subprocess. This is the common part
        # for some reason, fastp fails with interleaved input unless it is provided from stdin
        # for this reason, we will make a pipe
        command = [
            FASTP_CMD,
            "--in1",
            str(Path(work_dir) / (basename + "_R1.fq")),
            "--in2",
            str(Path(work_dir) / (basename + "_R2.fq")),
            "--disable_quality_filtering",
            "--disable_length_filtering",
            "--trim_poly_g",
            "--thread",
            str(threads),
            "--html",
            str(Path(work_dir) / (basename + "_fastp_paired.html")),
            "--json",
            str(Path(work_dir) / (basename + "_fastp_paired.json")),
            "--stdout",
            "--trim_front1",
            str(trim_bp[0]),
            "--trim_tail1",
            str(trim_bp[1]),
        ] + extra_command

        try:
            with open(Path(work_dir) / (basename + "_clean_paired.fq"), "wb") as outf:
                p = subprocess.run(
                    command,
                    check=True,
                    stderr=subprocess.PIPE,
                    stdout=outf,
                )
                if verbose:
                    eprint(' '.join(command))
                    eprint(p.stderr.decode())
                (Path(work_dir) / (basename + "_paired.fq")).unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            eprint(f"{basename}: fastp failed with paired reads, treating them as unpaired")
            if verbose:
                eprint(' '.join(command))
                eprint(e.stderr.decode())
                traceback.print_exc()

            with open((Path(work_dir) / (basename + "_unpaired.fq")),"a") as unpair_f:
                with open(Path(work_dir) / (basename + "_R1.fq"),"r") as pair_f:
                    for line in pair_f:
                        unpair_f.write(line)
                with open(Path(work_dir) / (basename + "_R2.fq"),"r") as pair_f:
                    for line in pair_f:
                        unpair_f.write(line)


    # and remove adapters from unpaired reads, if any
    if (Path(work_dir) / (basename + "_unpaired.fq")).is_file():
        #remove merge commands
        extra_command = [x for x in extra_command if x not in ["--merge", "--include_unmerged"]]

        command = [
            FASTP_CMD,
            "-i",
            str(Path(work_dir) / (basename + "_unpaired.fq")),
            "-o",
            str(Path(work_dir) / (basename + "_clean_unpaired.fq")),
            "--html",
            str(Path(work_dir) / (basename + "_fastp_unpaired.html")),
            "--json",
            str(Path(work_dir) / (basename + "_fastp_unpaired.json")),
            "--disable_quality_filtering",
            "--disable_length_filtering",
            "--trim_poly_g",
            "--thread",
            str(threads),
            "--trim_front1",
            str(trim_bp[0]),
            "--trim_tail1",
            str(trim_bp[1]),
        ] + extra_command

        try:
            p = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
            )

            if verbose:
                eprint(' '.join(command))
                eprint(p.stderr.decode())

        except subprocess.CalledProcessError as e:
            eprint(f"{basename}: fastp failed with unpaired reads")
            if verbose:
                eprint(' '.join(command))
                eprint(e.stderr.decode())
                traceback.print_exc()
            raise

        (Path(work_dir) / (basename + "_unpaired.fq")).unlink(missing_ok=True)

    # now that we finished cleaning, let's compress and move the final file to their destination folder
    # and assemble statistics
    # move files

    # create output folders if they do not exist

    try:
        outpath.parent.mkdir(parents=True)
    except OSError:
        pass

    command = ["cat"]
    command.extend([str(x) for x in Path(work_dir).glob("*_clean_*.fq")])

    with open(outpath, "wb") as outf:
        cat = subprocess.Popen(command, stdout=subprocess.PIPE)
        p = subprocess.run(
            ["pigz", "-p", str(threads)],
            stdin=cat.stdout,
            stdout=outf,
            stderr=subprocess.PIPE,
            check=True,
        )
        if verbose:
            eprint(' '.join(command))
            eprint(p.stderr.decode())

    # copy fastp reports
    for fastp_f in Path(work_dir).glob("*fastp*"):
        shutil.copy(fastp_f, outpath.parent / str(fastp_f.name))

    # stats: numbers of basepairs in each step (we recorded initial bp already when concatenating)
    clean = Path(work_dir).glob("*_clean_*.fq")
    if cut_adapters or merge_reads:
        clean_bp = 0
        for cl in clean:
            with open(cl, "rb") as infile:
                BUF_SIZE = 100000000
                tmp_lines = infile.readlines(BUF_SIZE)
                line_n = 0
                while tmp_lines:
                    for line in tmp_lines:
                        if line_n % 4 == 1:
                            clean_bp += len(line) - 1
                        line_n += 1
                    tmp_lines = infile.readlines(BUF_SIZE)
    else:
        clean_bp = np.nan

    stats = OrderedDict()
    done_time = pd.Timestamp.now()
    stats["clean_basepairs"] = clean_bp
    stats["cleaning_time"] = (done_time - start_time).total_seconds()

    shutil.rmtree(work_dir)
    ####END OF TEMPDIR BLOCK

    return stats

def run_parallel_reformats(sites_per_file, outfs, infile, seed, verbose=False, max_workers=None):
    """Helper function to parallelize reformat.sh execution."""
    # Create list of commands and their arguments
    commands = []
    for i, bp in enumerate(sites_per_file):
        command = [
            REFORMAT_CMD,
            "samplebasestarget=" + str(bp),
            "sampleseed=" + str(int(seed) + i),
            "breaklength=500",
            "ignorebadquality=t",
            "quantize=t", 
            "iupacToN=t",
            "qin=33",
            "in=" + str(infile),
            "out=" + str(outfs[i]),
            "overwrite=true",
            "verifypaired=f",
            "int=f"
        ]
        commands.append((command, i))

    def run_single_command(args):
        command, i = args
        try:
            p = subprocess.run(
                command,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL, 
                check=True,
            )
            if verbose:
                eprint(' '.join(command))
                if p.stderr:
                    eprint(p.stderr.decode())
            return True
        except subprocess.CalledProcessError as e:
            if verbose:
                eprint(f"Error in reformat.sh subprocess {i}:", ' '.join(command))
                if e.stderr:
                    eprint(e.stderr.decode())
            return False

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs and wait for completion
        results = list(executor.map(run_single_command, commands))

    # Check if all processes succeeded
    if not all(results):
        raise RuntimeError("One or more reformat processes failed")

def split_fastq(
    infile,
    outprefix,
    outfolder,
    min_bp=50000,
    max_bp=None,
    is_query=False,
    seed=None,
    overwrite=False,
    verbose=False,
    n_threads=1
):
    """
    Split a FASTQ file into multiple files with different numbers of reads.
    
    Args:
        infile: Path to input FASTQ file
        outprefix: Prefix for output files
        outfolder: Output directory
        min_bp: Minimum number of base pairs per file
        max_bp: Maximum number of base pairs per file
        is_query: Whether this is a query operation
        seed: Random seed
        overwrite: Whether to overwrite existing files
        verbose: Whether to print verbose output
        n_threads: Number of threads to use
        
    Returns:
        OrderedDict: Statistics dictionary
    """
    start_time = pd.Timestamp.now()

    # let's start by counting the number of sites in reads of the input file
    sites_seq = []
    with gzip.open(infile, "rb") as f:
        for nlines, l in enumerate(f):
            if nlines % 4 == 1:
                sites_seq.append(len(l) - 1)
    nsites = sum(sites_seq)

    # now let's make a list that stores the number of sites we will retain in each file
    if max_bp is None:
        sites_per_file = [int(nsites)]
    elif is_query or int(nsites) > min_bp:
        sites_per_file = [min(int(nsites), int(max_bp))]
    else:
        eprint( "Post-cleaning input file "
            + str(infile)
            + " has less than "
            + str(min_bp)
            + "bp, decrease --min_bp if you want to produce an image.")
        raise Exception("Input file has less than minimum data.")

    if not is_query:
        while sites_per_file[-1] > min_bp:
            oneless = sites_per_file[-1] - 1
            nzeros = int(math.log10(oneless))
            first_digit = int(oneless / (10**nzeros))

            if first_digit in [1, 2, 5]:
                sites_per_file.append(first_digit * (10 ** (nzeros)))
            else:
                multiplier = max([x for x in [1, 2, 5] if x < first_digit])
                sites_per_file.append(multiplier * (10 ** (nzeros)))

        if sites_per_file[-1] < min_bp:
            del sites_per_file[-1]


    # now we will use bbtools reformat.sh to subsample
    outfs = [
        Path(outfolder)
        / (
            outprefix
            + SAMPLE_BP_SEP
            + str(int(bp / 1000)).rjust(8, "0")
            + "K"
            + ".fq.gz"
        )
        for bp in sites_per_file
    ]

    if all([f.is_file() for f in outfs]):
        if not overwrite:
            eprint("Files exist. Skipping subsampling for file:", str(infile))
            return OrderedDict()
    
    run_parallel_reformats(sites_per_file, outfs, infile, seed, verbose=verbose, max_workers=n_threads)

    done_time = pd.Timestamp.now()

    stats = OrderedDict()

    stats["splitting_time"] = (done_time - start_time).total_seconds()
    stats["splitting_bp_per_file"] = ",".join([str(x) for x in sites_per_file])

    return stats

def count_kmers(
    infile, 
    outfolder, 
    threads=1, 
    k=7, 
    overwrite=False, 
    verbose=False
):
    """
    Count k-mers in a FASTQ file.
    
    Args:
        infile: Path to input FASTQ file
        outfolder: Output directory
        threads: Number of threads to use
        k: K-mer size
        overwrite: Whether to overwrite existing files
        verbose: Whether to print verbose output
        
    Returns:
        OrderedDict: Statistics dictionary
    """
    start_time = pd.Timestamp.now()

    Path(outfolder).mkdir(exist_ok=True)
    outfile = (
        str(Path(infile).name.removesuffix("".join(Path(infile).suffixes)))
        + BP_KMER_SEP
        + "k"
        + str(k)
        + ".fq.h5"
    )
    outpath = outfolder / outfile

    if not overwrite and outpath.is_file():
        eprint("File exists. Skipping kmer counting for file:", str(infile))
        return OrderedDict()

    with tempfile.TemporaryDirectory(prefix="dsk") as tempdir:
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, max=60),
        ):
            with attempt:
                command = [
                        DSK_CMD,
                        "-nb-cores",
                        str(threads),
                        "-kmer-size",
                        str(k),
                        "-abundance-min",
                        "1",
                        "-abundance-min-threshold",
                        "1",
                        "-max-memory",
                        "1000",
                        "-file",
                        str(infile),
                        "-out-tmp",
                        str(tempdir),
                        #'-out-dir', str(outfolder),
                        "-out",
                        str(outpath),
                    ]
                p = subprocess.run(
                    command,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                if verbose:
                    eprint(' '.join(command))
                    eprint(p.stderr.decode())

    done_time = pd.Timestamp.now()

    stats = OrderedDict()
    stats[str(k) + "mer_counting_time"] = (done_time - start_time).total_seconds()

    return stats

def make_image(
    infile,
    outfolder,
    kmer_mapping,
    threads=1,
    overwrite=False,
    verbose=False,
    labels=[],
    base_sd=0,
    base_sd_thresh=QUAL_THRESH,
    subfolder_levels=0,
    mapping_code='varKode',
):
    """
    Create an image from k-mer counts.
    
    Args:
        infile: Path to k-mer counts file
        outfolder: Output directory
        kmer_mapping: K-mer mapping table
        threads: Number of threads to use
        overwrite: Whether to overwrite existing files
        verbose: Whether to print verbose output
        labels: List of labels
        base_sd: Base frequency standard deviation
        base_sd_thresh: Base frequency standard deviation threshold
        subfolder_levels: Number of subfolder levels
        mapping_code: K-mer mapping method code
        
    Returns:
        OrderedDict: Statistics dictionary
    """
    in_basename = str(Path(infile).name.removesuffix("".join(Path(infile).suffixes)))
    in_base1, in_k = in_basename.split(BP_KMER_SEP)

    outfile = (in_base1 +
               BP_KMER_SEP +
               mapping_code +
               BP_KMER_SEP +
               in_k +
               ".png"
              )
    if subfolder_levels:
        hsh = list(hashlib.md5(outfile.encode("UTF-8")).hexdigest())
        for i in range(subfolder_levels):
            outfolder=outfolder/hsh.pop()
    Path(outfolder).mkdir(exist_ok=True, parents=True)


    if not overwrite and (outfolder / outfile).is_file():
        eprint("File exists. Skipping image for file:", str(infile))
        return OrderedDict()

    start_time = pd.Timestamp.now()
    kmer_size = len(kmer_mapping.index[0])

    with tempfile.TemporaryDirectory(prefix="dsk") as outdir:
        # first, dump dsk results as ascii, save in a pandas df and merge with mapping
        # mapping has kmers and their reverse complements, so we need to aggregate
        # to get only canonical kmers
        # when running in parallel, sometimes dsk fails, hard to know the reason
        # so we retry a few times before admitting defeat
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, max=60),
        ):
            with attempt:
                command = [
                        DSK2ASCII_CMD,
                        "-c",
                        "-file",
                        str(infile),
                        "-nb-cores",
                        str(threads),
                        "-out",
                        str(Path(outdir) / "dsk.txt"),
                        "-verbose",
                        "0",
                    ]
                dsk_out = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        if verbose:
            eprint(' '.join(command))
            eprint(dsk_out.stderr.decode())
            
        dsk_out = dsk_out.stdout.decode("UTF-8")
        counts = pd.read_csv(
            StringIO(dsk_out), sep=" ", names=["sequence", "count"], index_col=0
        )
        counts = kmer_mapping.join(counts).groupby(["x", "y"]).agg("mean").reset_index()


        counts.loc[:, "count"] = counts["count"].fillna(0)

        # Now we will place counts in an array, log the counts and rescale to use 8 bit integers
        array_width = kmer_mapping["x"].max() + 1
        array_height = kmer_mapping["y"].max() + 1

        # Now let's create the image:
        kmer_array = np.zeros(shape=[array_height, array_width])
        kmer_array[counts["x"],counts["y"]] = (counts["count"] + 1)  # we do +1 so empty cells are different from zero-count
        kmer_array = kmer_array.transpose() #PIL images have flipped x and y coords
        kmer_array = np.flip(kmer_array,0) #In PIL images, 0 is top in vertical axis (axis 0 after transposed)

        
        bins = np.quantile(kmer_array, np.arange(0, 1, 1 / 256))
        kmer_array = np.digitize(kmer_array, bins, right=False) - 1
        
        kmer_array = np.uint8(kmer_array)
        img = Image.fromarray(kmer_array, mode="L")

        # Now let's add the labels and other metadata:
        metadata = PngInfo()
        metadata.add_text("varkoderKeywords", LABELS_SEP.join(labels))
        metadata.add_text("varkoderBaseFreqSd", str(base_sd))
        metadata.add_text("varkoderLowQualityFlag", str(base_sd > base_sd_thresh))
        metadata.add_text("varkoderMapping", mapping_code)

        # finally, save the image
        img.save(Path(outfolder) / outfile, optimize=True, pnginfo=metadata)

    done_time = pd.Timestamp.now()
    stats = OrderedDict()
    stats["k" + str(kmer_size) + "_img_time"] = (done_time - start_time).total_seconds()

    return stats

def run_clean2img(
    it_row,
    kmer_mapping,
    args,
    np_rng,
    inter_dir,
    all_stats,
    stats_path,
    images_d,
    subfolder_levels=0,
):
    """
    Process a single sample from raw reads to image.
    
    Args:
        it_row: Row tuple from DataFrame
        kmer_mapping: K-mer mapping table
        args: Command line arguments
        np_rng: NumPy random number generator
        inter_dir: Intermediate directory
        all_stats: All statistics
        stats_path: Path to statistics file
        images_d: Output images directory
        subfolder_levels: Number of subfolder levels
        
    Returns:
        dict: Sample statistics
    """
    cores_per_process = args.cpus_per_thread

    x = it_row[1]
    stats = defaultdict(OrderedDict)

    clean_reads_f = inter_dir / "clean_reads" / (x["sample"] + ".fq.gz")
    split_reads_d = inter_dir / "split_fastqs"
    kmer_counts_d = inter_dir / (str(args.kmer_size) + "mer_counts")

    #### STEP B - clean reads and merge all files for each sample
    try:
        maxbp = int(humanfriendly.parse_size(args.max_bp))
    except:
        maxbp = None

    try:
        clean_stats = clean_reads(
            infiles=x["files"],
            outpath=clean_reads_f,
            cut_adapters=args.no_adapter is False,
            merge_reads=args.no_merge is False,
            deduplicate=args.no_deduplicate is False,
            trim_bp=args.trim_bp.split(','),
            max_bp=maxbp,
            threads=cores_per_process,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    except Exception as e:
        eprint("CLEAN FAIL:", x["files"])
        if args.verbose:
            eprint(e)
            traceback.print_exc()
        eprint("SKIPPING SAMPLE")
        stats[str(x["sample"])].update({"failed_step": "clean"})
        return stats

    stats[str(x["sample"])].update(clean_stats)

    #### STEP C - split clean reads into files with different number of reads
    eprint("Splitting fastqs for", x["sample"])
    if args.command == "image":
        try:
            split_stats = split_fastq(
                infile=clean_reads_f,
                outprefix=x["sample"],
                outfolder=split_reads_d,
                min_bp=humanfriendly.parse_size(args.min_bp),
                max_bp=maxbp,
                overwrite=args.overwrite,
                verbose=args.verbose,
                seed=str(it_row[0]) + str(np_rng.integers(low=0, high=2**32)),
                n_threads=cores_per_process
            )
        except Exception as e:
            eprint("SPLIT FAIL:", clean_reads_f)
            if args.verbose:
                eprint(e)
                traceback.print_exc()
            eprint("SKIPPING SAMPLE")
            stats[str(x["sample"])].update({"failed_step": "split"})
            return stats
    elif args.command == "query":
        try:
            split_stats = split_fastq(
                infile=clean_reads_f,
                outprefix=x["sample"],
                outfolder=split_reads_d,
                is_query=True,
                max_bp=maxbp,
                overwrite=args.overwrite,
                verbose=args.verbose,
                seed=str(it_row[0]) + str(np_rng.integers(low=0, high=2**32)),
                n_threads = cores_per_process
            )
        except Exception as e:
            eprint("SPLIT FAIL:", clean_reads_f)
            if args.verbose:
                eprint(e)
                traceback.print_exc()
            eprint("SKIPPING SAMPLE")
            stats[str(x["sample"])].update({"failed_step": "split"})
            return stats

    stats[str(x["sample"])].update(split_stats)

    eprint("Cleaning and splitting reads done for", x["sample"])

    #### STEP D - count kmers
    if args.command == "query" or not args.no_image:
        eprint("Counting kmers and creating images for", x["sample"])
        stats[str(x["sample"])][str(args.kmer_size) + "mer_counting_time"] = 0

        kmer_key = str(args.kmer_size) + "mer_counting_time"
        for infile in split_reads_d.glob(x["sample"] + SAMPLE_BP_SEP + "*"):
            try:
                count_stats = count_kmers(
                    infile=infile,
                    outfolder=kmer_counts_d,
                    threads=cores_per_process,
                    k=args.kmer_size,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )
            except Exception as e:
                eprint("K-MER COUNTING FAIL, SKIPPING FILE:", infile)
                if args.verbose:
                    eprint(e)
                    traceback.print_exc()
                continue

            try:
                stats[str(x["sample"])][kmer_key] += count_stats[kmer_key]
            except KeyError as e:
                if e.args[0] == kmer_key:
                    pass
                else:
                    raise (e)

        #### STEP E - create images
        # the table mapping canonical kmers to pixels is stored as a feather file in
        # the same folder as this script

        img_key = "k" + str(args.kmer_size) + "_img_time"

        stats[str(x["sample"])][img_key] = 0
        for infile in kmer_counts_d.glob(x["sample"] + SAMPLE_BP_SEP + "*"):
            base_sd = get_basefrequency_sd(
                Path(inter_dir, "clean_reads").glob(str(x["sample"]) + "_fastp_*.json")
            )
            stats[str(x["sample"])]["base_frequencies_sd"] = base_sd
            
            try:
                img_stats = make_image(
                    infile=infile,
                    outfolder=images_d,
                    kmer_mapping=kmer_mapping,
                    overwrite=args.overwrite,
                    threads=cores_per_process,
                    verbose=args.verbose,
                    labels=x["labels"],
                    base_sd=base_sd,
                    subfolder_levels=subfolder_levels,
                    mapping_code = args.kmer_mapping
                )
            except (IndexError, pd.errors.ParserError) as e:
                eprint("IMAGE FAIL:", infile)
                if args.verbose:
                    eprint(e)
                    traceback.print_exc()
                eprint("SKIPPING IMAGE")
                stats[str(x["sample"])].update({"failed_step": "image"})
                continue
            try:
                stats[str(x["sample"])][img_key] += img_stats[img_key]
            except KeyError as e:
                if e.args[0] == img_key:
                    pass
                else:
                    raise (e)

        eprint("Images done for", x["sample"])
    return stats


def run_clean2img_wrapper(args_tuple):
    """
    Wrapper function for run_clean2img to use with multiprocessing.
    
    Args:
        args_tuple: Arguments tuple
        
    Returns:
        dict: Sample statistics
    """
    return run_clean2img(*args_tuple)


def process_stats(stats, condensed_files, args, stats_path, images_d, all_stats, qual_thresh, labels_sep):
    """
    Process and update statistics.
    
    Args:
        stats: Sample statistics
        condensed_files: Files table
        args: Command line arguments
        stats_path: Path to statistics file
        images_d: Output images directory
        all_stats: All statistics
        qual_thresh: Quality threshold
        labels_sep: Label separator
    """
    # Check if stats.csv exists
    if stats_path.exists():
        try:
            all_stats.update(read_stats(stats_path))
        except Exception as e:
            eprint(f"Error updating stats: {e}")
    else:
        eprint(f"'{stats_path}' not found. Initializing empty stats.")

    for k in stats.keys():
        all_stats[str(k)].update(stats[k])

    stats_df = stats_to_csv(all_stats, stats_path)

    if args.command == "image" and args.label_table:
        merged_data = (
            condensed_files.merge(
                stats_df[["sample", "base_frequencies_sd"]], how="left", on="sample"
            )
            .assign(
                possible_low_quality=lambda x: x["base_frequencies_sd"] > qual_thresh
            )
            .assign(labels=lambda x: x["labels"].apply(lambda y: labels_sep.join(y)))
        )
        merged_data[["sample", "labels", "possible_low_quality"]].to_csv(
            images_d / "labels.csv" ,
            index = False
        )


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