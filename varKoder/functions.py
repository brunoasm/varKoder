#!/usr/bin/env python

# imports used for all commands
import pkg_resources
import contextlib
import traceback
from pathlib import Path
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from functools import partial

from io import StringIO
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from sklearn.exceptions import UndefinedMetricWarning

import pandas as pd, numpy as np, itertools, tempfile, shutil, subprocess, functools, hashlib
import pandas.errors
import os, re, sys, gzip, time, humanfriendly, random, multiprocessing, math, json, warnings
from concurrent.futures import ThreadPoolExecutor
from math import log

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL.Image import Resampling

from fastai.data.all import DataBlock, ColSplitter, ColReader
from fastai.vision.all import (
    ImageBlock,
    MultiCategoryBlock,
    CategoryBlock,
    vision_learner,
    parent_label
)
from fastai.vision.all import aug_transforms, Resize, ResizeMethod
from fastai.distributed import to_parallel, detach_parallel
from fastai.metrics import accuracy, accuracy_multi, PrecisionMulti, RecallMulti, RocAuc
from fastai.learner import Learner, load_learner
from fastai.torch_core import set_seed, default_device
from fastai.callback.mixup import CutMix, MixUp
from fastai.losses import CrossEntropyLossFlat  # , BCEWithLogitsLossFlat
from fastai.callback.core import Callback, CancelValidException
from fastai.torch_core import set_seed

from torch import nn
from torch.nn import CrossEntropyLoss
import torch

from timm import create_model
from timm.loss import AsymmetricLossMultiLabel
from huggingface_hub import from_pretrained_fastai

# define filename separators
label_sample_sep = "+"
labels_sep = ";"
bp_kmer_sep = "+"
sample_bp_sep = "@"
qual_thresh = 0.01
mapping_choices = ['varKode', 'cgr']
custom_archs = ['fiannaca2018', 'arias2022']

# ignore sklearn warning during training
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# defining a function to print to stderr more easily
# idea from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# this function process the input file or folder and returns a table with files
def process_input(inpath, is_query=False, no_pairs=False):
    # first, check if input is a folder
    # if it is, make input table from the folder
    # try:
    if inpath.is_dir() and not is_query:
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
                            files_records.append(
                                {
                                    "labels": (taxon.name,),
                                    "sample": sample.name,
                                    "files": taxon / sample.name / fl.name,
                                }
                            )
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
        files_records = list()
        # start by checking if input directory contains directories
        contains_dir = any(f.is_dir() or 
                           (f.is_symlink() and Path(os.readlink(f)).is_dir()) 
                           for f in inpath.iterdir())
        # if there are no subdirectories, or no_pairs is True treat each fastq as a single sample. Otherwise, use each directory for a sample
        if not contains_dir:
            for i, fl in enumerate(inpath.rglob('*')):
                if (
                    fl.name.endswith("fq")
                    or fl.name.endswith("fastq")
                    or fl.name.endswith("fq.gz")
                    or fl.name.endswith("fastq.gz")
                ):
                    files_records.append(
                        {
                            "labels": ("query",),
                            "sample": str(i) + "_" + fl.name.split(".")[0],
                            "files": fl,
                        }
                    )

        else:
            for sample in inpath.iterdir():
                if sample.resolve().is_dir():
                    #eprint("is_dir")
                    for fl in sample.iterdir():
                        #eprint(fl)
                        if (
                            fl.name.endswith("fq")
                            or fl.name.endswith("fastq")
                            or fl.name.endswith("fq.gz")
                            or fl.name.endswith("fastq.gz")
                        ):
                            files_records.append(
                                {
                                    "labels": ("query",),
                                    "sample": sample.name,
                                    "files": sample / fl.name,
                                }
                            )
        #eprint(files_records)

        files_table = (
            pd.DataFrame(files_records)
            .groupby(["labels", "sample"])
            .agg(list)
            .reset_index()
        )

        if not files_table.shape[0]:
            raise Exception("Folder detected, but no records read. Check format.")

    # if it isn't a folder, read csv table input
    else:
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

    # except:
    #    raise Exception('Could not parse input csv file or folder structure, double check.')

    files_table["sample"] = files_table["sample"].astype(str)

    files_table = (
        files_table.loc[:, ["labels", "sample", "files"]]
        .groupby("sample")
        .agg("sum")
        .map(lambda x: sorted(set(x)))
        .reset_index()
    )

    return files_table


# this function reads the json file produced by fastp within the clean_reads function
# and returns the standard deviation in base frequencies from positions 1-40 in forward reads.
# This is expected to be 0 for high-quality samples but increases in low-quality ones
def get_basefrequency_sd(file_list):
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

#helper function to preprocess files        
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
    
    opener = gzip.open if filename.endswith('gz') else open
    with opener(filename, 'rb') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # Sequence line
                total_length += len(line.strip())
                reads_counted += 1
            if reads_counted >= sample_size:
                break
    
    return total_length / reads_counted if reads_counted > 0 else 0, reads_counted

#helper function to preprocess files
def count_total_reads(filename):
    """Count total number of reads in a FASTQ file using system tools for efficiency."""
    
    # Convert to absolute path
    abs_path = str(Path(filename).resolve())
    
    try:
        if filename.endswith('gz'):
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
            with gzip.open(abs_path, 'rb') if filename.endswith('gz') else open(abs_path, 'rb') as f:
                line_count = sum(1 for _ in f)
                return line_count // 4
        except Exception as e:
            eprint(f"Failed to count reads: {str(e)}")
            raise

#helper function to preprocess files
def calculate_reads_needed(files_info, max_bp):
    """
    Calculate how many reads to take from each file to achieve target coverage.
    
    Args:
        files_info: Dictionary containing file information including average read lengths
        max_bp: Target base pair count
    
    Returns:
        Dictionary with number of reads to take from each file
    """
    reads_to_take = {}
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

#helper function to preprocess files
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
        opener = gzip.open if filename.endswith('gz') else open
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

#helper function to preprocess files
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


# cleans illumina reads and saves results as a single merged fastq file to outpath
# cleaning includes:
# 1 - adapter removal
# 2 - deduplication
# 3 - merging
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
    start_time = time.time()

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
    #with tempfile.TemporaryDirectory(prefix='barcoding_clean_' + basename) as work_dir:
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
            "fastp",
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
                    eprint(p.stderr.decode())
                    traceback.print_exc(file=sys.stdout)
                (Path(work_dir) / (basename + "_paired.fq")).unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            eprint(f"{basename}: fastp failed with paired reads, treating them as unpaired")

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
            "fastp",
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
                eprint(p.stderr.decode())
                traceback.print_exc(file=sys.stdout)

        except subprocess.CalledProcessError as e:
            eprint(f"{basename}: fastp failed with unpaired reads")
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
    done_time = time.time()
    stats["clean_basepairs"] = clean_bp
    stats["cleaning_time"] = done_time - start_time
    # eprint([x for x in Path(work_dir).glob(basename + '_fastp_*.json')])
    # stats['base_frequencies_sd'] = get_basefrequency_sd(Path(work_dir).glob(basename + '_fastp_*.json'))

    shutil.rmtree(work_dir)
    ####END OF TEMPDIR BLOCK

    return stats

#Helper function to parallelize reformat.sh execution
def run_parallel_reformats(sites_per_file, outfs, infile, seed, verbose=False, max_workers=None):
    # Create list of commands and their arguments
    commands = []
    for i, bp in enumerate(sites_per_file):
        command = [
            "reformat.sh",
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
                eprint(p.stderr.decode())
            return True
        except subprocess.CalledProcessError as e:
            if verbose:
                eprint(f"Error in subprocess {i}: {e}")
            return False

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs and wait for completion
        results = list(executor.map(run_single_command, commands))

    # Check if all processes succeeded
    if not all(results):
        raise RuntimeError("One or more reformat processes failed")

# function takes a fastq file as input and splits it into several files
# to be saved in outfolder with outprefix as prefix in the file name
# this is because we found that NNs could accurately identify samples
# but below a certain level of coverage they were sensitive to amount
# of data used as input.
# Therefore, to better generalize we split the input files in several
# The first one randomly pulls half of the reads, the next gets half of the remaining, etc
# until there are less than min_bp left
# if max_bp is set, the first file will use max_bp or half of the reads, whatever is lower
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
    start_time = time.time()

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
        eprint( "Input file "
            + str(infile)
            + " has less than "
            + str(min_bp)
            + "bp, raise --min_bp if you want to produce an image.")
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
            + sample_bp_sep
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

    run_parallel_reformats(sites_per_file, outfs, infile, seed, verbose=False, max_workers=n_threads)

    done_time = time.time()

    stats = OrderedDict()

    stats["splitting_time"] = done_time - start_time
    stats["splitting_bp_per_file"] = ",".join([str(x) for x in sites_per_file])

    return stats


# counts kmers for fastq infile, saving results as an hdf5 container
# in some cases dsk lauches an hdf5 error of file access when using more than 1 thread. For that reason, will keep one thread for now
def count_kmers(infile, outfolder, threads=1, k=7, overwrite=False, verbose=False):
    start_time = time.time()

    Path(outfolder).mkdir(exist_ok=True)
    outfile = (
        str(Path(infile).name.removesuffix("".join(Path(infile).suffixes)))
        + bp_kmer_sep
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
                p = subprocess.run(
                    [
                        "dsk",
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
                    ],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                if verbose:
                    eprint(p.stderr.decode())

    done_time = time.time()

    stats = OrderedDict()
    stats[str(k) + "mer_counting_time"] = done_time - start_time

    return stats

# This function returns the coordinates of kmers according to chaos game representation
# for a given kmer_size
# By default, both a canonical kmer and its reverse complement map to 2 pixels
# If compact is True, they will both map to a single pixel, for a more compact representation
def get_cgr(kmer_size):

    #following corners from Jeffrey, using cartesian coords
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
    df = pd.concat([pd.DataFrame({
                      'x': coords[:, 0],
                      'y': coords[:, 1]
                      }, index=[''.join(nucleotide_map[seq]) 
                                for seq in all_sequences]),
                    pd.DataFrame({
                      'x': coords[:, 0],
                      'y': coords[:, 1]
                      }, index=[''.join(nucleotide_map[seq]) 
                                for seq in rev_complement])
                   ])

    # Replace coords with integer numbers
    sq_side = len(df['x'].drop_duplicates())
    df['x'] = (sq_side*(df['x'] - df['x'].min())).astype(int)
    df['y'] = (sq_side*(df['y'] - df['y'].min())).astype(int)


    return df




# This function returns a table mapping specific kmers of size kmer-size to
# a x and y coordinates in a square grid
# method options: varkode, cgr
# cgr: chaos game representation
def get_kmer_mapping(kmer_size = 7, method = 'varKode'):


    if method == 'varKode':
        map_path = pkg_resources.resource_filename(
            "varKoder", f"kmer_mapping/{kmer_size}mer_mapping.parquet"
        )
        kmer_mapping = pd.read_parquet(map_path).set_index("kmer")
    elif method == 'cgr':
        kmer_mapping = get_cgr(kmer_size)
    else:
        raise Exception('method must be "varKode" or "cgr"')

    return kmer_mapping


# given an input dsk hdf5 file, make an image with kmer counts
# kmer_mapping is the instruction of which mapping to use:
### * k+
# we do not need to save the ascii dsk kmer counts to disk, but the program requires a file
# so we will create a temporary file and delete it
def make_image(
    infile,
    outfolder,
    kmer_mapping,
    threads=1,
    overwrite=False,
    verbose=False,
    labels=[],
    base_sd=0,
    base_sd_thresh=qual_thresh,
    subfolder_levels=0,
    mapping_code='varKode',
):
    
    
    in_basename = str(Path(infile).name.removesuffix("".join(Path(infile).suffixes)))
    in_base1, in_k = in_basename.split(bp_kmer_sep)

    outfile = (in_base1 +
               bp_kmer_sep +
               mapping_code +
               bp_kmer_sep +
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

    start_time = time.time()
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
                dsk_out = subprocess.run(
                    [
                        "dsk2ascii",
                        "-c",
                        "-file",
                        str(infile),
                        "-nb-cores",
                        str(threads),
                        "-out",
                        str(Path(outdir) / "dsk.txt"),
                        "-verbose",
                        "0",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        if verbose:
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
        metadata.add_text("varkoderKeywords", labels_sep.join(labels))
        metadata.add_text("varkoderBaseFreqSd", str(base_sd))
        metadata.add_text("varkoderLowQualityFlag", str(base_sd > qual_thresh))
        metadata.add_text("varkoderMapping", mapping_code)

        # finally, save the image
        img.save(Path(outfolder) / outfile, optimize=True, pnginfo=metadata)

    done_time = time.time()
    stats = OrderedDict()
    stats["k" + str(kmer_size) + "_img_time"] = done_time - start_time

    return stats


# Function: read stats file:
def read_stats(stats_path):
    return pd.read_csv(stats_path, index_col=[0], dtype={0: str}, low_memory=False).to_dict(
        orient="index"
    )


# Function: save stats dict as a csv file:
def stats_to_csv(all_stats, stats_path):
    df = pd.DataFrame.from_dict(all_stats, orient="index").rename_axis(index=["sample"])
    df.to_csv(stats_path)
    return df.reset_index()


### To be able to run samples in parallel using python, the image pipeline has to be encoded
### into a function
### this will take as input:
### one row of the condensed_files dataframe, the stats dictionary and the number of cores to use per sample


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
    label_sample_sep=label_sample_sep,
    humanfriendly=humanfriendly,
    defaultdict=defaultdict,
    eprint=eprint,
    make_image=make_image,
    count_kmers=count_kmers,
):
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
        for infile in split_reads_d.glob(x["sample"] + sample_bp_sep + "*"):
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
        for infile in kmer_counts_d.glob(x["sample"] + sample_bp_sep + "*"):
            base_sd = get_basefrequency_sd(
                Path(inter_dir, "clean_reads").glob(str(x["sample"]) + "_fastp_*.json")
            )
            stats[str(x["sample"])]["base_frequencies_sd"] = base_sd
            kmer_mapping = get_kmer_mapping(
                kmer_size = args.kmer_size,
                method = args.kmer_mapping)     
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
                    traceback.print_exc(file=sys.stdout)
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


## This is just an unwrapper to be able to use run_clean2img() with multiprocessing
def run_clean2img_wrapper(args_tuple):
    # print(f"args_tuple received: {args_tuple}")
    # Unpack the arguments
    return run_clean2img(*args_tuple)


# This processes stats after creating images
def process_stats(
    stats,
    condensed_files,
    args,
    stats_path,
    images_d,
    all_stats,
    qual_thresh,
    labels_sep,
):
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


###### Functions to train a nn


# Function: select training and validation set given kmer size and amount of data (as a list).
# minbp_filter will only consider samples that have yielded at least that amount of data
def get_train_val_sets():
    file_path = [
        x.absolute()
        for x in (Path("images_" + str(kmer_size))).ls()
        if x.suffix == ".png"
    ]
    taxon = [
        x.name.split(sample_bp_sep)[0].split(label_sample_sep)[0] for x in file_path
    ]
    sample = [
        x.name.split(sample_bp_sep)[0].split(label_sample_sep)[1] for x in file_path
    ]
    n_bp = [
        int(x.name.split(sample_bp_sep)[1].split(bp_kmer_sep)[0].replace("K", "000"))
        for x in file_path
    ]

    df = pd.DataFrame(
        {"taxon": taxon, "sample": sample, "n_bp": n_bp, "path": file_path}
    )

    if minbp_filter is not None:
        included_samples = df.loc[df["n_bp"] == 200000000]["sample"].drop_duplicates()
    else:
        included_samples = df["sample"].drop_duplicates()

    df = df.loc[df["sample"].isin(included_samples)]

    valid = (
        df[["taxon", "sample"]]
        .drop_duplicates()
        .groupby("taxon")
        .apply(lambda x: x.sample(n_valid, replace=False))
        .reset_index(drop=True)
        .assign(is_valid=True)
        .merge(df, on=["taxon", "sample"])
    )

    train = df.loc[~df["sample"].isin(valid["sample"])].assign(is_valid=False)
    train = train.loc[train["n_bp"].isin(bp_training)]

    train_valid = pd.concat([valid, train]).reset_index(drop=True)
    return train_valid


# Function: retrieve varKoder labels as a list
def get_varKoder_labels(img_path):
    return [x for x in Image.open(img_path).info.get("varkoderKeywords").split(";")]


# Function: retrieve varKoder quality flag
def get_varKoder_qual(img_path):
    return bool(Image.open(img_path).info.get("varkoderLowQualityFlag"))


# Function: retrieve basefreq sd
def get_varKoder_freqsd(img_path):
    return float(Image.open(img_path).info.get("varkoderBaseFreqSd"))

# Function: retrieve mapping
def get_varKoder_mapping(img_path):
    return str(Image.open(img_path).info.get("varkoderMapping"))

# Function: retrieve kmer size and kmer mapping method from image file name
def get_metadata_from_img_filename(img_path):
    sample_name, split2 = Path(img_path).name.removesuffix('.png').split(sample_bp_sep)
    try:
        n_bp, img_kmer_mapping, img_kmer_size = split2.split(bp_kmer_sep)
    except ValueError: #backwards compatible with varKoder v0.X
        n_bp, img_kmer_size = split2.split(bp_kmer_sep)
        img_kmer_mapping = 'varKode'
        
    n_bp = int(n_bp[:-1])*1000
    img_kmer_size = int(img_kmer_size[1:])


    return {'sample': sample_name,
            'bp': n_bp,
            'img_kmer_mapping': img_kmer_mapping,
            'img_kmer_size': img_kmer_size,
            'path':Path(img_path)
           }
    
# Function: retrieve varKoder base frequency sd as a float and apply an exponential function to turn it into a loss function weight
def get_varKoder_quality_weights(img_path):
    base_sd = float(Image.open(img_path).info.get("varkoderBaseFreqSd"))
    weight = 2 / (1 + math.exp(20 * base_sd))
    return weight


# Custom training loop for fastai to use sample weights
# def custom_weighted_training_loop(learn, sample_weights):
#    model, opt = learn.model, learn.opt
#    for xb, yb in learn.dls.train:
#        # Get the corresponding sample weights for the current batch
#        batch_sample_weights = [sample_weights[i] for i in learn.dls.train.get_idxs()]
#
#        # Convert the sample weights to a tensor
#        batch_sample_weights_tensor = torch.tensor(batch_sample_weights, device=xb.device, dtype=torch.float)
#
#        # Calculate the loss with the custom loss function
#        loss = learn.loss_func(model(xb), yb, batch_sample_weights_tensor)
#
#        # Backward pass
#        loss.backward()
#
#        # Optimization step
#        opt.step()
#
#        # Zero the gradients
#        opt.zero_grad()
#
##Callback to add sample weights
# class CustomWeightedTrainingCallback(Callback):
#    def __init__(self, sample_weights):
#        self.sample_weights = sample_weights
#
#    def before_fit(self):
#        self.learn._do_one_batch = lambda: custom_weighted_training_loop(self.learn, self.sample_weights)
#
## Modified AsymmetricLossMultiLabel from timm library to include sample weights
# class CustomWeightedAsymmetricLossMultiLabel(nn.Module):
#    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
#        super(CustomWeightedAsymmetricLossMultiLabel, self).__init__()
#
#        self.gamma_neg = gamma_neg
#        self.gamma_pos = gamma_pos
#        self.clip = clip
#        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#        self.eps = eps
#
#    def forward(self, x, y, sample_weights=None):
#        """
#        Parameters
#        ----------
#        x: input logits
#        y: targets (multi-label binarized vector) or tuple of targets for MixUp
#        sample_weights: per-sample weights, should have the same shape as y
#        """
#
#        if isinstance(y, tuple):
#            y1, y2, lam = y
#            y = lam * y1 + (1 - lam) * y2
#            if sample_weights is not None:
#                sample_weights = lam * sample_weights + (1 - lam) * sample_weights
#
#        # Calculating Probabilities
#        x_sigmoid = torch.sigmoid(x)
#        xs_pos = x_sigmoid
#        xs_neg = 1 - x_sigmoid
#
#        # Asymmetric Clipping
#        if self.clip is not None and self.clip > 0:
#            xs_neg = (xs_neg + self.clip).clamp(max=1)
#
#        # Basic CE calculation
#        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
#        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
#        loss = los_pos + los_neg
#
#        # Asymmetric Focusing
#        if self.gamma_neg > 0 or self.gamma_pos > 0:
#            if self.disable_torch_grad_focal_loss:
#                torch._C.set_grad_enabled(False)
#            pt0 = xs_pos * y
#            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
#            pt = pt0 + pt1
#            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
#            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
#            if self.disable_torch_grad_focal_loss:
#                torch._C.set_grad_enabled(True)
#            loss *= one_sided_w
#
#        # Apply sample weights
#        if sample_weights is not None:
#            loss *= sample_weights
#
#        return -loss.sum()

# Function: build a custom models for linearized varKodes or CGRs based on prior work
# Define classes for custom models

class Arias2022Head(nn.Module):
    def __init__(self, n_classes):
        super(Arias2022Head, self).__init__()
        self.head = nn.Sequential(nn.Linear(64, n_classes))
    def forward(self, x):
        return self.head(x)

class Arias2022Body(nn.Module):
    def __init__(self):
        super(Arias2022Body, self).__init__()
        self.body = nn.Sequential(
                nn.Flatten(), #reshape to 1D array
                nn.LazyLinear(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Dropout(0.5))

    def forward(self, x):
        x = x[:, 0, :, :] #keep only one channel
        x = self.body(x)
        return x

class Fiannaca2018Head(nn.Module):
    def __init__(self,n_classes):
        super(Fiannaca2018Head, self).__init__()
        self.head = nn.Sequential(nn.Linear(500, n_classes))

    def forward(self, x):
        #x = x.unsqueeze(1)  # Add channel dimension for convolution layers
        return self.head(x)

class Fiannaca2018Body(nn.Module):
    def __init__(self):
        super(Fiannaca2018Body, self).__init__()
        self.flatten = nn.Flatten()
        self.body = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=5),  # First convolutional layer
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Pooling layer

            nn.Conv1d(5, 10, kernel_size=5),  # Second convolutional layer
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Pooling layer

            nn.Flatten(),
            nn.LazyLinear(500),  # Adjust the size based on the output of previous layers
            nn.ReLU())

    def forward(self, x):
        x = x[:, 0, :, :] #keep only one channel
        x = self.flatten(x)
        x = x.unsqueeze(1)
        x = self.body(x)
        return x

class Fiannaca2018Model(nn.Module):
    def __init__(self,n_classes):
        super(Fiannaca2018Model, self).__init__()
        self.model = nn.Sequential(Fiannaca2018Body(),Fiannaca2018Head(n_classes))

    def forward(self, x):
        x = self.model(x)
        return x

class Arias2022Model(nn.Module):
    def __init__(self,n_classes):
        super(Arias2022Model, self).__init__()
        self.model = nn.Sequential(Arias2022Body(),Arias2022Head(n_classes))

    def forward(self, x):
        x = self.model(x)
        return x

def build_custom_model(architecture, dls):
    if architecture == 'arias2022':
        custom_model = Arias2022Model(len(dls.vocab))
    elif architecture == 'fiannaca2018':
        custom_model = Fiannaca2018Model(len(dls.vocab))
    else:
        raise Exception('Custom models must be one of: fiannaca2018 arias2022')

    # Initiallize LazyLinear with dummy batch
    xb, yb = dls.one_batch()
    input_image_size = xb.shape[-2:]  
    dummy_batch = torch.randn((1, 1, input_image_size[0], input_image_size[1]))  
    custom_model(dummy_batch)

    return custom_model
    
# Define callback to skip validation
class SkipValidationCallback(Callback):
    def before_validate(self):
        raise CancelValidException


# Function: build dataloaders, learners and train
def train_nn(
    df,
    architecture,
    valid_pct=0.2,
    max_bs=64,
    base_lr=1e-3,
    model_state_dict=None,
    epochs=30,
    freeze_epochs=0,
    normalize=True,
    callbacks=CutMix,
    max_lighting=0,
    p_lighting=0,
    pretrained=False,
    loss_fn=CrossEntropyLoss(),
    is_multilabel=False,
    metrics_threshold=0.7,
    gamma_neg=4,
    verbose=True,
    num_workers = 0,
    no_metrics = False
):
    # if skipping validation metrics, add NoValidation callback
    if no_metrics:
        if isinstance(callbacks,list):
            callbacks.append(SkipValidationCallback())
        else:
            callbacks=[callbacks,SkipValidationCallback()]


    # find a batch size that is a power of 2 and splits the dataset in about 10 batches
    batch_size = 2 ** round(log(df[~df["is_valid"]].shape[0] / 10, 2))
    batch_size = min(batch_size, max_bs)

    # set kind of splitter for DataBlock
    if "is_valid" in df.columns:
        sptr = ColSplitter()
    else:
        sptr = RandomSplitter(valid_pct=valid_pct)

    # check if item resizing is necessary
    item_transforms = None
    if not architecture in custom_archs:
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
            

    # set batch transforms
    transforms = aug_transforms(
        do_flip=False,
        max_rotate=0,
        max_zoom=1,
        max_lighting=max_lighting,
        max_warp=0,
        p_affine=0,
        p_lighting=p_lighting,
    )

    # set DataBlock
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

    # create data loaders with calculated batch size and appropriate device
    dls = dbl.dataloaders(df, 
        bs=batch_size, 
        device=default_device(), 
        num_workers=num_workers)

    # create learner
    if is_multilabel:
        # find all labels that are not 'low_quality:True'
        labels = [i for i, x in enumerate(dls.vocab) if x != "low_quality:True"]
        # define metrics
        precision = PrecisionMulti(labels=labels, average="micro", thresh=metrics_threshold)
        recall = RecallMulti(labels=labels, average="micro", thresh=metrics_threshold)
        auc = RocAuc(average="micro")
        metrics = [auc, precision, recall]
    else:
        metrics = accuracy

    if architecture in custom_archs:
        #build model
        custom_model = build_custom_model(architecture, dls)
        
        learn = Learner(dls, 
                        custom_model, 
                        metrics=metrics, 
                        cbs=callbacks,
                        loss_func=loss_fn
                       ).to_fp16()
        
    else:
        learn = vision_learner(dls,
                               architecture,
                               metrics=metrics,
                               normalize=normalize,
                               pretrained=pretrained,
                               cbs=callbacks,
                               loss_func=loss_fn,
                            ).to_fp16()
   
    # if there a pretrained model body weights, replace them
    if model_state_dict:
        old_state_dict = learn.state_dict()
        new_state_dict = {
            k: v
            for k, v in model_state_dict.items()
            if k in old_state_dict and old_state_dict[k].size() == v.size()
        }
        learn.model.load_state_dict(new_state_dict, strict=False)

    # Check for multiple GPUs and parallelize if available
    is_parallel = torch.backends.cuda.is_built() and torch.cuda.device_count() > 1
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

##Function: get the reverse complement of a sequence string
def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reverse_complement = "".join(complement[base] for base in reversed(seq))
    return reverse_complement



## Function: remap an image
## in_mapping and out_mapping should be one of the supported kmer_mapping 
def remap(img, k, in_mapping, out_mapping, sum_rc=False):
    if (not in_mapping in mapping_choices) or (not out_mapping in mapping_choices):
        raise Exception('Input and output mapping must be one of: ' + str(mapping_choices))
        
    in_mp = get_kmer_mapping(k,in_mapping)
    out_mp = get_kmer_mapping(k,out_mapping)

    merged = in_mp.merge(out_mp, how='inner',left_index=True, right_index=True,suffixes=['_in', '_out'])
    #x and y are reversed for PIL images
    merged['y_in'] = merged['y_in'].max()-merged['y_in']
    merged['y_out'] = merged['y_out'].max()-merged['y_out']

    new_img = img.resize((merged['y_out'].max()+1,merged['x_out'].max()+1),
                    resample=Image.NEAREST)
    new_img_array = np.zeros(np.array(new_img).shape,dtype=np.uint8)

    old_img_array = np.array(img)

    x_out = merged['x_out'].values
    y_out = merged['y_out'].values
    x_in = merged['x_in'].values
    y_in = merged['y_in'].values
    if sum_rc:
        np.add.at(new_img_array, (y_out, x_out), old_img_array[y_in, x_in])
        new_img_array = np.uint8((new_img_array-new_img_array.min())/new_img_array.max()*255)
    else:
        new_img_array[y_out,x_out] = old_img_array[y_in,x_in]

    new_img.putdata(new_img_array.flatten('A'))

    return(new_img)

#Function: processes one image for remapping
def process_remapping(f_data, output_mapping, sum_rc):
    # Open the image
    image = Image.open(f_data['path'])
    
    # Remap the image
    new_img = remap(image, 
                    f_data['img_kmer_size'], 
                    f_data['img_kmer_mapping'], 
                    output_mapping,
                    sum_rc
                   )
    
    # Create a PngInfo object and add the necessary info
    pnginfo = PngInfo()
    for k, v in new_img.info.items():
        if k == 'VarkoderMapping':
            pnginfo.add_text(k, f_data['img_kmer_mapping'])
        else:
            pnginfo.add_text(k, str(v))
    
    # Create the necessary directories
    f_data['outfile_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # Save the new image
    new_img.save(f_data['outfile_path'], optimize=True, pnginfo=pnginfo)
        





   



