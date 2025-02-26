#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert command module for varKoder.

This module contains functionality for converting between different k-mer mapping
methods for existing varKode images. It allows users to transform varKodes to 
Chaos Game Representations (CGR) and vice versa.
"""

import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from varKoder.core.config import (
    SAMPLE_BP_SEP, BP_KMER_SEP, MAPPING_CHOICES,
    DEFAULT_KMER_SIZE
)
from varKoder.core.utils import (
    eprint, get_kmer_mapping, get_metadata_from_img_filename,
    get_varKoder_labels, get_varKoder_qual, get_varKoder_freqsd
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def process_remapping(f_data: Dict[str, Any], output_mapping: str, sum_rc: bool = False) -> None:
    """
    Process a single image for remapping.
    
    Args:
        f_data: Dictionary with image data and metadata
        output_mapping: Target k-mer mapping method
        sum_rc: Whether to sum reverse complements when converting from CGR to varKode
    """
    # Skip if input and output mappings are the same
    if f_data['img_kmer_mapping'] == output_mapping:
        return
    
    # Skip if the file already exists
    if os.path.exists(f_data['outfile_path']) and not os.access(f_data['outfile_path'], os.W_OK):
        return
    
    # Create output directory if it doesn't exist
    f_data['outfile_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # Read input image
    original_img = Image.open(f_data['path'])
    
    # Get k-mer size from metadata
    kmer_size = f_data['img_kmer_size']
    
    # Get k-mer mapping for both source and target
    source_mapping = get_kmer_mapping(kmer_size, f_data['img_kmer_mapping'])
    target_mapping = get_kmer_mapping(kmer_size, output_mapping)
    
    # Get pixel array from original image
    img_array = np.array(original_img)
    
    # Create empty output array
    side = 2**(kmer_size-1)
    output_array = np.zeros((side, side), dtype=np.uint8)
    
    # Map pixels from source to target
    if f_data['img_kmer_mapping'] == 'cgr' and output_mapping == 'varKode' and sum_rc:
        # Special case for CGR to varKode with reverse complement summing
        kmers = set(source_mapping.index)
        for kmer in kmers:
            x_source, y_source = source_mapping.loc[kmer]['x'], source_mapping.loc[kmer]['y']
            try:
                coord_target = target_mapping.loc[kmer]
                output_array[coord_target['y'], coord_target['x']] += img_array[y_source, x_source]
            except KeyError:
                # Kmer not found in target mapping, might be due to reverse complement handling
                pass
    else:
        # Standard remapping
        for idx, row in source_mapping.iterrows():
            try:
                x_source, y_source = row['x'], row['y']
                coord_target = target_mapping.loc[idx]
                output_array[coord_target['y'], coord_target['x']] = img_array[y_source, x_source]
            except KeyError:
                # Kmer not found in target mapping, might be due to reverse complement handling
                pass
    
    # Create output image
    output_img = Image.fromarray(output_array)
    
    # Copy metadata from original image
    metadata = PngInfo()
    
    # Get labels if present
    try:
        labels = get_varKoder_labels(f_data['path'])
        metadata.add_text("varkoderKeywords", ";".join(labels))
    except:
        pass
    
    # Get quality flag if present
    try:
        qual_flag = get_varKoder_qual(f_data['path'])
        metadata.add_text("varkoderLowQualityFlag", "1" if qual_flag else "0")
    except:
        pass
    
    # Get base frequency standard deviation if present
    try:
        basefreq_sd = get_varKoder_freqsd(f_data['path'])
        metadata.add_text("varkoderBaseFreqSd", str(basefreq_sd))
    except:
        pass
    
    # Add mapping and k-mer size info
    metadata.add_text("varkoderMapping", output_mapping)
    metadata.add_text("varkoderKmerSize", str(kmer_size))
    
    # Add base pair info if available
    if f_data['bp'] is not None:
        metadata.add_text("varkoderBp", str(f_data['bp']))
    
    # Save output image
    output_img.save(f_data['outfile_path'], pnginfo=metadata)


class ConvertCommand:
    """
    Class for handling the convert command functionality in varKoder.
    
    This class implements methods to convert between different k-mer
    mapping methods (varKode and CGR) for existing images.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize ConvertCommand with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        
        # Validate arguments
        self._validate_args()
        
        # Set up paths
        self.input_dir = Path(args.input)
        self.output_dir = Path(args.outdir)
        
    def _validate_args(self) -> None:
        """Validate the command line arguments."""
        # Check if output directory exists and handle overwrite
        if not self.args.overwrite and Path(self.args.outdir).exists():
            raise Exception(
                "Output directory exists, use --overwrite if you want to overwrite it."
            )
            
        # Validate output mapping
        if self.args.output_mapping not in MAPPING_CHOICES:
            raise ValueError(f"Output mapping must be one of: {', '.join(MAPPING_CHOICES)}")
            
        # Validate input mapping if provided
        if self.args.input_mapping and self.args.input_mapping not in MAPPING_CHOICES:
            raise ValueError(f"Input mapping must be one of: {', '.join(MAPPING_CHOICES)}")
            
        # Validate kmer size if provided
        if self.args.kmer_size and self.args.kmer_size not in range(5, 10):
            raise ValueError("K-mer size must be between 5 and 9")
        
    def _collect_image_files(self) -> List[Dict[str, Any]]:
        """
        Collect and process image files to be converted.
        
        Returns:
            List of dictionaries containing image metadata
        """
        image_files = []
        
        for f in Path(self.args.input).rglob("*.png"):
            try:
                # Try to extract metadata from filename
                img_metadata = get_metadata_from_img_filename(f)
                
                # Override with command line arguments if provided
                if self.args.input_mapping:
                    img_metadata['img_kmer_mapping'] = self.args.input_mapping
                    
                if self.args.kmer_size:
                    img_metadata['img_kmer_size'] = self.args.kmer_size

            except:
                # Use defaults if metadata extraction fails
                img_metadata = {
                    'sample': f.stem,
                    'bp': 0,
                    'img_kmer_mapping': self.args.input_mapping or 'varKode',
                    'img_kmer_size': self.args.kmer_size or DEFAULT_KMER_SIZE,
                    'path': f
                }

            # Determine output path
            if img_metadata['sample'] and img_metadata['bp']:
                # Construct standardized filename
                fname = (
                    f"{img_metadata['sample']}{SAMPLE_BP_SEP}"
                    f"{int(img_metadata['bp'] / 1000):08d}K{BP_KMER_SEP}"
                    f"{self.args.output_mapping}{BP_KMER_SEP}"
                    f"k{img_metadata['img_kmer_size']}.png"
                )
                
                # Preserve directory structure
                rel_path = f.relative_to(Path(self.args.input)).parent
                img_metadata['outfile_path'] = (Path(self.args.outdir) / rel_path / fname)
            else:
                # Preserve entire path structure if metadata is not available
                img_metadata['outfile_path'] = Path(self.args.outdir) / f.relative_to(Path(self.args.input))
                
            image_files.append(img_metadata)
            
        return image_files
    
    def run(self) -> None:
        """
        Run the convert command to transform images between mapping methods.
        """
        # Create output directory if it doesn't exist
        Path(self.args.outdir).mkdir(parents=True, exist_ok=True)
        
        # Collect image files for processing
        image_files = self._collect_image_files()
        
        eprint(f"Found {len(image_files)} files to convert.")
        eprint(f"Converted images will be written to {self.args.outdir}")

        # Process files in parallel or sequentially
        if self.args.n_threads > 1:
            with multiprocessing.Pool(self.args.n_threads) as pool:
                process_partial = partial(
                    process_remapping, 
                    output_mapping=self.args.output_mapping,
                    sum_rc=self.args.sum_reverse_complements
                )
                
                results = list(tqdm(
                    pool.imap(process_partial, image_files), 
                    total=len(image_files), 
                    desc="Processing images"
                ))
        else:
            # Process sequentially
            for f_data in tqdm(image_files, desc="Processing images"):
                process_remapping(
                    f_data, 
                    self.args.output_mapping,
                    self.args.sum_reverse_complements
                )


def run_convert_command(args: Any) -> None:
    """
    Run the convert command with the given arguments.
    
    This is the main entry point for the convert command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
    """
    convert_cmd = ConvertCommand(args)
    convert_cmd.run()