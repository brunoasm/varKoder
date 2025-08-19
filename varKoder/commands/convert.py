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
    DEFAULT_KMER_SIZE, LABEL_SAMPLE_SEP, LABELS_SEP, QUAL_THRESH
)
from varKoder.core.utils import (
    eprint, get_kmer_mapping, get_metadata_from_img_filename,
    get_varKoder_labels, get_varKoder_qual, get_varKoder_freqsd,
    format_bp_human_readable
)

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def remap(img, k, in_mapping, out_mapping, sum_rc=False):
    """
    Remap an image from one k-mer mapping to another.
    
    Args:
        img: Input image
        k: K-mer size
        in_mapping: Input k-mer mapping method
        out_mapping: Output k-mer mapping method
        sum_rc: Whether to sum reverse complements
        
    Returns:
        Remapped image
    """
    if (not in_mapping in MAPPING_CHOICES) or (not out_mapping in MAPPING_CHOICES):
        raise Exception('Input and output mapping must be one of: ' + str(MAPPING_CHOICES))
        
    in_mp = get_kmer_mapping(k, in_mapping)
    out_mp = get_kmer_mapping(k, out_mapping)

    merged = in_mp.merge(out_mp, how='inner', left_index=True, right_index=True, suffixes=['_in', '_out'])
    # x and y are reversed for PIL images
    merged['y_in'] = merged['y_in'].max() - merged['y_in']
    merged['y_out'] = merged['y_out'].max() - merged['y_out']

    new_img = img.resize((merged['y_out'].max()+1, merged['x_out'].max()+1), 
                    resample=Image.NEAREST)
    new_img_array = np.zeros(np.array(new_img).shape, dtype=np.uint8)

    old_img_array = np.array(img)

    x_out = merged['x_out'].values
    y_out = merged['y_out'].values
    x_in = merged['x_in'].values
    y_in = merged['y_in'].values
    if sum_rc:
        np.add.at(new_img_array, (y_out, x_out), old_img_array[y_in, x_in])
        new_img_array = np.uint8((new_img_array-new_img_array.min())/new_img_array.max()*255)
    else:
        new_img_array[y_out, x_out] = old_img_array[y_in, x_in]

    new_img.putdata(new_img_array.flatten('A'))

    return new_img


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
    for k, v in image.info.items():
        # Update mapping info
        if k == 'varkoderMapping':
            pnginfo.add_text(k, output_mapping)
        else:
            pnginfo.add_text(k, str(v))
    
    # Create the necessary directories
    f_data['outfile_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # Save the new image
    new_img.save(f_data['outfile_path'], optimize=True, pnginfo=pnginfo)


class ConvertCommand:
    """
    Class for handling the convert command functionality in varKoder.
    
    This class implements methods to convert between different k-mer mapping methods
    for existing varKode images.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize ConvertCommand with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        
        # Check if output directory exists
        if not args.overwrite:
            if Path(args.outdir).exists():
                raise Exception(
                    "Output directory exists, use --overwrite if you want to overwrite it."
                )
    
    def _collect_image_files(self) -> List[Dict[str, Any]]:
        """
        Collect and process image files for conversion.
        
        Returns:
            List of dictionaries with image information
        """
        image_files = []
        for f in Path(self.args.input).rglob("*.png"):
            try:
                img_metadata = get_metadata_from_img_filename(f)
                if self.args.input_mapping:  # If input mapping passed as argument, it has priority
                    img_metadata['img_kmer_mapping'] = self.args.input_mapping
                if self.args.kmer_size:  # If input kmer size passed as argument, it has priority
                    img_metadata['img_kmer_size'] = self.args.kmer_size

            except:
                img_metadata = {
                    'sample': None,
                    'bp': None,
                    'img_kmer_mapping': self.args.input_mapping,
                    'img_kmer_size': self.args.kmer_size,
                    'path': f
                }

            if img_metadata['sample'] and img_metadata['bp']:
                fname = (
                         f"{img_metadata['sample']}{SAMPLE_BP_SEP}"
                         f"{format_bp_human_readable(int(img_metadata['bp']))}{BP_KMER_SEP}"
                         f"{self.args.output_mapping}{BP_KMER_SEP}"
                         f"k{img_metadata['img_kmer_size']}.png"
                        )
                img_metadata['outfile_path'] = (Path(self.args.outdir)/
                                                Path(*img_metadata['path'].relative_to(Path(self.args.input)).parent.parts[1:])/
                                                fname)
            else:
                img_metadata['outfile_path'] = Path(self.args.outdir)/Path(*img_metadata['path'].parts[1:])
                
            image_files.append(img_metadata)
        
        return image_files
    
    def run(self) -> None:
        """
        Run the convert command.
        """
        # Collect image files
        image_files = self._collect_image_files()
        
        eprint(f"Found {len(image_files)} files to convert.")
        eprint(f"Converted images will be written to {self.args.outdir}")
        
        # Process images in parallel or sequentially
        if self.args.n_threads > 1:
            with multiprocessing.Pool(self.args.n_threads) as pool:
                process_partial = partial(process_remapping, 
                                          output_mapping=self.args.output_mapping,
                                          sum_rc=self.args.sum_reverse_complements)
                results = list(tqdm(pool.imap(process_partial, image_files), 
                               total=len(image_files), 
                               desc="Processing images"))
        else:
            for f_data in tqdm(image_files, desc="Processing images"):
                process_remapping(f_data, 
                                 self.args.output_mapping,
                                 self.args.sum_reverse_complements)
        
        eprint("Conversion complete.")


def run_convert_command(args: Any) -> None:
    """
    Run the convert command with the given arguments.
    
    This is the main entry point for the convert command, called by the CLI.
    
    Args:
        args: Parsed command line arguments
    """
    convert_cmd = ConvertCommand(args)
    convert_cmd.run()