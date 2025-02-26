#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command modules for varKoder.

This package contains the implementation of the various commands available
in the varKoder tool:
- image: Generate varKode images from DNA sequences
- train: Train a neural network model on varKode images
- query: Query a trained model with new DNA sequences
- convert: Convert between different k-mer mapping methods
"""

__all__ = ["image"]