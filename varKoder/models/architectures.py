#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network architectures for varKoder.

This module provides custom neural network architectures for DNA barcode
classification with varKode images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiannacaModel(nn.Module):
    """
    Implementation of the CNN architecture from Fiannaca et al. (2018).
    
    Based on: Fiannaca, Antonino, et al. "Deep learning models for bacteria 
    taxonomic classification of metagenomic data." BMC bioinformatics 19.7 (2018): 61-76.
    """
    
    def __init__(self, num_classes, is_multilabel=True):
        """
        Initialize FiannacaModel.
        
        Args:
            num_classes (int): Number of output classes
            is_multilabel (bool): Whether this is a multi-label classification task
        """
        super(FiannacaModel, self).__init__()
        
        self.is_multilabel = is_multilabel
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output predictions
        """
        # Convert grayscale to single channel if needed
        if x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)
            
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 16 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.fc3(x)
        
        if not self.training and not self.is_multilabel:
            x = F.softmax(x, dim=1)
            
        return x


class AriasModel(nn.Module):
    """
    Implementation of the CNN architecture from Arias et al. (2022).
    
    Based on: Arias, César F., et al. "A general model for species 
    identification from sequence data." Molecular ecology resources 
    22.7 (2022): 2474-2491.
    """
    
    def __init__(self, num_classes, is_multilabel=True):
        """
        Initialize AriasModel.
        
        Args:
            num_classes (int): Number of output classes
            is_multilabel (bool): Whether this is a multi-label classification task
        """
        super(AriasModel, self).__init__()
        
        self.is_multilabel = is_multilabel
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output predictions
        """
        # Convert grayscale to single channel if needed
        if x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)
            
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = self.gap(x)
        x = x.view(-1, 64)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.fc2(x)
        
        if not self.training and not self.is_multilabel:
            x = F.softmax(x, dim=1)
            
        return x


def fiannaca2018_model(num_classes, is_multilabel=True):
    """
    Create a FiannacaModel instance.
    
    Args:
        num_classes (int): Number of output classes
        is_multilabel (bool): Whether this is a multi-label classification task
        
    Returns:
        FiannacaModel: Initialized model
    """
    return FiannacaModel(num_classes, is_multilabel)


def arias2022_model(num_classes, is_multilabel=True):
    """
    Create an AriasModel instance.
    
    Args:
        num_classes (int): Number of output classes
        is_multilabel (bool): Whether this is a multi-label classification task
        
    Returns:
        AriasModel: Initialized model
    """
    return AriasModel(num_classes, is_multilabel)