#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics for model evaluation in varKoder.

This module provides custom metrics for evaluating neural network models
in multi-label classification tasks.
"""

import torch
from fastai.metrics import Metric


class PrecisionMulti(Metric):
    """
    Precision metric for multi-label classification.
    
    Calculates the precision: TP / (TP + FP)
    """
    
    def __init__(self, thresh=0.5):
        """
        Initialize PrecisionMulti metric.
        
        Args:
            thresh (float): Threshold for positive prediction
        """
        self.thresh = thresh
        self.reset()
        
    def reset(self):
        """Reset accumulated values."""
        self.tp = 0
        self.fp = 0
        
    def accumulate(self, learn):
        """
        Accumulate true positives and false positives from a batch.
        
        Args:
            learn: fastai Learner object with predictions and target
        """
        pred, targ = learn.pred.sigmoid(), learn.y
        pred = (pred > self.thresh).float()
        self.tp += (pred * targ).sum().item()
        self.fp += (pred * (1 - targ)).sum().item()
        
    @property
    def value(self):
        """
        Calculate precision from accumulated values.
        
        Returns:
            float: Precision value
        """
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)


class RecallMulti(Metric):
    """
    Recall metric for multi-label classification.
    
    Calculates the recall: TP / (TP + FN)
    """
    
    def __init__(self, thresh=0.5):
        """
        Initialize RecallMulti metric.
        
        Args:
            thresh (float): Threshold for positive prediction
        """
        self.thresh = thresh
        self.reset()
        
    def reset(self):
        """Reset accumulated values."""
        self.tp = 0
        self.fn = 0
        
    def accumulate(self, learn):
        """
        Accumulate true positives and false negatives from a batch.
        
        Args:
            learn: fastai Learner object with predictions and target
        """
        pred, targ = learn.pred.sigmoid(), learn.y
        pred = (pred > self.thresh).float()
        self.tp += (pred * targ).sum().item()
        self.fn += ((1 - pred) * targ).sum().item()
        
    @property
    def value(self):
        """
        Calculate recall from accumulated values.
        
        Returns:
            float: Recall value
        """
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)


class F1ScoreMulti(Metric):
    """
    F1 score metric for multi-label classification.
    
    Calculates the F1 score: 2 * (precision * recall) / (precision + recall)
    """
    
    def __init__(self, thresh=0.5):
        """
        Initialize F1ScoreMulti metric.
        
        Args:
            thresh (float): Threshold for positive prediction
        """
        self.thresh = thresh
        self.precision = PrecisionMulti(thresh)
        self.recall = RecallMulti(thresh)
        
    def reset(self):
        """Reset precision and recall metrics."""
        self.precision.reset()
        self.recall.reset()
        
    def accumulate(self, learn):
        """
        Accumulate precision and recall from a batch.
        
        Args:
            learn: fastai Learner object with predictions and target
        """
        self.precision.accumulate(learn)
        self.recall.accumulate(learn)
        
    @property
    def value(self):
        """
        Calculate F1 score from precision and recall.
        
        Returns:
            float: F1 score value
        """
        prec = self.precision.value
        rec = self.recall.value
        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)