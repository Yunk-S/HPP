"""
Data Cleaning and Transformation Utilities for HPP-SAM.

This module provides utilities for cleaning and transforming data samples,
including format validation, mask filtering, point normalization, and random sampling.

Reference:
- 数据整理模块.md (Cleaners section)
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from .unified_schema import UnifiedPointSample


class SanitizeExample:
    """
    Validates and sanitizes input data samples.
    Ensures data conforms to expected format and value ranges.
    """
    
    def __init__(self, min_points: int = 100, max_points: int = 100000):
        self.min_points = min_points
        self.max_points = max_points
    
    def __call__(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and sanitize a sample. Returns None if sample is invalid."""
        if "coords" not in sample or "features" not in sample:
            return None
        
        coords = sample["coords"]
        features = sample["features"]
        
        if coords.shape[0] < self.min_points or coords.shape[0] > self.max_points:
            return None
        
        if coords.shape[0] != features.shape[0]:
            return None
        
        if not (-5 < coords.max() < 5 and -5 < coords.min() < 5):
            shift = np.mean(coords, axis=0)
            scale = np.max(np.linalg.norm(coords - shift, ord=2, axis=1)) + 1e-6
            sample["coords"] = (coords - shift) / scale
        
        return sample


class FilterInvalidMasks:
    """
    Filters out invalid masks from ground truth annotations.
    """
    
    def __init__(
        self, 
        min_area: int = 10,
        max_ratio: float = 0.9,
    ):
        self.min_area = min_area
        self.max_ratio = max_ratio
    
    def __call__(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter invalid masks from a sample."""
        if "gt_masks" not in sample:
            return sample
        
        gt_masks = sample["gt_masks"]
        num_points = gt_masks.shape[1] if gt_masks.ndim > 1 else len(gt_masks)
        max_area = int(num_points * self.max_ratio)
        
        valid_mask_indices = []
        for i in range(len(gt_masks)):
            mask = gt_masks[i]
            area = mask.sum()
            if area >= self.min_area and area <= max_area:
                valid_mask_indices.append(i)
        
        if len(valid_mask_indices) == 0:
            return None
        
        sample["gt_masks"] = gt_masks[valid_mask_indices]
        
        if "mask_areas" in sample:
            sample["mask_areas"] = np.array([
                sample["gt_masks"][i].sum() 
                for i in range(len(sample["gt_masks"]))
            ])
        
        return sample


class NormalizePoints:
    """Normalizes point cloud coordinates to [-1, 1] range."""
    
    def __init__(self, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize point coordinates."""
        coords = sample["coords"].copy()
        
        if self.center:
            shift = np.mean(coords, axis=0)
            coords = coords - shift
        
        if self.scale:
            scale = np.max(np.linalg.norm(coords, ord=2, axis=1)) + 1e-6
            coords = coords / scale
        
        sample["coords"] = coords.astype(np.float32)
        return sample


class RandomSamplePoints:
    """Randomly samples points from the point cloud."""
    
    def __init__(self, num_points: int):
        self.num_points = num_points
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly sample points."""
        coords = sample["coords"]
        features = sample["features"]
        gt_masks = sample["gt_masks"]
        
        num_current_points = coords.shape[0]
        
        if num_current_points == self.num_points:
            return sample
        
        if num_current_points > self.num_points:
            indices = np.random.choice(
                num_current_points, 
                self.num_points, 
                replace=False
            )
            indices = np.sort(indices)
        else:
            indices = np.random.choice(
                num_current_points, 
                self.num_points, 
                replace=True
            )
        
        sample["coords"] = coords[indices]
        sample["features"] = features[indices]
        
        if gt_masks is not None:
            sample["gt_masks"] = gt_masks[:, indices]
        
        return sample


class ComposeTransforms:
    """Composes multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
            if sample is None:
                return None
        return sample
