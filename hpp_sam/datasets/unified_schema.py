"""
Unified Point Sample Schema for HPP-SAM.

This module defines the unified data schema for HPP-SAM, providing compatibility
with PartNet, PartNet-Mobility, ScanNet, and PartNeXt datasets.

Reference:
- 数据整理模块.md
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import torch


@dataclass
class UnifiedPointSample:
    """
    Unified data schema for HPP-SAM datasets.
    
    This schema ensures compatibility across different datasets while providing
    auxiliary metadata for improved sampling, debugging, and analysis.
    
    Attributes:
        coords: [N, 3] Point cloud coordinates (normalized to [-1, 1]).
        features: [N, F] Point cloud features (e.g., RGB, normals).
        gt_masks: [M, N] Ground truth binary masks.
        
        # Auxiliary metadata (for sampling and analysis)
        sample_id: Unique sample identifier.
        mask_areas: [M] Area of each mask (number of points).
        mask_level: Hierarchical level of masks (if available).
        node_ids: Node IDs for hierarchical datasets.
        parent_ids: Parent node IDs.
        depths: Depth of each node in hierarchy.
        is_leaf: Whether each node is a leaf.
        
        # Source information
        source_dataset: Name of source dataset (e.g., "PartNet", "PartNeXt").
        object_category: Object category (if available).
    """
    # Core data
    coords: np.ndarray
    features: np.ndarray
    gt_masks: np.ndarray
    
    # Auxiliary metadata
    sample_id: Optional[str] = None
    mask_areas: Optional[np.ndarray] = None
    mask_level: Optional[int] = None
    node_ids: Optional[np.ndarray] = None
    parent_ids: Optional[np.ndarray] = None
    depths: Optional[np.ndarray] = None
    is_leaf: Optional[np.ndarray] = None
    
    # Source information
    source_dataset: Optional[str] = None
    object_category: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize data."""
        if self.coords.dtype != np.float32:
            self.coords = self.coords.astype(np.float32)
        if self.features.dtype != np.float32:
            self.features = self.features.astype(np.float32)
        if self.gt_masks.dtype != bool:
            self.gt_masks = self.gt_masks.astype(bool)
        if self.mask_areas is None:
            self.mask_areas = np.array([mask.sum() for mask in self.gt_masks], dtype=np.float32)
    
    def to_torch(self) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors for model input."""
        return {
            "coords": torch.from_numpy(self.coords).float(),
            "features": torch.from_numpy(self.features).float(),
            "gt_masks": torch.from_numpy(self.gt_masks).bool(),
        }
    
    def has_valid_prompt(self, min_mask_area: int = 10) -> bool:
        """Check if the sample has valid prompts for training."""
        if self.mask_areas is None:
            return self.gt_masks.sum(axis=1).min() >= min_mask_area
        return (self.mask_areas >= min_mask_area).any()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedPointSample":
        """Create from dictionary."""
        return cls(
            coords=data["coords"],
            features=data.get("features", np.ones((len(data["coords"]), 3), dtype=np.float32)),
            gt_masks=data["gt_masks"],
            sample_id=data.get("sample_id"),
            mask_areas=data.get("mask_areas"),
            mask_level=data.get("mask_level"),
            node_ids=data.get("node_ids"),
            parent_ids=data.get("parent_ids"),
            depths=data.get("depths"),
            is_leaf=data.get("is_leaf"),
            source_dataset=data.get("source_dataset"),
            object_category=data.get("object_category"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coords": self.coords,
            "features": self.features,
            "gt_masks": self.gt_masks,
            "sample_id": self.sample_id,
            "mask_areas": self.mask_areas,
            "mask_level": self.mask_level,
            "node_ids": self.node_ids,
            "parent_ids": self.parent_ids,
            "depths": self.depths,
            "is_leaf": self.is_leaf,
            "source_dataset": self.source_dataset,
            "object_category": self.object_category,
        }
