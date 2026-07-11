"""
We follow how torchvision transforms are implemented, and use `set_transform` method to apply data augmentation to the dataset.
It is easier to compose multiple data augmentation methods, and specify them in a config file.
Note that `set_transform` handles a batch of examples. `set_transform` is also compatible with IterableDataset.
Refer to https://huggingface.co/docs/datasets/v2.0.0/en/image_process#data-augmentation.
"""

from typing import Callable, Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, examples):
        for transform in self.transforms:
            examples = transform(examples)
        return examples

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Transform:
    def apply(self, example: dict):
        return example

    def __call__(self, examples: Dict[str, List]):
        keys = examples.keys()
        ret = [
            self.apply({k: v for k, v in zip(keys, example)})
            for example in zip(*[examples[k] for k in keys])
        ]
        examples.update({k: [x[k] for x in ret] for k in keys})
        return examples


class ToTensor(Transform):
    def apply(self, example):
        for k in ["coords", "features"]:
            example[k] = torch.tensor(np.array(example[k]), dtype=torch.float)
        for k in ["gt_masks"]:
            example[k] = torch.tensor(np.array(example[k]), dtype=torch.bool)
        return example


class NormalizeFeatures(Transform):
    """Normalize features to [-1, 1] range."""
    
    def apply(self, example):
        features = np.array(example["features"])
        # Normalize to [0, 1] then to [-1, 1]
        if features.max() > 1.0:
            features = features / 255.0
        features = features * 2.0 - 1.0
        example["features"] = features.astype(np.float32)
        return example


class FilterInvalidMasks(Transform):
    """Filter out masks with too few points."""
    
    def __init__(self, min_points: int = 16):
        self.min_points = min_points
    
    def apply(self, example):
        gt_masks = np.array(example["gt_masks"])  # [M, N]
        
        # Count points in each mask
        valid_mask = gt_masks.sum(axis=1) >= self.min_points
        
        # Keep at least one mask
        if not valid_mask.any():
            valid_mask[0] = True
        
        example["gt_masks"] = gt_masks[valid_mask]
        
        return example


def normalize_points(points: np.ndarray):
    """Normalize the point cloud into a unit sphere."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    points = points.astype(np.float32, copy=False)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    norm = max(float(norm), 1e-12)
    return (points / norm).astype(np.float32, copy=False)


class RandomSamplePoints(Transform):
    """Randomly sample a fixed number of points."""
    
    def __init__(self, num_points: int):
        self.num_points = num_points
    
    def apply(self, example):
        coords = np.asarray(example["coords"])
        features = np.asarray(example["features"])
        gt_masks = np.array(example["gt_masks"])  # [M, N]
        
        n = len(coords)
        
        if n >= self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
        else:
            indices = np.random.choice(n, self.num_points, replace=True)
        
        example["coords"] = coords[indices].astype(np.float32, copy=False)
        example["features"] = features[indices].astype(np.float32, copy=False)
        example["gt_masks"] = gt_masks[:, indices]
        
        return example


class NormalizePoints(Transform):
    def apply(self, example):
        example["coords"] = normalize_points(np.asarray(example["coords"]))
        return example


class NormalizeColor(Transform):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def apply(self, example):
        features = np.array(example["features"]) / 255
        if self.mean is not None:
            features = features - self.mean
        if self.std is not None:
            features = features / self.std
        example["features"] = features.astype(np.float32, copy=False)
        return example


class RandomSample(Transform):
    """Backward-compatible point sampler used by legacy configs."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def apply(self, example):
        coords = np.asarray(example["coords"])
        features = np.asarray(example["features"])
        gt_masks = np.asarray(example["gt_masks"])  # [M, N]

        n = len(coords)
        if n >= self.num_samples:
            indices = np.random.choice(n, self.num_samples, replace=False)
        else:
            indices = np.random.choice(n, self.num_samples, replace=True)

        example["coords"] = coords[indices].astype(np.float32, copy=False)
        example["features"] = features[indices].astype(np.float32, copy=False)
        if gt_masks.ndim == 2:
            example["gt_masks"] = gt_masks[:, indices]
        else:
            example["gt_masks"] = gt_masks
        return example

class SampleSingleMask(Transform):
    """ Constantly sample a single mask from the gt_masks. """

    def __init__(self, mask_id):
        self.mask_id = mask_id

    def apply(self, example):
        masks = example["gt_masks"]
        example["gt_masks"] = [masks[self.mask_id]]
        return example

class RandomSampleMask(Transform):
    """Randomly sample a fixed number of masks."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def apply(self, example):
        masks = example["gt_masks"]
        num_masks = len(masks)
        if num_masks < self.num_samples:
            mask_idx = np.random.choice(
                num_masks, self.num_samples - num_masks, replace=False
            )
            mask_idx = np.concatenate([np.arange(num_masks), mask_idx])
        elif num_masks > self.num_samples:
            mask_idx = np.random.choice(num_masks, self.num_samples, replace=False)
        else:
            mask_idx = np.arange(num_masks)
        example["gt_masks"] = [masks[i] for i in mask_idx]
        return example


class RandomRotateAlongAxis(Transform):
    def __init__(self, axis: str = "y"):
        assert axis in ["x", "y", "z"], "axis must be one of ['x', 'y', 'z']"
        self.axis = axis

    def apply(self, example):
        rot = Rotation.from_euler(self.axis, np.random.uniform(-180, 180), degrees=True)
        example["coords"] = rot.apply(example["coords"]).astype(np.float32, copy=False)
        return example


class RandomRotatePerbuate(Transform):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def apply(self, example):
        angles = np.clip(
            np.random.normal(0, self.angle_sigma, 3),
            -self.angle_clip,
            self.angle_clip,
        )
        rot = Rotation.from_euler("XYZ", angles)
        example["coords"] = rot.apply(example["coords"]).astype(np.float32, copy=False)
        return example


class RandomScale(Transform):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def apply(self, example):
        scale = np.random.uniform(self.low, self.high)
        example["coords"] = (np.asarray(example["coords"]) * scale).astype(np.float32, copy=False)
        return example


def main():
    from datasets import load_dataset, load_from_disk
    
    dataset = load_dataset(
        "SeaLab/partnet-shapenet",
        token="YOUR_HF_TOKEN",
        streaming=False,
        # streaming=True,
        split="test",
    )
    # dataset = load_from_disk("/jigu-haosu-vol/huggingface/datasets/partnet-shapenet/")
    # dataset = dataset["test"]
    # print(dataset)

    dataset = dataset.rename_columns(
        {"xyz": "coords", "rgb": "features", "mask": "gt_masks"}
    )
    print(dataset)
    dataset = dataset.select_columns(["coords", "features", "gt_masks"])
    print(dataset)

    transform = Compose(
        [
            RandomSampleMask(2),
            RandomSample(10000),
            RandomScale(0.8, 1.0),
            RandomRotatePerbuate(),
            RandomRotateAlongAxis(),
            ToTensor(),
        ]
    )

    if isinstance(dataset, torch.utils.data.IterableDataset):
        dataset = dataset.map(transform, batched=True, batch_size=1)
        data = next(iter(dataset))
    else:
        dataset.set_transform(transform)
        data = dataset[0]
    for k, v in data.items():
        v = np.array(v)
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
