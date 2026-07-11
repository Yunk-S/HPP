# https://github.com/baaivision/Uni3D/blob/main/models/point_encoder.py
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torkit3d.nn.functional import batch_index_select
from torkit3d.ops.sample_farthest_points import sample_farthest_points
from torkit3d.ops.chamfer_distance import chamfer_distance


def fps(points: torch.Tensor, num_samples: int):
    """A wrapper of farthest point sampling (FPS).

    Args:
        points: [B, N, 3]. Input point clouds.
        num_samples: int. The number of points to sample.

    Returns:
        torch.Tensor: [B, num_samples, 3]. Sampled points.
    """
    idx = sample_farthest_points(points, num_samples)
    sampled_points = batch_index_select(points, idx, dim=1)
    return sampled_points


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


class KNNGrouper(nn.Module):
    """Group points based on K nearest neighbors.

    A number of points are sampled as centers by farthest point sampling (FPS).
    Each group is formed by the center and its k nearest neighbors.
    """

    def __init__(self, num_groups, group_size, radius=None, centralize_features=False):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, use_fps=True):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features.
            use_fps: bool. Whether to use farthest point sampling.
                If not, `xyz` should already be sampled by FPS.

        Returns:
            dict: {
                features: [B, G, K, 3 + C]. Group features.
                centers: [B, G, 3]. Group centers.
                knn_idx: [B, G, K]. The indices of k nearest neighbors.
            }
        """
        batch_size, num_points, _ = xyz.shape
        with torch.no_grad():
            if use_fps:
                fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
                centers = batch_index_select(xyz, fps_idx, dim=1)
            else:
                fps_idx = torch.arange(self.num_groups, device=xyz.device)
                fps_idx = fps_idx.expand(batch_size, -1).clone()
                centers = xyz[:, : self.num_groups]
            _, knn_idx = knn_points(centers, xyz, self.group_size)  # [B, G, K]

        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * G * K]

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, G, K, 3]
        # NOTE: Follow PointNext to normalize the relative position
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(
            batch_size, self.num_groups, self.group_size, features.shape[-1]
        )

        group_feats = [nbr_xyz, nbr_feats]
        if self.centralize_features:
            center_feats = batch_index_select(features, fps_idx, dim=1)
            group_feats.append(nbr_feats - center_feats.unsqueeze(2))

        group_feats = torch.cat(group_feats, dim=-1)
        return dict(
            features=group_feats, centers=centers, knn_idx=knn_idx, fps_idx=fps_idx
        )


def group_with_centers_and_knn(
    xyz: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    knn_idx: torch.Tensor,
    radius: float = None,
    centralize_features: bool = False,
    center_idx: torch.Tensor = None,
):
    """Group points based on K nearest neighbors.

    Args:
        xyz: [B, N, 3]. Input point clouds.
        features: [B * M, N, C]. Point features. Support multiple features for the same point cloud.
        centers: [B, L, 3]. Group centers.
        knn_idx: [B, L, K]. The indices of k nearest neighbors.

    Returns:
        torch.Tensor: [B * M, L, K, 3 + C]. Group features.
    """
    assert xyz.dim() == features.dim(), (xyz.shape, features.shape)
    assert xyz.shape[1] == features.shape[1], (xyz.shape, features.shape)
    assert xyz.shape[0] == centers.shape[0] == knn_idx.shape[0]
    assert knn_idx.shape[:2] == centers.shape[:2], (knn_idx.shape, centers.shape)

    # 1. Compute neighborhood coordinates
    batch_size, num_points, _ = xyz.shape
    _, num_patches, patch_size = knn_idx.shape

    batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
    batch_offset = batch_offset.reshape(-1, 1, 1)
    knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * L * K]

    nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
    nbr_xyz = nbr_xyz.reshape(batch_size, num_patches, patch_size, 3)
    nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, L, K, 3]
    if radius is not None:
        # dist = torch.linalg.norm(nbr_xyz, dim=-1, ord=2)
        # print(dist.max(), dist.min(), dist.mean())
        nbr_xyz = nbr_xyz / radius

    # 2. Compute neighborhood features
    batch_size2 = features.shape[0]
    repeats = features.shape[0] // xyz.shape[0]
    knn_idx2 = torch.repeat_interleave(knn_idx, repeats, dim=0)  # [B*M,L,K]

    batch_offset = torch.arange(batch_size2, device=xyz.device) * num_points
    batch_offset = batch_offset.reshape(-1, 1, 1)
    knn_idx_flat = (knn_idx2 + batch_offset).reshape(-1)  # [B*M*L*K]
    nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
    nbr_feats = nbr_feats.reshape(
        batch_size2, num_patches, patch_size, features.shape[-1]
    )

    # 3. Concatenate features
    nbr_xyz = torch.repeat_interleave(nbr_xyz, repeats, dim=0)
    group_feats = [nbr_xyz, nbr_feats]
    if centralize_features:
        center_idx = torch.repeat_interleave(center_idx, repeats, dim=0)
        center_feats = batch_index_select(features, center_idx, dim=1)
        group_feats.append(nbr_feats - center_feats.unsqueeze(2))
    return torch.cat(group_feats, dim=-1)


class NNGrouper(nn.Module):
    """Group points based on the nearest neighbors."""

    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        with torch.no_grad():
            fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
            centers = batch_index_select(xyz, fps_idx, dim=1)
            _, nn_idx = knn_points(xyz, centers, 1)  # [B, N, 1]

        # Compute the relative position of each point to its nearest center
        nn_idx = nn_idx.squeeze(-1)
        nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)  # [B, N, 3]

        # Normalize the relative position
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)

        group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
        return dict(features=group_feats, centers=centers, nn_idx=nn_idx)

def group_with_centers_and_nn(
    xyz: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    nn_idx: torch.Tensor,
):
    """
    Group points based on the voronoi diagram.

    Args:
        xyz: [B, N, 3]. Input point clouds.
        features: [B, N, C]. Point features.
        centers: [B, L, 3]. Group centers.
        nn_idx: [B, N]. The indices of the nearest neighbors.

    Returns:
        torch.Tensor: [B, L, 3 + C]. Group features.
    """
    nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)  # [B, N, 3]
    dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
    nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)
    group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
    return group_feats


class VoronoiGrouper(nn.Module):
    """
    Point-SAM风格的Voronoi分词器（分组部分）。

    核心思想：
    1. FPS采样L个中心点
    2. 每个点分配给最近中心（1-NN），形成Voronoi图
    3. 计算相对位置特征（输出给VoronoiPatchEncoder进行编码和聚合）

    注意：编码和聚合由VoronoiPatchEncoder完成，与NNGrouper+PatchEncoderNN的模式一致。
    """

    def __init__(
        self,
        num_groups: int,
        in_channels: int = 4,  # 兼容接口，不再使用
        hidden_channels: int = 128,  # 兼容接口，不再使用
        out_channels: int = 256,  # 兼容接口，不再使用
        use_fps: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.use_fps = use_fps

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features (包含xyz的相对位置).

        Returns:
            dict: {
                'point_features': [B, N, 3+C]. 每个点的特征（相对位置+原始特征，未聚合）.
                'nn_idx': [B, N]. 每个点所属的Voronoi cell索引.
                'centers': [B, L, 3]. Voronoi中心点.
                'nn_assignment': [B, N]. 1-NN分配结果，用于decoder插值.
            }
        """
        with torch.no_grad():
            if self.use_fps:
                fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
                centers = batch_index_select(xyz, fps_idx, dim=1)
            else:
                centers = xyz[:, :self.num_groups]

            _, nn_idx = knn_points(xyz, centers, 1)
            nn_idx = nn_idx.squeeze(-1)

        # 计算相对位置特征（与NNGrouper一致）
        relative_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)
        dist = torch.linalg.norm(relative_xyz, dim=-1, keepdim=True)
        relative_xyz = relative_xyz / torch.clamp(dist, min=1e-8)

        # 拼接相对位置+距离+原始特征（与NNGrouper一致）
        point_features = torch.cat([relative_xyz, dist, features], dim=-1)

        return {
            'point_features': point_features,  # [B, N, 3+1+C] 未聚合的特征
            'nn_idx': nn_idx,  # [B, N] 每个点所属的cell索引
            'centers': centers,  # [B, L, 3]
            'nn_assignment': nn_idx,  # 用于decoder插值
        }


def voronoi_scatter_max(
    point_features: torch.Tensor,
    nn_idx: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    在Voronoi cell内使用scatter-max进行特征聚合。
    
    这是一个独立的函数，可以用于任意的point features聚合。
    
    Args:
        point_features: [B, N, D]. 每个点的特征.
        nn_idx: [B, N]. 每个点所属的cell索引.
        num_groups: 聚合后的特征数量.
    
    Returns:
        torch.Tensor: [B, num_groups, D]. 聚合后的特征.
    """
    batch_size, num_points, dim = point_features.shape
    output = torch.zeros(
        batch_size, num_groups, dim,
        device=point_features.device,
        dtype=point_features.dtype
    )
    output = torch.scatter_reduce(
        output,
        dim=1,
        index=nn_idx.unsqueeze(-1).expand(-1, -1, dim).clone(),
        src=point_features,
        reduce="amax",
    )
    return output


class VoronoiPatchEncoder(nn.Module):
    """
    基于Voronoi分词的高效Patch编码器。

    参考Point-SAM的PatchEncoderNN设计，使用两步scatter_reduce + concat：
    1. scatter_reduce max 将点特征聚合到cell级别
    2. gather回每个点
    3. concat [cell_max, point] 增强感受野
    4. conv2进一步编码
    5. 再次scatter_reduce max

    与KNN的PatchEncoder不同，这里每个点只属于一个cell，
    使用scatter_reduce代替group操作，更加高效。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list = None,
        use_gradient_checkpointing: bool = False,  # 暂时禁用以排查 ZeRO-3 NCCL 超时死锁
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, out_channels]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 与PatchEncoderNN类似的结构
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.GELU(),
            nn.Linear(hidden_channels[0], hidden_channels[0]),
        )
        # 注意：hidden[0]*2 因为有concat操作
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_channels[0] * 2, hidden_channels[1]),
            nn.LayerNorm(hidden_channels[1]),
            nn.GELU(),
            nn.Linear(hidden_channels[1], out_channels),
        )

    def _forward_impl(
        self,
        x_after_conv1: torch.Tensor,
        nn_idx: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """
        Args:
            x_after_conv1: [B, N, H]. conv1 之后的点特征（与 Point-SAM PatchEncoderNN 一致）.
            nn_idx: [B, N]. 每个点所属的Voronoi cell索引.
            num_groups: Voronoi cell数量.

        Returns:
            torch.Tensor: [B, num_groups, C_out]. Patch embeddings.
        """
        # Materialize the expand to avoid zero-stride views in scatter_reduce under FSDP
        nn_idx_expanded1 = nn_idx.unsqueeze(-1).expand(-1, -1, x_after_conv1.shape[-1]).clone()

        # 第一步scatter_reduce max聚合
        y = torch.zeros(
            x_after_conv1.shape[0], num_groups, x_after_conv1.shape[-1],
            device=x_after_conv1.device, dtype=x_after_conv1.dtype
        )
        y = torch.scatter_reduce(
            y, dim=1,
            index=nn_idx_expanded1,
            src=x_after_conv1,
            reduce="max",
        )

        # materialize nn_idx_expanded2 AFTER y is defined
        nn_idx_expanded2 = nn_idx.unsqueeze(-1).expand(-1, -1, y.shape[-1]).clone()

        # gather回每个点
        x_max = torch.gather(
            y, dim=1,
            index=nn_idx_expanded2
        )

        # concat [cell_max, point]
        x = torch.cat([x_max, x_after_conv1], dim=-1)

        # conv2编码
        x = self.conv2(x)

        # materialize nn_idx_expanded3 AFTER x is defined
        nn_idx_expanded3 = nn_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]).clone()

        # 再次scatter_reduce max
        y = torch.zeros(
            x.shape[0], num_groups, x.shape[-1],
            device=x.device, dtype=x.dtype
        )
        y = torch.scatter_reduce(
            y, dim=1,
            index=nn_idx_expanded3,
            src=x,
            reduce="max",
        )

        return y

    def forward(
        self,
        point_patches: torch.Tensor,
        nn_idx: torch.Tensor = None,
        num_groups: int = None,
    ) -> torch.Tensor:
        """
        Args:
            point_patches: [B, N, C_in]. 每个点的特征.
            nn_idx: [B, N]. 每个点所属的cell索引.
            num_groups: Voronoi cell数量.
        """
        x = self.conv1(point_patches)
        if self.training and self.use_gradient_checkpointing:
            return checkpoint(
                self._forward_impl,
                x,
                nn_idx,
                num_groups,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x, nn_idx, num_groups)


def compute_interp_weights(query: torch.Tensor, key: torch.Tensor, k=3, eps=1e-8):
    """Compute interpolation weights for each query point.

    Args:
        query: [B, Nq, 3]. Query points.
        key: [B, Nk, 3]. Key points.
        k: int. The number of nearest neighbors.
        eps: float. A small value to avoid division by zero.

    Returns:
        torch.Tensor: [B, Nq, K], indices of the k nearest neighbors in the key.
        torch.Tensor: [B, Nq, K], interpolation weights.
    """
    dist, idx = knn_points(query, key, k)
    inv_dist = 1.0 / torch.clamp(dist.square(), min=eps)
    normalizer = torch.sum(inv_dist, dim=2, keepdim=True)
    weight = inv_dist / normalizer  # [B, Nq, K]
    return idx, weight


def interpolate_features(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """
    Interpolates features based on the given index and weight.

    Memory-optimized version: processes in chunks to reduce peak memory usage.
    For very large tensors, uses sequential processing with explicit memory cleanup.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_keys, num_features).
        index (torch.Tensor): The index tensor of shape (batch_size, num_queries, K).
        weight (torch.Tensor): The weight tensor of shape (batch_size, num_queries, K).

    Returns:
        torch.Tensor: The interpolated features tensor of shape (batch_size, num_queries, num_features).
    """
    B, Nq, K = index.shape
    num_features = x.shape[-1]
    num_keys = x.shape[1]

    # Memory-efficient chunk size based on available memory
    # Process batch dimension to reduce peak memory
    max_batch_chunk = max(1, B // 4)  # Process 1/4 of batch at a time
    results = []

    for b_start in range(0, B, max_batch_chunk):
        b_end = min(b_start + max_batch_chunk, B)
        batch_size = b_end - b_start

        # Get chunk of x
        x_chunk = x[b_start:b_end]  # [batch_chunk, num_keys, num_features]
        index_chunk = index[b_start:b_end]  # [batch_chunk, Nq, K]
        weight_chunk = weight[b_start:b_end]  # [batch_chunk, Nq, K]

        # Compute flat index for this batch chunk
        batch_offset = torch.arange(batch_size, device=x.device).reshape(-1, 1, 1) * num_keys
        index_flat = (index_chunk + batch_offset).flatten()  # [batch_chunk * Nq * K]

        # Gather and compute interpolation for this chunk
        x_flat = x_chunk.flatten(0, 1)  # [batch_chunk * num_keys, num_features]
        _x = x_flat[index_flat].reshape(batch_size, Nq, K, num_features)
        result_chunk = (_x * weight_chunk.unsqueeze(-1)).sum(-2)

        results.append(result_chunk)

        # Explicit memory cleanup for this chunk
        del x_chunk, index_chunk, weight_chunk, x_flat, _x, index_flat

    # Concatenate results
    return torch.cat(results, dim=0)


def interpolate_features_original(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """
    Original interpolate_features - kept for reference.
    """
    B, Nq, K = index.shape
    batch_offset = torch.arange(B, device=x.device).reshape(-1, 1, 1) * x.shape[1]
    index_flat = (index + batch_offset).flatten()  # [B*Nq*K]
    _x = x.flatten(0, 1)[index_flat].reshape(B, Nq, K, x.shape[-1])
    return (_x * weight.unsqueeze(-1)).sum(-2)


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int):
    """Repeat slices along ``dim`` (same semantics as the old expand/flatten path).

    Do not implement this with ``expand``: under FSDP + autocast, expanded views can
    hit invalid storage (``setStorage ... out of bounds for storage of size 0``).
    ``torch.repeat_interleave`` materializes a proper tensor.
    """
    if repeats == 1:
        return x
    return torch.repeat_interleave(x, repeats, dim=dim)


@torch.no_grad()
def sample_prompts_adapter(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
    is_eval = False,
):
    """Select prompt sampler based on iou."""
    if pred_logits is None:
        return sample_fixed_points(
            points, gt_masks, pred_logits, threshold, from_error_region=True
        )
    else:
        batch_size, num_masks, _ = gt_masks.shape

        # if the batch iou is less than 0.5, use fixed sampler
        gt_masks_copy = gt_masks.reshape(batch_size * num_masks, -1)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold

        # ZeRO-3 / DDP：iou 由各 rank 本地 batch 决定时，若部分 rank 走 sample_prompts、
        # 部分走 sample_fixed_points，forward 子图不一致会在 DeepSpeed 参数 all_gather 上死锁
        # （与 hyperpoint_sam.forward 里 mask_refinement_iterations 的 broadcast 同理）。
        union = (gt_masks_copy | pred_masks).sum()
        use_sample_prompts = False
        if not is_eval and union.item() > 0:
            inter = (gt_masks_copy & pred_masks).sum()
            iou_f = (inter.float() / union.float()).item()
            use_sample_prompts = iou_f >= 1.0 - 1e-6

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dev = points.device
            t = torch.zeros(1, dtype=torch.long, device=dev)
            if dist.get_rank() == 0:
                t.fill_(1 if use_sample_prompts else 0)
            dist.broadcast(t, src=0)
            use_sample_prompts = bool(t.item())

        if not use_sample_prompts or is_eval:
            return sample_fixed_points(
                points, gt_masks, pred_logits, threshold, from_error_region=False
            )
        return sample_prompts(points, gt_masks, pred_logits, threshold)


@torch.no_grad()
def sample_prompts(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
):
    """Sample prompts from point clouds given ground-truth and predicted masks.

    Args:
        points: [B, N, 3]. Input point clouds.
        gt_masks: [B, M, N], bool. Ground-truth (binary) masks.
        pred_logits: A float tensor of shape [B*M, N]. Predicted logits.
            If None, the prompt points will be sampled from the ground-truth masks.

    Returns:
        torch.Tensor: [B*M, 1, 3]. Prompt points.
        torch.Tensor: [B*M, 1], bool. Prompt labels.
    """
    batch_size, num_masks, _ = gt_masks.shape

    # The prompt point will be sampled from the error region.
    if pred_logits is None:
        diff_masks = gt_masks
    else:
        pred_logits = pred_logits.reshape(batch_size, num_masks, -1)
        assert gt_masks.shape == pred_logits.shape, (gt_masks.shape, pred_logits.shape)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold
        diff_masks = gt_masks != pred_masks

    prompt_coords, prompt_labels = [], []
    for i in range(batch_size):
        for j in range(num_masks):
            diff_inds = torch.nonzero(diff_masks[i, j])  # [?, 1]
            if len(diff_inds) == 0:
                diff_inds = torch.nonzero(gt_masks[i, j])
            diff_inds = diff_inds.squeeze(1)  # [?]
            if diff_inds.numel() == 0:
                c, l = _fallback_prompt_point(points[i], gt_masks[i, j])
                prompt_coords.append(c)
                prompt_labels.append(l)
            else:
                r = torch.randint(
                    0, diff_inds.numel(), (1,), device=diff_inds.device
                )
                idx = diff_inds[r]
                prompt_coords.append(points[i][idx])
                prompt_labels.append(gt_masks[i, j][idx])

    prompt_coords = torch.stack(prompt_coords)
    prompt_labels = torch.stack(prompt_labels)
    return prompt_coords, prompt_labels


def _fallback_prompt_point(
    points_ij: torch.Tensor, gt_ij: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Border / Chamfer 采样失败时：优先从 GT 前景随机取一点，否则随机取点云上一点。"""
    fg = torch.nonzero(gt_ij, as_tuple=False).view(-1)
    if fg.numel() > 0:
        r = torch.randint(0, fg.numel(), (1,), device=fg.device)
        idx = fg[r]
    else:
        r = torch.randint(0, points_ij.shape[0], (1,), device=points_ij.device)
        idx = r
    return points_ij[idx].view(1, 3), gt_ij[idx].view(1)


def _ensure_prompt_pair(
    coords: Optional[torch.Tensor],
    label: Optional[torch.Tensor],
    points_ij: torch.Tensor,
    gt_ij: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if coords is not None and label is not None:
        return coords, label
    return _fallback_prompt_point(points_ij, gt_ij)


@torch.no_grad()
def sample_fixed_points(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
    from_error_region: bool = False,
):
    """Sample prompts from point clouds given ground-truth and predicted masks.

    Args:
        points: [B, N, 3]. Input point clouds.
        gt_masks: [B, M, N], bool. Ground-truth (binary) masks.
        pred_logits: A float tensor of shape [B*M, N]. Predicted logits.
            If None, the prompt points will be sampled from the ground-truth masks.

    Returns:
        torch.Tensor: [B*M, 1, 3]. Prompt points.
        torch.Tensor: [B*M, 1], bool. Prompt labels.
    """
    batch_size, num_masks, _ = gt_masks.shape

    # The prompt point will be sampled from the error region.
    if pred_logits is None:
        fn = gt_masks
        fp = torch.zeros_like(fn)
    else:
        pred_logits = pred_logits.reshape(batch_size, num_masks, -1)
        assert gt_masks.shape == pred_logits.shape, (gt_masks.shape, pred_logits.shape)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold
        fn = gt_masks & ~pred_masks
        fp = ~gt_masks & pred_masks

    prompt_points, prompt_labels = [], []
    if from_error_region:
        mask = fn | fp
        for i in range(batch_size):
            for j in range(num_masks):
                coords, label, _ = sample_furthest_points_from_border(
                    points[i], mask[i, j], gt_masks[i, j]
                )
                coords, label = _ensure_prompt_pair(
                    coords, label, points[i], gt_masks[i, j]
                )
                prompt_points.append(coords)
                prompt_labels.append(label)
    else:
        for i in range(batch_size):
            for j in range(num_masks):
                pprompt_coord, pprompt_label, pdist = (
                    sample_furthest_points_from_border(
                        points[i], fn[i, j], gt_masks[i, j]
                    )
                )
                nprompt_coord, nprompt_label, ndist = (
                    sample_furthest_points_from_border(
                        points[i], fp[i, j], gt_masks[i, j]
                    )
                )
                if pdist > ndist:
                    c, l = _ensure_prompt_pair(
                        pprompt_coord, pprompt_label, points[i], gt_masks[i, j]
                    )
                    prompt_points.append(c)
                    prompt_labels.append(l)
                elif ndist == -1:
                    pprompt_coord, pprompt_label, pdist = (
                        sample_furthest_points_from_border(
                            points[i], gt_masks[i, j], gt_masks[i, j]
                        )
                    )
                    c, l = _ensure_prompt_pair(
                        pprompt_coord, pprompt_label, points[i], gt_masks[i, j]
                    )
                    prompt_points.append(c)
                    prompt_labels.append(l)
                else:
                    c, l = _ensure_prompt_pair(
                        nprompt_coord, nprompt_label, points[i], gt_masks[i, j]
                    )
                    prompt_points.append(c)
                    prompt_labels.append(l)

    prompt_points = torch.stack(prompt_points)
    prompt_labels = torch.stack(prompt_labels)
    return prompt_points, prompt_labels


def sample_furthest_points_from_border(
    coords: torch.Tensor, lables: torch.Tensor, gt: torch.Tensor
):
    """
    Sample points from the border of the mask.

    Args:
        coords: [N, 3]. Input point clouds.
        lables: [N]. Point labels.
        gt: [N]. Ground-truth labels.
    """
    bg_inds = lables == 0
    fg_inds = lables == 1

    # if bg_inds or fg_inds is empty, return None
    if bg_inds.sum() == 0 or fg_inds.sum() == 0:
        return None, None, -1

    # torkit3d chamfer CUDA 仅支持 float32；bf16 autocast 下需显式提升精度
    xyz_fg = coords[fg_inds][None, ...].float()
    xyz_bg = coords[bg_inds][None, ...].float()
    min_dists, _ = chamfer_distance(xyz_fg, xyz_bg)

    # Sample the farthest points from the border
    center_idx = torch.argmax(min_dists)
    center_coords = coords[fg_inds][center_idx]
    center_dist = torch.max(min_dists)
    center_label = gt[fg_inds][center_idx]

    return center_coords[None, ...], center_label[None, ...], center_dist


class PatchEncoder(nn.Module):
    """Encode point patches following the PointNet structure for segmentation.
    
    Uses gradient checkpointing to reduce activation memory during training.
    
    Memory Optimization:
        - use_gradient_checkpointing: Wraps forward pass in torch.utils.checkpoint
        - checkpoint_policy: Controls what is checkpointed
            - "full": Checkpoint entire forward (maximum memory savings)
            - "split": Checkpoint conv1 and conv2 separately (balanced)
            - "auto": Automatic selection based on layer size
    """

    def __init__(
        self, 
        in_channels,
        out_channels,
        hidden_dims: list[int],
        use_gradient_checkpointing: bool = False,  # 暂时禁用以排查 ZeRO-3 NCCL 超时死锁
        # "full": checkpoint 整个 forward（ZeRO-3 兼容性最好）；"split": checkpoint conv2 only。
        # ⚠️ "split" 在 ZeRO-3 下与 checkpoint + hook 交互会触发死锁，改用 "full"。
        checkpoint_policy: str = "full",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_policy = checkpoint_policy

        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def _forward_conv1(self, point_patches: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv1 only (for checkpointing)."""
        return self.conv1(point_patches)

    def _forward_with_checkpoint(self, point_patches: torch.Tensor):
        """Forward pass wrapped in gradient checkpointing (both conv1 and conv2)."""
        if self.checkpoint_policy == "full":
            # _forward_full 已含 max-pool，返回 [B, L, C_out]；不可再对中间维做一次 max
            return checkpoint(
                self._forward_full,
                point_patches,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        if self.checkpoint_policy == "split":
            # conv1 外置，仅对 conv2 做 checkpoint（与 FSDP 需 use_reentrant=False）
            x = self._forward_conv1(point_patches)
            y = torch.max(x, dim=-2, keepdim=True).values
            x = torch.cat([y.expand_as(x), x], dim=-1)
            x = checkpoint(
                self._forward_conv2_only,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            return torch.max(x, dim=-2).values
        # auto / 其它：退化为无 checkpoint，避免错误递归
        return self._forward_without_checkpoint(point_patches)

    def _forward_full(self, point_patches: torch.Tensor):
        """Complete forward pass (used for full checkpointing)."""
        x = self.conv1(point_patches)  # [B, L, K, C_hidden]
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y

    def _forward_conv2_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv2 only (for checkpointing)."""
        return self.conv2(x)

    def forward(self, point_patches: torch.Tensor):
        # point_patches: [B, L, K, C_in]
        if self.training and self.use_gradient_checkpointing:
            y = self._forward_with_checkpoint(point_patches)
        else:
            y = self._forward_without_checkpoint(point_patches)
        return y

    def _forward_without_checkpoint(self, point_patches: torch.Tensor):
        """Forward pass without gradient checkpointing."""
        x = self.conv1(point_patches)  # [B, L, K, C_hidden]
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y
    
class PatchEncoderNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims: list[int], use_gradient_checkpointing: bool = False) -> None:  # 暂时禁用以排查 ZeRO-3 NCCL 超时死锁
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def _forward_impl(self, x: torch.Tensor, nn_idx: torch.Tensor, center_number: int) -> torch.Tensor:
        """Core forward (used by both regular and checkpointed paths)."""
        y = torch.zeros([x.shape[0], center_number, x.shape[-1]], device=x.device, dtype=x.dtype)
        y = torch.scatter_reduce(y, 1, nn_idx, x, "max")
        x_max = torch.zeros_like(x)
        x_max = torch.gather(y, 1, nn_idx.unsqueeze(-1).expand_as(y))
        x = torch.cat([x_max, x], dim=-1)
        x = self.conv2(x)
        y = torch.zeros([x.shape[0], center_number, x.shape[-1]], device=x.device, dtype=x.dtype)
        y = torch.scatter_reduce(y, 1, nn_idx, x, "max")
        return y

    def forward(self, point_patches: torch.Tensor, nn_idx: torch.Tensor, center_number: int) -> torch.Tensor:
        # point_patches: [B, N, C_in]
        if self.training and self.use_gradient_checkpointing:
            x = checkpoint(
                self._forward_impl,
                self.conv1(point_patches),
                nn_idx,
                center_number,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x = self.conv1(point_patches)
            x = self._forward_impl(x, nn_idx, center_number)
        return x
