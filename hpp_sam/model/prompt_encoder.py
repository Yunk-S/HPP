from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from torkit3d.nn.functional import batch_index_select

from torch.utils.checkpoint import checkpoint

from .common import PatchEncoder, group_with_centers_and_knn


def expand_copy(tensor: torch.Tensor, *dims) -> torch.Tensor:
    """Expand tensor and materialize the result (FSDP/AMP-safe).

    Unlike ``tensor.expand(...).reshape(...)`` which can produce a view with
    invalid storage under FSDP sharding + autocast, this function always returns
    a contiguous tensor with proper storage.
    """
    return tensor.unsqueeze(0).expand(*dims).clone()


def voronoi_mask_encoder_forward(
    self,
    masks: torch.Tensor,
    coords: torch.Tensor,
    nn_assignment: torch.Tensor,
) -> torch.Tensor:
    """
    Voronoi模式下的mask编码。

    参考Point-SAM的MaskEncoderNN实现：
    1. 计算每个点到其所属cell中心的相对位置
    2. 拼接 [mask, relative_xyz, dist] 作为点特征
    3. 使用MLP提取点特征
    4. 使用scatter_reduce进行cell级别的max聚合
    5. 最终输出每个cell的聚合特征

    Args:
        masks: [B * M, N], float. Mask inputs.
        coords: [B, N, 3]. Point coordinates.
        nn_assignment: [B, N]. 每个点所属的Voronoi cell索引.

    Returns:
        torch.Tensor: [B * M, L, embed_dim]. Dense embeddings.
    """
    batch_size, num_points, _ = coords.shape
    # 分布式：各 rank 的 Voronoi 分配 max 索引可能不同 → num_groups 不一致会使
    # 下游张量形状分叉，ZeRO-3 集体通信与部分 rank 卡在 mask_encoder 内表现一致。
    _local_max = nn_assignment.max().to(dtype=torch.float32, device=coords.device)
    if dist.is_available() and dist.is_initialized():
        _mv = _local_max.reshape(1).clone()
        dist.all_reduce(_mv, op=dist.ReduceOp.MAX)
        num_groups = int(_mv.item()) + 1
    else:
        num_groups = int(_local_max.item()) + 1

    if masks is None:
        # [1, 1, D] -> [B, L, D]; use repeat_interleave to materialize (FSDP-safe)
        dense_embeddings = torch.repeat_interleave(
            self.no_mask_embed.weight.reshape(1, 1, -1),
            batch_size * num_groups,
            dim=0,
        ).reshape(batch_size, num_groups, -1)
        return dense_embeddings

    masks = masks.detach()
    mask_batch = masks.shape[0] // batch_size  # M

    # ========== 1. 扩展数据维度 ==========
    # nn_assignment: [B, N] -> [B*M, N]; materialize to avoid zero-stride views under FSDP
    nn_assignment_expanded = nn_assignment.unsqueeze(0).expand(mask_batch, -1, -1).clone()
    nn_assignment_expanded = nn_assignment_expanded.reshape(mask_batch * batch_size, num_points)

    # coords: [B, N, 3] -> [B*M, N, 3]; materialize to avoid zero-stride views under FSDP
    coords_expanded = coords.unsqueeze(0).expand(mask_batch, -1, -1, -1).clone()
    coords_expanded = coords_expanded.reshape(mask_batch * batch_size, num_points, 3)

    # masks: [B*M, N] -> [B*M, N, 1]
    masks_expanded = masks.unsqueeze(-1)

    # centers: 使用scatter_reduce计算每个Voronoi cell的中心
    # centers: [B, L, 3]
    nn_idx_expanded_for_centers = nn_assignment.unsqueeze(-1).expand(-1, -1, 3).clone()  # [B, N, 3]
    centers_temp = torch.zeros(
        batch_size, num_groups, 3,
        device=coords.device, dtype=coords.dtype
    )
    centers_temp = torch.scatter_reduce(
        centers_temp,
        dim=1,
        index=nn_idx_expanded_for_centers,
        src=coords,
        reduce="mean",
    )

    # centers_expanded: [B, L, 3] -> [B*M, L, 3]; materialize to avoid zero-stride views under FSDP
    centers_expanded = centers_temp.unsqueeze(0).expand(mask_batch, -1, -1, -1).clone()
    centers_expanded = centers_expanded.reshape(mask_batch * batch_size, num_groups, 3)

    # ========== 2. 计算点特征 [mask, relative_xyz, dist] ==========
    # 使用 batch_index_select 获取每个点所属的中心坐标
    # nn_assignment_expanded: [B*M, N], 值范围 [0, L-1]
    # materialize the expand to avoid zero-stride views under FSDP
    point_centers = torch.gather(
        centers_expanded,
        dim=1,
        index=nn_assignment_expanded.unsqueeze(-1).expand(-1, -1, 3).clone()
    )  # [B*M, N, 3]

    # 相对位置
    relative_xyz = coords_expanded - point_centers  # [B*M, N, 3]
    dist = torch.linalg.norm(relative_xyz, dim=-1, keepdim=True)  # [B*M, N, 1]
    relative_xyz = relative_xyz / torch.clamp(dist, min=1e-8)

    # 拼接点特征: [mask(1) + rel_xyz(3) + dist(1)] = 5 channels
    point_features = torch.cat([masks_expanded, relative_xyz, dist], dim=-1)  # [B*M, N, 5]

    # ========== 3. MLP编码 + scatter-max聚合 ==========
    # 使用 VoronoiPatchEncoder 进行编码和聚合
    # 注意：VoronoiPatchEncoder 期望输入 [B*M, N, C] 并输出 [B*M, L, D]
    dense_embeddings = self.patch_encoder(point_features, nn_assignment_expanded, num_groups)

    return dense_embeddings


# 辅助函数：Voronoi模式的patch编码（与PatchEncoderNN类似）
class VoronoiPatchEncoderForMask(nn.Module):
    """
    Voronoi模式的Patch编码器，专用于MaskEncoder。

    流程：
    1. MLP编码每个点的特征
    2. scatter_reduce进行cell级别的max聚合
    3. 最终输出每个cell的聚合特征

    参考Point-SAM的MaskEncoderNN设计。
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
            hidden_channels = [256, out_channels]  # 与PatchEncoderNN类似的设计

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 与MaskEncoderNN类似的MLP结构
        self.first_nn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.GELU(),
        )
        self.second_nn = nn.Sequential(
            nn.Linear(hidden_channels[0], hidden_channels[1]),
            nn.LayerNorm(hidden_channels[1]),
            nn.GELU(),
            nn.Linear(hidden_channels[1], out_channels),
        )

    def _forward_impl(
        self,
        point_features: torch.Tensor,
        nn_idx: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """
        Args:
            point_features: [B*M, N, C_in]. 每个点的特征.
            nn_idx: [B*M, N]. 每个点所属的cell索引.
            num_groups: cell数量.

        Returns:
            torch.Tensor: [B*M, num_groups, C_out]. 每个cell的聚合特征.
        """
        # 第一步MLP编码
        feature = self.first_nn(point_features)  # [B*M, N, hidden[0]]

        # scatter_reduce进行cell级别的max聚合
        # 输出: [B*M * num_groups, hidden[0]]
        aggregated = torch.zeros(
            feature.shape[0] * num_groups, feature.shape[-1],
            device=feature.device,
            dtype=feature.dtype,
        )
        # 将nn_idx展平到[0, B*M*num_groups)范围
        batch_idx = torch.arange(
            feature.shape[0], device=feature.device
        ).unsqueeze(1).unsqueeze(-1)  # [B*M, 1, 1]
        group_idx = nn_idx.unsqueeze(-1).expand(-1, -1, feature.shape[-1]).clone()  # [B*M, N, hidden]
        scatter_idx = (batch_idx * num_groups + group_idx).flatten(0, 1)

        aggregated = torch.scatter_reduce(
            aggregated,
            dim=0,
            index=scatter_idx,
            src=feature.flatten(0, 1),
            reduce="amax",
        )  # [B*M*num_groups, hidden]

        # reshape回 [B*M, num_groups, hidden]
        aggregated = aggregated.view(feature.shape[0], num_groups, -1)

        # 第二步MLP
        dense_embeddings = self.second_nn(aggregated)  # [B*M, num_groups, out_channels]

        return dense_embeddings

    def forward(
        self,
        point_features: torch.Tensor,
        nn_idx: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        if self.training and self.use_gradient_checkpointing:
            return checkpoint(
                self._forward_impl,
                point_features,
                nn_idx,
                num_groups,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(point_features, nn_idx, num_groups)


# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)


class PointEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Embeds point prompts.

        Args:
            points: [..., 3]. Point coordinates.
            labels: [...], integer (or boolean). Point labels.

        Returns:
            torch.Tensor: [..., embed_dim]. Embedded points.
        """
        assert points.shape[:-1] == labels.shape
        point_embedding = self.pe_layer.forward(points)
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding


class MaskEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        in_channels=4,
        radius=None,
        centralize_features=False,
        use_gradient_checkpointing=False,  # 暂时禁用以排查 ZeRO-3 NCCL 超时死锁
        # Voronoi模式参数
        use_voronoi=False,
        voronoi_hidden_channels=128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels  # (x, y, z, logit)
        self.radius = radius
        self.centralize_features = centralize_features
        self.use_voronoi = use_voronoi

        if use_voronoi:
            self.patch_encoder = VoronoiPatchEncoderForMask(
                in_channels=in_channels + 4,  # mask + xyz + dist = 1 + 3 + 1 = 5
                out_channels=embed_dim,
                hidden_channels=[256, embed_dim],  # 与MaskEncoderNN类似的设计
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            self.patch_encoder = PatchEncoder(
                in_channels, embed_dim, [128, 512], 
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        masks: Union[torch.Tensor, None],
        coords: torch.Tensor,
        centers: torch.Tensor,
        knn_idx: torch.Tensor,
        center_idx: torch.Tensor = None,
        nn_assignment: torch.Tensor = None,
    ) -> torch.Tensor:
        """Embeds mask inputs.

        Args:
            masks: [B * M, N], float. Mask inputs.
            coords: [B, N, 3]. Point coordinates.
            centers: [B, L, 3]. Center coordinates.
            knn_idx: [B, L, K]. KNN indices (KNN模式).
            center_idx: [B, L]. Index of center in the point cloud.
            nn_assignment: [B, N]. 每个点所属的Voronoi cell索引 (Voronoi模式).

        Returns:
            torch.Tensor: [B * M, L, embed_dim]. Dense embeddings.
        """
        if masks is None:
            # Materialize the expand to avoid zero-stride views under FSDP
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0], centers.shape[1], -1
            ).clone()
        else:
            masks = masks.detach()
            if self.use_voronoi and nn_assignment is not None:
                # Voronoi模式
                return voronoi_mask_encoder_forward(self, masks, coords, nn_assignment)
            else:
                # KNN模式（默认）
                patches = group_with_centers_and_knn(
                    coords,
                    masks.unsqueeze(-1),
                    centers,
                    knn_idx,
                    radius=self.radius,
                    center_idx=center_idx,
                    centralize_features=self.centralize_features,
                )
                dense_embeddings = self.patch_encoder(patches)
        return dense_embeddings


class MaskEncoderHier(nn.Module):
    """PointNet++ style with hierarchical grouping."""

    def __init__(
        self, 
        embed_dim, 
        in_channels=4,
        radius: list[float] = None,
        use_gradient_checkpointing=False,  # 暂时禁用以排查 ZeRO-3 NCCL 超时死锁
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels  # (x, y, z, logit)
        self.radius = radius

        self.patch_encoder1 = PatchEncoder(
            in_channels, 128, [64, 128],
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.patch_encoder2 = PatchEncoder(
            128 + 3, embed_dim, [128, 256],
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        masks: Union[torch.Tensor, None],
        coords: torch.Tensor,
        centers1: torch.Tensor,
        knn_idx1: torch.Tensor,
        centers2: torch.Tensor,
        knn_idx2: torch.Tensor,
    ) -> torch.Tensor:
        if masks is None:
            # Materialize the expand to avoid zero-stride views under FSDP
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers2.shape[0], centers2.shape[1], -1
            ).clone()
            return dense_embeddings
        else:
            masks = masks.detach()
            patches1 = group_with_centers_and_knn(
                coords,
                masks.unsqueeze(-1),
                centers1,
                knn_idx1,
                radius=self.radius[0] if self.radius else None,
            )
            x1 = self.patch_encoder1(patches1)

            patches2 = group_with_centers_and_knn(
                centers1,
                x1,
                centers2,
                knn_idx2,
                radius=self.radius[1] if self.radius else None,
            )
            x2 = self.patch_encoder2(patches2)
            return [x1, x2]


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x) + x


class ResMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GroupNN(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, centers, feats, idx, mask_batch):
        batch_size, num_points, _ = xyz.shape
        neighborhood = xyz.flatten(0, 1) - centers.flatten(0, 1)[idx]  # [B, N, 3]
        neighborhood = neighborhood.view(batch_size, num_points, 3)
        neighborhood = torch.repeat_interleave(neighborhood, mask_batch, 0)

        # normalize relative position
        dist = torch.linalg.norm(
            neighborhood, dim=-1, keepdim=True, ord=2, dtype=xyz.dtype
        )
        neighborhood = neighborhood / (dist + 1e-8)

        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        )
        idx = idx.view(batch_size, num_points) - idx_base
        idx_base = (
            torch.arange(0, batch_size * mask_batch, device=xyz.device).view(-1, 1)
            * self.num_group
        )
        idx = torch.repeat_interleave(idx, mask_batch, 0) + idx_base
        idx = idx.view(-1)
        neighborhood_feats = torch.cat(
            [feats.unsqueeze(-1), neighborhood, dist], dim=-1
        )
        return neighborhood_feats, idx


class MaskEncoderNN(nn.Module):
    def __init__(self, encoder_channel, num_group):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.num_group = num_group
        self.first_nn = nn.Linear(5, 1024)
        self.second_nn = ResMlp(1024, 1024, self.encoder_channel, 3)
        self.no_mask_embed = nn.Embedding(1, encoder_channel)

    def forward(self, point_groups, idx, centers, xyz):
        """
        point_groups : B N 4
        idx : B N
        -----------------
        feature_global : B G C
        """
        if point_groups is None:
            # Materialize the expand to avoid zero-stride views under FSDP
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0], centers.shape[1], -1
            ).clone()
            return dense_embeddings

        point_groups = point_groups.unsqueeze(-1) 
        bs, n, _ = point_groups.shape

        # encoder
        nbr_xyz = xyz - batch_index_select(centers, idx, dim=1)  # [B, N, 3]
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = torch.repeat_interleave(nbr_xyz, point_groups.shape[0] // nbr_xyz.shape[0], 0)
        dist = torch.repeat_interleave(dist, point_groups.shape[0] // dist.shape[0], 0)
        idx = torch.repeat_interleave(idx, point_groups.shape[0] // idx.shape[0], 0)
        point_groups = torch.cat([point_groups, nbr_xyz, dist], dim=-1)
        feature = self.first_nn(point_groups)  # B N 256
        aggregrate_feature = torch.zeros(
            [bs * self.num_group, feature.shape[-1]],
            device=point_groups.device,
            dtype=point_groups.dtype,
        )
        # Materialize expand used as index to avoid scatter_reduce issues under FSDP
        scatter_idx = idx.unsqueeze(-1).expand(-1, feature.shape[-2], feature.shape[-1]).clone().flatten(0, 1)
        aggregrate_feature = torch.scatter_reduce(
            aggregrate_feature,
            0,
            scatter_idx,
            feature.flatten(0, 1),
            "amax",
        )
        feature = self.second_nn(aggregrate_feature)  # B G 1024
        feature_global = feature.view(bs, self.num_group, -1)  # B G 1024
        return feature_global


class PromptEncoderNN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_group: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # mask encoder
        self.group_divider = GroupNN(num_group, group_size)
        self.mask_encoder = MaskEncoderNN(embed_dim, num_group)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        point_embedding = self.pe_layer.forward(points)
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def embed_masks(
        self,
        xyz: torch.Tensor,
        centers: torch.Tensor,
        masks: torch.Tensor,
        idx: torch.Tensor,
        mask_batch: int = 1,
    ) -> torch.Tensor:
        """Embeds mask inputs."""
        if masks is None:
            # Materialize the expand to avoid zero-stride views under FSDP
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0] * mask_batch, centers.shape[1], -1
            ).clone()
            return dense_embeddings
        masks = masks.detach()
        grouped_masks, idx = self.group_divider(xyz, centers, masks, idx, mask_batch)
        # grouped_masks: [BBM, N, 4], idx: [BBM, N]
        dense_embeddings = self.mask_encoder(grouped_masks, idx, centers, xyz)
        return dense_embeddings
