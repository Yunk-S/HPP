import timm
import torch
import torch.nn as nn

from .point_cloud_encoder import PointCloudEncoder
from .prompt_encoder import MaskEncoder
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
from .hyperpoint_sam import HPPSAM


def build_hpp_sam(cfg):
    """Build HPP-SAM model (Hyperbolic Point Prompt Segment Anything Model).
    
    This function builds the HPP-SAM model with:
    - HyperPromptBranch: Hyperbolic prompt transformation with HRA
    - HyperCrossAttention: Radius-conditioned cross-attention
    
    Args:
        cfg: Configuration object with the following attributes:
            - encoder_model_name: timm model name for point transformer
            - drop_path_rate: dropout path rate
            - encoder_embed_dim: embedding dimension
            - encoder_num_group: number of point groups
            - encoder_group_size: group size for point grouping
            - encoder_patch_dropout: patch dropout rate
            - prompt_embed_dim: prompt embedding dimension
            - decoder_trans_depth: transformer decoder depth
            - decoder_embed_dim: decoder embedding dimension
            - decoder_num_head: number of attention heads
            - decoder_mlp_dim: MLP dimension in decoder
            - decoder_trans_dim: transformer dimension for decoder
            - prompt_iters: number of prompt iterations
            - hra_curvature: Hyperbolic curvature c. Default 0.01.
            - hra_type: HRA type, "diagonal" or "block_diagonal". Default "block_diagonal".
            - block_size: Block size for block-diagonal HRA. Default 8.
            - tau_min: Minimum temperature. Default 0.1.
            - tau_max: Maximum temperature. Default 2.0.
            - use_hyperbolic_cross_attn: Enable hyperbolic cross-attention. Default True.
    
    Returns:
        HPPSAM: The built HPP-SAM model
    """
    point_transformer = timm.create_model(
        cfg.encoder_model_name, drop_path_rate=cfg.drop_path_rate
    )
    from .point_cloud_encoder import PatchEmbed
    patch_embed = PatchEmbed(
        in_channels=6,
        out_channels=cfg.encoder_embed_dim,
        num_patches=cfg.encoder_num_group,
        patch_size=cfg.encoder_group_size,
        radius=cfg.get("encoder_radius", None),
    )
    pc_encoder = PointCloudEncoder(
        patch_embed=patch_embed,
        transformer=point_transformer,
        embed_dim=cfg.encoder_embed_dim,
        patch_drop_rate=cfg.encoder_patch_dropout,
    )
    prompt_encoder = MaskEncoder(
        embed_dim=cfg.prompt_embed_dim,
    )
    
    hra_curvature = getattr(cfg, 'hra_curvature', 0.01)
    use_hyp = getattr(cfg, 'use_hyperbolic_cross_attn', True)
    
    # 显存优化参数：从 mask_decoder.transformer 配置中读取（兼容 Hydra 内联配置）
    decoder_cfg = cfg.mask_decoder.get('transformer', {}) if hasattr(cfg, 'mask_decoder') else {}
    q_chunk_size = decoder_cfg.get('q_chunk_size', 128)
    k_chunk_size = decoder_cfg.get('k_chunk_size', 128)
    use_ball_query = decoder_cfg.get('use_ball_query', True)
    k_neighbors = decoder_cfg.get('k_neighbors', 64)

    two_way_transformer = TwoWayTransformer(
        cfg.decoder_trans_depth,
        cfg.decoder_embed_dim,
        cfg.decoder_num_head,
        cfg.decoder_mlp_dim,
        use_hyperbolic_cross_attn=use_hyp,
        curvature=hra_curvature,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        use_ball_query=use_ball_query,
        k_neighbors=k_neighbors,
    )
    mask_decoder = MaskDecoder(
        cfg.decoder_trans_dim, two_way_transformer, curvature=hra_curvature
    )
    
    hra_type = getattr(cfg, 'hra_type', 'block_diagonal')
    block_size = getattr(cfg, 'block_size', 8)
    tau_min = getattr(cfg, 'tau_min', 0.1)
    tau_max = getattr(cfg, 'tau_max', 2.0)
    
    model = HPPSAM(
        pc_encoder=pc_encoder,
        mask_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        prompt_iters=cfg.prompt_iters,
        hra_curvature=hra_curvature,
        hra_type=hra_type,
        block_size=block_size,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    return model


# unit test
if __name__ == "__main__":
    point_transformer: nn.Module = timm.create_model("eva02_giant_patch14_448")
    print(torch.backends.cuda.flash_sdp_enabled())
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    print(torch.backends.cuda.math_sdp_enabled())
    print(point_transformer)
