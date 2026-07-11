# HPP-SAM Model Package
#
# Core architecture: HPPSAM (Hyperbolic Point Prompt Segment Anything Model)
# Key innovations:
#   - HyperPromptBranch: HRA (Hyperbolic Radius Adjustment) for prompt conditioning
#   - HyperCrossAttention: radius-conditioned hyperbolic cross-attention
#   - Mixed-manifold: Euclidean encoder + hyperbolic decoder

from .hyperpoint_sam import HPPSAM
from .transformer import TwoWayTransformer, TwoWayAttentionBlock, Attention, MLPBlock
from .mask_decoder import MaskDecoder, MaskDecoderHier, AuxInputs
from .hyper_ops import HyperOps, create_hyper_ops
from .hyper_prompt_branch import HyperPromptBranch, create_hyper_prompt_branch
from .hyper_cross_attention import HyperCrossAttention, create_hyper_cross_attention

__all__ = [
    # Core model
    "HPPSAM",
    # Transformer
    "TwoWayTransformer",
    "TwoWayAttentionBlock",
    "Attention",
    "MLPBlock",
    # Decoder
    "MaskDecoder",
    "MaskDecoderHier",
    "AuxInputs",
    # Hyperbolic operations
    "HyperOps",
    "create_hyper_ops",
    "HyperPromptBranch",
    "create_hyper_prompt_branch",
    "HyperCrossAttention",
    "create_hyper_cross_attention",
]
