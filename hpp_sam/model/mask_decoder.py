"""
HPP-SAM MaskDecoder with Hyperbolic Cross-Attention.

This module extends the standard MaskDecoder to support hyperbolic cross-attention
by accepting additional hyperbolic prompt parameters from HyperPromptBranch.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import compute_interp_weights, interpolate_features, repeat_interleave
from .hyper_ops import HyperOps


@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor
    features: torch.Tensor
    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        curvature: float = 0.01,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.hra_curvature = curvature
        self._hyper_ops = HyperOps(curvature=curvature, eps=1e-5)
        # Aligns with HyperPromptBranch.alpha: map learnable token embeddings into the ball.
        # Warm start 0.1 ensures mask tokens have meaningful hyperbolic radius.
        self.output_token_hyp_scale = nn.Parameter(torch.tensor(0.1))
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.output_upscaling = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def _output_upscaling_with_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Output upscaling with gradient checkpointing to save memory."""
        return self.output_upscaling(x)

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        multimask_output: bool,
        hyperbolic_prompt: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
        prompt_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given pointcloud and prompt embeddings.

        Arguments:
          pc_embeddings: the embeddings from the point cloud encoder
          pc_pe: positional encoding with the shape of pc_embeddings
          sparse_prompt_embeddings: the embeddings of the points and boxes
          dense_prompt_embeddings: the embeddings of the mask inputs
          multimask_output: Whether to return multiple masks or a single mask.
          hyperbolic_prompt: hyperbolic prompt from HyperPromptBranch.
          temperature: attention temperature τ(r_i).
          radius: radius r_i.
          prompt_coords: [B_sparse, P, 3] 3D positions for sparse prompts (Ball Query).
        """
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks, iou_pred = self.predict_masks(
            pc_embeddings=pc_embeddings,
            pc_pe=pc_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            aux_inputs=aux_inputs,
            mask_slice=mask_slice,
            hyperbolic_prompt=hyperbolic_prompt,
            temperature=temperature,
            radius=radius,
            prompt_coords=prompt_coords,
        )

        return masks, iou_pred

    def predict_masks(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        mask_slice: slice = None,
        hyperbolic_prompt: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
        prompt_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        # Materialize expand: when dim=1 is expanded from 1->T, stride=0 under FSDP.
        # clone() makes it a proper tensor that works in subsequent torch.cat / other ops.
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        ).clone()
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        repeats = tokens.shape[0] // pc_embeddings.shape[0]
        src = repeat_interleave(pc_embeddings, repeats, dim=0)
        pos_src = repeat_interleave(pc_pe, repeats, dim=0)
        src = src + dense_prompt_embeddings

        hyperbolic_tokens = None
        temperature_expanded = None
        radius_expanded = None
        num_prefix = 1 + self.num_mask_tokens

        if hyperbolic_prompt is not None:
            scaled_output = self.output_token_hyp_scale * output_tokens
            output_tokens_h = self._hyper_ops.exp0(scaled_output)
            hyperbolic_tokens = torch.cat((output_tokens_h, hyperbolic_prompt), dim=1)
            temperature_pad = temperature
            radius_pad = radius
            if temperature is not None:
                # Materialize expand: dim=1 expanded from 1->num_prefix (stride=0 under FSDP)
                prefix_t = temperature.mean(dim=1, keepdim=True).expand(
                    -1, num_prefix
                ).clone()
                temperature_pad = torch.cat([prefix_t, temperature], dim=1)
            if radius is not None:
                # Materialize expand: dim=1 expanded from 1->num_prefix (stride=0 under FSDP)
                prefix_r = radius.mean(dim=1, keepdim=True).expand(-1, num_prefix).clone()
                radius_pad = torch.cat([prefix_r, radius], dim=1)
            if temperature_pad is not None:
                temp_repeats = tokens.shape[0] // temperature_pad.shape[0]
                temperature_expanded = repeat_interleave(
                    temperature_pad, temp_repeats, dim=0
                )
            if radius_pad is not None:
                radius_repeats = tokens.shape[0] // radius_pad.shape[0]
                radius_expanded = repeat_interleave(radius_pad, radius_repeats, dim=0)

        pc_pos_expanded = repeat_interleave(aux_inputs.centers, repeats, dim=0)
        prompt_pos_expanded = None
        if hyperbolic_tokens is not None and prompt_coords is not None:
            # Materialize expand: dim=1 expanded from 1->num_prefix (stride=0 under FSDP)
            prefix_pos = aux_inputs.centers.mean(dim=1, keepdim=True).expand(
                -1, num_prefix, -1
            ).clone()
            prefix_pos = repeat_interleave(prefix_pos, repeats, dim=0)
            prompt_pos_expanded = torch.cat([prefix_pos, prompt_coords], dim=1)

        hs, src = self.transformer(
            src,
            pos_src,
            tokens,
            hyperbolic_prompt=hyperbolic_tokens,
            temperature=temperature_expanded,
            radius=radius_expanded,
            pc_pos=pc_pos_expanded,
            prompt_pos=prompt_pos_expanded,
        )
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        coords = aux_inputs.coords
        centers = aux_inputs.centers
        interp_index = aux_inputs.interp_index
        interp_weight = aux_inputs.interp_weight
        if interp_index is None or interp_weight is None:
            with torch.no_grad():
                interp_index, interp_weight = compute_interp_weights(coords, centers)
            aux_inputs.interp_index = interp_index
            aux_inputs.interp_weight = interp_weight

        _repeats = tokens.shape[0] // interp_index.shape[0]
        interp_index = repeat_interleave(interp_index, _repeats, dim=0)
        interp_weight = repeat_interleave(interp_weight, _repeats, dim=0)

        # Memory-efficient interpolation using chunked processing
        interp_embedding = interpolate_features(src, interp_index, interp_weight)

        # Use gradient checkpointing for output upscaling to save activation memory
        if self.training and self.use_gradient_checkpointing:
            upscaled_embedding = checkpoint(
                self._output_upscaling_with_checkpoint,
                interp_embedding,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            upscaled_embedding = self.output_upscaling(interp_embedding)

        del interp_embedding

        hyper_in_list: List[torch.Tensor] = []
        mask_indices = list(range(self.num_mask_tokens))
        if mask_slice is not None:
            mask_indices = mask_indices[mask_slice]
        for i in mask_indices:
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = hyper_in @ upscaled_embedding.transpose(-1, -2)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if mask_slice is not None:
            iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class MaskDecoderHier(nn.Module):
    """Hierarchical upscaling decoder for HPP-SAM."""

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        encoder_dim: int = 128,
        curvature: float = 0.01,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.hra_curvature = curvature
        self._hyper_ops = HyperOps(curvature=curvature, eps=1e-5)
        self.output_token_hyp_scale = nn.Parameter(torch.tensor(0.01))

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 2, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.output_upscaling2 = nn.Sequential(
            nn.Linear(transformer_dim + encoder_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.output_upscaling1 = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.LayerNorm(transformer_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_dim // 2, transformer_dim // 2),
            nn.GELU(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs1: AuxInputs,
        aux_inputs2: AuxInputs,
        multimask_output: bool,
        hyperbolic_prompt: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks, iou_pred = self.predict_masks(
            pc_embeddings=pc_embeddings,
            pc_pe=pc_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            aux_inputs1=aux_inputs1,
            aux_inputs2=aux_inputs2,
            mask_slice=mask_slice,
            hyperbolic_prompt=hyperbolic_prompt,
            temperature=temperature,
            radius=radius,
        )

        return masks, iou_pred

    def predict_masks(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs1: AuxInputs,
        aux_inputs2: AuxInputs,
        mask_slice: slice = None,
        hyperbolic_prompt: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        # Materialize expand: when dim=1 is expanded from 1->T, stride=0 under FSDP.
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        ).clone()
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        repeats = tokens.shape[0] // pc_embeddings.shape[0]
        src = repeat_interleave(pc_embeddings, repeats, dim=0)
        pos_src = repeat_interleave(pc_pe, repeats, dim=0)
        src = src + dense_prompt_embeddings

        hyperbolic_tokens = None
        temperature_expanded = None
        radius_expanded = None
        
        if hyperbolic_prompt is not None:
            scaled_output = self.output_token_hyp_scale * output_tokens
            output_tokens_h = self._hyper_ops.exp0(scaled_output)
            hyperbolic_tokens = torch.cat((output_tokens_h, hyperbolic_prompt), dim=1)
            if temperature is not None:
                temp_repeats = tokens.shape[0] // temperature.shape[0]
                temperature_expanded = repeat_interleave(temperature, temp_repeats, dim=0)
            if radius is not None:
                radius_repeats = tokens.shape[0] // radius.shape[0]
                radius_expanded = repeat_interleave(radius, radius_repeats, dim=0)

        hs, src = self.transformer(
            src, 
            pos_src, 
            tokens,
            hyperbolic_prompt=hyperbolic_tokens,
            temperature=temperature_expanded,
            radius=radius_expanded,
        )
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        interp_embedding = self.upscale_features(src, aux_inputs2, concat_feats=True)
        upscaled_embedding = self.output_upscaling2(interp_embedding)
        interp_embedding = self.upscale_features(upscaled_embedding, aux_inputs1)
        upscaled_embedding = self.output_upscaling1(interp_embedding)

        hyper_in_list: List[torch.Tensor] = []
        mask_indices = list(range(self.num_mask_tokens))
        if mask_slice is not None:
            mask_indices = mask_indices[mask_slice]
        for i in mask_indices:
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = hyper_in @ upscaled_embedding.transpose(-1, -2)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if mask_slice is not None:
            iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def upscale_features(
        self, src: torch.Tensor, aux_inputs: AuxInputs, concat_feats: bool = False
    ):
        coords = aux_inputs.coords
        centers = aux_inputs.centers
        interp_index = aux_inputs.interp_index
        interp_weight = aux_inputs.interp_weight
        if interp_index is None or interp_weight is None:
            with torch.no_grad():
                interp_index, interp_weight = compute_interp_weights(coords, centers)
            aux_inputs.interp_index = interp_index
            aux_inputs.interp_weight = interp_weight

        _repeats = src.shape[0] // interp_index.shape[0]
        interp_index = repeat_interleave(interp_index, _repeats, dim=0)
        interp_weight = repeat_interleave(interp_weight, _repeats, dim=0)

        interp_embedding = interpolate_features(src, interp_index, interp_weight)
        if concat_feats:
            features = repeat_interleave(aux_inputs.features, _repeats, dim=0)
            interp_embedding = torch.cat((interp_embedding, features), dim=-1)
        return interp_embedding
