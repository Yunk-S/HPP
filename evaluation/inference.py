import sys

sys.path.append(".")

import argparse
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed
import torch.nn as nn
from hpp_sam.utils.torch_utils import replace_with_fused_layernorm
from safetensors.torch import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="hpp", help="Hydra config name"
    )
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./pretrained/hpp/model.safetensors",
    )
    args, unknown_args = parser.parse_known_args()

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)

    # ---------------------------------------------------------------------------- #
    # Load pre-trained model
    # ---------------------------------------------------------------------------- #
    load_model(model, args.ckpt_path)

    # ---------------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------------- #
    model.eval()
    model.cuda()
    coords = ...
    colors = ...
    labels = ...

    # normalize coords
    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / coords.norm(dim=2, keepdim=True).max()

    # normalize colors
    colors = colors / 255

    # coords: [B, N, 3]
    # colors: [B, N, 3]
    # labels: [B, M, N] (bool, M = num masks per scene)
    data = {"coords": coords, "features": colors, "gt_masks": labels}
    outputs = model(**data, is_eval=True)

if __name__ == "__main__":
    main()
