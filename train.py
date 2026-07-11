import argparse
import faulthandler
import json
import os
import shutil
import signal
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.accelerator import ProjectConfiguration
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed, tqdm
from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import ConcatDataset, DataLoader

import wandb
from hpp_sam.datasets.transforms import Compose
from hpp_sam.model.loss import compute_iou
from hpp_sam.utils.part_benchmark_metrics import (
    BENCHMARK_CLICKS,
    compute_benchmark_from_aux,
    compute_benchmark_from_epoch_ious,
    format_benchmark_for_postfix,
    print_benchmark_metric_help,
)
from hpp_sam.utils.hang_debug import forward_hang_probe
from hpp_sam.utils.torch_utils import replace_with_fused_layernorm, worker_init_fn

# Import learning diagnostics and regularization
try:
    from hpp_sam.utils.learning_diagnostics import (
        LearningDiagnostics,
        compute_mask_diversity,
        check_learning_vs_memorization,
    )
    from hpp_sam.model.regularization import (
        AdaptiveRegularization,
        create_adaptive_regularizer,
    )
    HAS_LEARNING_DIAGNOSTICS = True
except ImportError as e:
    HAS_LEARNING_DIAGNOSTICS = False
    print(f"[Warning] Learning diagnostics not available: {e}")


class _SkipBatch(Exception):
    """非有限 loss 等：跳过当前 batch，不中断 epoch。"""

class _BadBatchDetected(Exception):
    """模型 forward/backward 遇到无法通过 skip batch 恢复的错误（如非法显存访问）。
    
    区别于 _SkipBatch：_SkipBatch 是已知的可控跳过；_BadBatchDetected 表示
    某个特定 batch 的数据/状态触发了一个需要排查的 bug。
    """


def _is_recoverable_cuda_runtime_error(exc: BaseException) -> bool:
    """仅将「换一批/清缓存可能恢复」的显存不足视为可跳过。

    旧实现用 ``"cuda" in str(exc)`` 或 ``illegal memory`` 泛匹配，会把
    **非法显存访问**、launch failure 等也当成可恢复错误并静默跳过。
    这类错误跳过**不能**修复 GPU 上下文，后续几乎每个算子都会再报错，
    表现为 tqdm 长时间 ``SKIP:cuda runtime`` 且 ``gstep``/``micro`` 卡住。

    真正可尝试跳过的情况主要是 OOM（峰值显存波动、个别样本过大）。
    """
    try:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    return any(m in msg for m in oom_markers)


def _hang_debug_level(cfg) -> int:
    """死锁/挂起调试等级。

    - 环境变量 ``HPP_HANG_DEBUG``：``1`` / ``true`` 仅前若干 batch 打全链路；``2`` 每个 batch 都打（日志极多）。
    - 配置 ``hang_debug: 1`` 或 ``2`` 可覆盖（与 env 取较大值）。
    """
    env = os.environ.get("HPP_HANG_DEBUG", "").strip().lower()
    env_lvl = 0
    if env in ("2", "all", "verbose"):
        env_lvl = 2
    elif env in ("1", "true", "on", "yes"):
        env_lvl = 1
    cfg_lvl = 0
    try:
        cfg_lvl = int(OmegaConf.select(cfg, "hang_debug", default=0) or 0)
    except Exception:
        try:
            cfg_lvl = int(cfg.get("hang_debug", 0) or 0)
        except Exception:
            cfg_lvl = 0
    return max(0, min(2, max(env_lvl, cfg_lvl)))


def _hang_debug_first_batches(cfg) -> int:
    try:
        return int(OmegaConf.select(cfg, "hang_debug_first_batches", default=5) or 5)
    except Exception:
        try:
            return int(cfg.get("hang_debug_first_batches", 5) or 5)
        except Exception:
            return 5


def _hang_debug_cuda_sync(cfg) -> bool:
    if os.environ.get("HPP_HANG_DEBUG_SYNC", "").strip() in ("1", "true", "yes", "on"):
        return True
    try:
        return bool(OmegaConf.select(cfg, "hang_debug_cuda_sync", default=False))
    except Exception:
        return bool(cfg.get("hang_debug_cuda_sync", False))


def _hang_debug_payload(cfg, batch_idx: int = -1) -> Optional[dict]:
    """构造传给 ``model(..., hang_debug=...)`` 与 ``forward_hang_probe`` 的 payload；level 0 时返回 None。"""
    lvl = _hang_debug_level(cfg)
    if lvl == 0:
        return None
    return {
        "level": lvl,
        "batch_idx": batch_idx,
        "first_batches": _hang_debug_first_batches(cfg),
        "cuda_sync": _hang_debug_cuda_sync(cfg),
    }


def _hang_probe(
    cfg,
    stage: str,
    *,
    batch_idx: int = -1,
    extra: str = "",
) -> None:
    """在关键点打印 ``[HANG-DEBUG]`` 行：带 rank、单调时钟；可选 ``cuda synchronize`` 区分 GPU 是否已跑完。

    用法::

        export HPP_HANG_DEBUG=1
        # 更准确但很慢（每个探针等 GPU）:
        export HPP_HANG_DEBUG_SYNC=1
        # NCCL 集体通信日志（需重跑）:
        export NCCL_DEBUG=INFO

    卡死时看 **各 rank 最后一条** ``stage``：停在 ``dataloader_next`` 多为 DataLoader/worker；
    停在 ``forward_begin`` 之后看 ``fwd.*``（模型内细粒度阶段）；``backward`` / ``deepspeed_step`` 多为 ZeRO-3 或优化器。
    """
    forward_hang_probe(_hang_debug_payload(cfg, batch_idx), stage, extra)


def _register_hang_debug_handlers() -> None:
    """Linux 下 ``kill -USR1 <pid>`` 可 dump 全进程 Python 栈（卡死时另开终端对 rank0 的 python 发信号）。"""
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (ValueError, OSError):
            pass


def build_dataset(cfg):
    """加载数据，兼容两种配置结构：
    1) 旧结构: cfg.dataset.path
    2) 新结构: cfg.path
    """
    ds = cfg.dataset if "dataset" in cfg else cfg
    path = ds.path
    split = ds.get("split", "train")
    max_samples = ds.get("max_samples", None)

    if os.path.isdir(path):
        keep_in_memory = ds.get("keep_in_memory", False)
        loaded = load_from_disk(path, keep_in_memory=keep_in_memory)
        dataset = loaded[split]
    else:
        hub_kwargs = OmegaConf.to_container(ds, resolve=True)
        if not isinstance(hub_kwargs, dict):
            hub_kwargs = dict(hub_kwargs)
        hub_kwargs.pop("keep_in_memory", None)
        hub_kwargs.pop("max_samples", None)
        loaded = load_dataset(**hub_kwargs)
        if isinstance(loaded, DatasetDict):
            dataset = loaded[split]
        else:
            dataset = loaded

    # Rename columns if needed (some datasets use xyz/rgb/mask, PartNeXt uses coords/features/gt_masks)
    rename_map = {"xyz": "coords", "rgb": "features", "mask": "gt_masks"}
    current_columns = set(dataset.column_names)
    needed_renames = {k: v for k, v in rename_map.items() if k in current_columns}
    if needed_renames:
        dataset = dataset.rename_columns(needed_renames)

    # Select required columns
    required_cols = ["coords", "features", "gt_masks"]
    available_cols = [c for c in required_cols if c in dataset.column_names]
    dataset = dataset.select_columns(available_cols)

    # Limit dataset size for quick overfit / smoke experiments
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples > 0:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

    # 已删除 skip_indices_file 机制（用户要求训练过程中不跳过任何数据集）

    dataset.set_transform(Compose(cfg.transforms))

    if "repeats" in cfg:
        from torch.utils.data import Subset  # fmt: skip
        dataset = Subset(dataset, list(range(len(dataset))) * cfg.repeats)

    return dataset


def _is_hf_dataset_disk(path: str) -> bool:
    """是否为 HuggingFace `datasets.save_to_disk` 可加载目录（避免空目录/半目录误判）。"""
    p = Path(path)
    if not p.is_dir():
        return False
    if (p / "dataset_dict.json").is_file():
        return True
    # 单 split 直接保存的目录
    if (p / "dataset_info.json").is_file() and (p / "state.json").is_file():
        return True
    return False


def build_datasets(cfg):
    """自动检测并加载可用的数据集。"""
    available_datasets = []

    # 如果是混合数据集配置
    if "dataset_dict" in cfg:
        for key, dataset_cfg in cfg.dataset_dict.items():
            ds = dataset_cfg.dataset if "dataset" in dataset_cfg else dataset_cfg
            path = ds.get("path", "")
            if os.path.isdir(path) and _is_hf_dataset_disk(path):
                print(f"[Auto-detect] Found available dataset: {key} at {path}")
                available_datasets.append(build_dataset(dataset_cfg))
            elif os.path.isdir(path):
                print(
                    f"[Auto-detect] Skipping incomplete HF dataset at {key}={path} "
                    f"(need dataset_dict.json at root, or dataset_info.json+state.json for a single split)"
                )
            else:
                print(f"[Auto-detect] Skipping unavailable dataset: {key} at {path}")

        if not available_datasets:
            raise FileNotFoundError("No available datasets found! Please convert PartNeXt data first.")

        return ConcatDataset(available_datasets)
    else:
        # 单数据集配置
        path = cfg.get("path", "")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        if not _is_hf_dataset_disk(path):
            raise FileNotFoundError(
                f"Dataset path is not a valid HuggingFace `save_to_disk` folder: {path}\n"
                f"Expect root `dataset_dict.json`, or `dataset_info.json` + `state.json`."
            )
        return build_dataset(cfg)


class _NoOpPbar:
    """分布式下非主进程不打印 tqdm，避免多 rank 刷屏。"""

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


# NOTE: We separately instantiate each component for fine-grained control.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="hpp",
        help="Hydra 主配置名（如 hpp=Giant，hpp_memory_efficient=省显存 Large）",
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    args, unknown_args = parser.parse_known_args()

    # DeepSpeed / torchrun launcher injects --local_rank=N into sys.argv.
    # Hydra 1.x override grammar only accepts "key=value", NOT "--key=value".
    # parse_known_args() strips --local_rank=N so it ends up in unknown_args,
    # but Hydra compose still fails on the "--" prefix. Strip it here.
    # Also remove any launcher-injected args we don't want Hydra to see.
    filtered_overrides = []
    for arg in unknown_args:
        if arg.startswith("--local_rank="):
            continue  # skip deepspeed/torchrun injected rank
        if arg.startswith("--"):
            # Hydra override grammar: strip leading "--" (e.g. "--lr=1e-4" -> "lr=1e-4")
            stripped = arg[2:]
            if "=" in stripped:
                filtered_overrides.append(stripped)
            # else: bare --arg without =, skip (not a Hydra override)
        else:
            filtered_overrides.append(arg)

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=filtered_overrides)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    _register_hang_debug_handlers()
    if _hang_debug_level(cfg) > 0:
        print(
            "[HANG-DEBUG] enabled: HPP_HANG_DEBUG=1 → first hang_debug_first_batches batches; "
            "HPP_HANG_DEBUG=2 → every batch (verbose); HPP_HANG_DEBUG_SYNC=1 → cuda.synchronize "
            "at each probe (slow, pinpoints GPU vs CPU wait). "
            "Inside forward, look for stages prefixed with fwd. (e.g. fwd.after_pc_encoder, "
            "fwd.iter0.before_mask_decoder). "
            "Also try NCCL_DEBUG=INFO. Linux: kill -USR1 <train_pid> dumps Python stacks.",
            flush=True,
        )

    # Prepare (flat) hyperparameters for logging
    hparams = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "batch_size": cfg.train_dataloader.batch_size * cfg.gradient_accumulation_steps,
    }

    # Check cuda and cudnn settings
    torch.backends.cudnn.benchmark = True
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    # Enable TF32 for faster computation (if supported)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster computation")

    seed = cfg.get("seed", 42)

    # DeepSpeed/torchrun 已设置 LOCAL_RANK；分布式启动前仅 rank0 打印，避免 8 行重复日志。
    _is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    if _is_main_process:
        print_benchmark_metric_help()
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)
    _pi = getattr(model, "prompt_iters", None)
    if _is_main_process and _pi is not None and _pi < max(BENCHMARK_CLICKS):
        print(
            f"[metrics] model.prompt_iters={_pi} is less than {max(BENCHMARK_CLICKS)}: "
            f"IoU@{max(BENCHMARK_CLICKS)} will be NaN. Set model.prompt_iters>={max(BENCHMARK_CLICKS)} "
            "for full Point-SAM / PartNeXt-style tables."
        )

    # ---------------------------------------------------------------------------- #
    # Initialize with pre-trained weights if provided
    # ---------------------------------------------------------------------------- #
    if cfg.pretrained_ckpt_path:
        print("Loading pretrained weight from", cfg.pretrained_ckpt_path)
        pretrained = torch.load(cfg.pretrained_ckpt_path)
        # Hardcoded for Uni3D
        state_dict = {}
        for name in pretrained["module"].keys():
            if "point_encoder.encoder2trans" in name:
                # print(name)
                suffix = name[len("point_encoder.encoder2trans.") :]
                state_dict[f"patch_proj.{suffix}"] = pretrained["module"][name]
                # print(name, pretrained["module"][name].shape)
            if "point_encoder.pos_embed" in name:
                # print(name)
                suffix = name[len("point_encoder.pos_embed.") :]
                state_dict[f"pos_embed.{suffix}"] = pretrained["module"][name]
            if "point_encoder.visual" in name:
                # print(name)
                suffix = name[len("point_encoder.visual.") :]
                state_dict[f"transformer.{suffix}"] = pretrained["module"][name]
        missing_keys = model.pc_encoder.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    # ---------------------------------------------------------------------------- #
    # Setup dataloaders
    # ---------------------------------------------------------------------------- #
    train_dataset_cfg = hydra.utils.instantiate(cfg.train_dataset)
    train_dataset = build_datasets(train_dataset_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        **cfg.train_dataloader,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    if cfg.val_freq > 0:
        val_dataset_cfg = hydra.utils.instantiate(cfg.val_dataset)
        val_dataset = build_dataset(val_dataset_cfg)
        val_dataloader = DataLoader(
            val_dataset, **cfg.val_dataloader, worker_init_fn=worker_init_fn
        )

    # ---------------------------------------------------------------------------- #
    # Setup optimizer
    # ---------------------------------------------------------------------------- #
    params = []
    for name, module in model.named_children():
        # NOTE: Different learning rates can be set for different modules
        if name == "pc_encoder":
            params += [{"params": module.parameters(), "lr": cfg.lr}]
        else:
            params += [{"params": module.parameters(), "lr": cfg.lr}]

    # ---------------------------------------------------------------------------- #
    # Initialize accelerator
    # ---------------------------------------------------------------------------- #
    project_config = ProjectConfiguration(
        cfg.project_dir, automatic_checkpoint_naming=True, total_limit=1
    )

    # ============================================================ #
    # DeepSpeed ZeRO-3 模式 (优先于 FSDP/DDP)
    # 优势: 更精细的 CPU 内存管理，比 FSDP 的 CPU offload 更高效
    # ============================================================ #
    use_deepspeed = cfg.get("deepspeed", {}).get("enabled", False)
    use_fsdp = cfg.get("fsdp", {}).get("enabled", False)

    if use_deepspeed:
        import deepspeed
        import deepspeed.comm as dist

        ds_config = cfg.deepspeed
        stage = ds_config.get("stage", 3)
        offload_optimizer = ds_config.get("offload_optimizer", False)
        offload_param = ds_config.get("offload_param", False)
        offload_optimizer_device = ds_config.get("offload_optimizer_device", "cpu")
        offload_param_device = ds_config.get("offload_param_device", "cpu")
        overlap_comm = ds_config.get("overlap_comm", True)
        contiguous_gradients = ds_config.get("contiguous_gradients", True)
        reduce_bucket_size = ds_config.get("reduce_bucket_size", 50000000)
        stage3_max_live_parameters = ds_config.get("stage3_max_live_parameters", 1000000)
        stage3_max_reuse_distance = ds_config.get("stage3_max_reuse_distance", 1000000)
        stage3_param_persistence_threshold = ds_config.get("stage3_param_persistence_threshold", 100000)
        stage3_prefetch_bucket_size = ds_config.get("stage3_prefetch_bucket_size", 50000000)

        # 构建 DeepSpeed ZeRO 优化配置
        # 勿加入 round_robin_steps：DeepSpeed 0.18+ 的 zero_optimization Pydantic 模型
        # 已禁止未声明字段，会触发 Extra forbidden（与 ZeRO-3 烟测脚本一致）。
        zero_optimization = {
            "stage": stage,
            "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
            "stage3_max_live_parameters": stage3_max_live_parameters,
            "stage3_max_reuse_distance": stage3_max_reuse_distance,
            "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
            "reduce_bucket_size": reduce_bucket_size,
            "overlap_comm": overlap_comm,
            "contiguous_gradients": contiguous_gradients,
        }

        # 添加 CPU offload 配置
        if offload_optimizer:
            zero_optimization["offload_optimizer"] = {
                "device": offload_optimizer_device,
                "pin_memory": True,
            }

        if offload_param:
            zero_optimization["offload_param"] = {
                "device": offload_param_device,
                "pin_memory": True,
            }

        # 混合精度配置
        _mp = str(cfg.get("mixed_precision", "no") or "no").lower()
        if _mp in ("bf16", "bfloat16"):
            fp16_config = {"enabled": False}
            bf16_config = {"enabled": True}
            ds_mixed_precision = "bf16"
        elif _mp in ("fp16", "float16"):
            fp16_config = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
            bf16_config = {"enabled": False}
            ds_mixed_precision = "fp16"
        else:
            fp16_config = {"enabled": False}
            bf16_config = {"enabled": False}
            ds_mixed_precision = "no"

        # DeepSpeed 要求: train_batch_size == micro_batch * grad_acc * world_size
        # （见 deepspeed/runtime/config.py _batch_assertion）
        # deepspeed launcher 在子进程里已设置 WORLD_SIZE；单卡本地跑时默认为 1。
        _ds_micro = int(cfg.train_dataloader.get("batch_size", 1))
        _ds_grad_acc = int(cfg.gradient_accumulation_steps)
        _ds_world = int(os.environ.get("WORLD_SIZE", "1"))
        _ds_train_bs = _ds_micro * _ds_grad_acc * _ds_world

        # 完整 DeepSpeed 配置
        # nccl_timeout：PyTorch ProcessGroupNCCL watchdog 超时，单位为毫秒（ms）。
        # 必须在 deepspeed_config 中设置，环境变量 NCCL_TIMEOUT 对 PyTorch 2.x ProcessGroupNCCL
        # 不生效！调试时 5 分钟（300000ms），确认稳定后可改回 1800000ms（30 分钟）。
        deepspeed_config = {
            "train_batch_size": _ds_train_bs,
            "train_micro_batch_size_per_gpu": _ds_micro,
            "gradient_accumulation_steps": _ds_grad_acc,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            "zero_optimization": zero_optimization,
            "fp16": fp16_config,
            "bf16": bf16_config,
            "gradient_clipping": cfg.get("max_grad_value", 1.0),
            "zero_allow_untested_optimizer": True,
            "nccl_timeout": 600000,  # 10 分钟（毫秒）；确认稳定后可改回 1800000（30 分钟）
            "scheduler": {
                "enabled": True,
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": cfg.lr,
                    "warmup_num_steps": 100,
                    "total_num_steps": cfg.max_steps,
                },
            },
        }

        # 保存 DeepSpeed 配置到临时文件
        ds_config_path = os.path.join(cfg.project_dir, "deepspeed_config.json")
        os.makedirs(cfg.project_dir, exist_ok=True)
        with open(ds_config_path, "w") as f:
            json.dump(deepspeed_config, f, indent=2)

        print(f"[DeepSpeed] Configuration saved to {ds_config_path}")
        print(
            f"[DeepSpeed] Batch: train_batch_size={_ds_train_bs} "
            f"(micro_per_gpu={_ds_micro} * grad_acc={_ds_grad_acc} * world_size={_ds_world})"
        )
        print(f"[DeepSpeed] Stage: ZeRO-{stage}")
        print(f"[DeepSpeed] NCCL Timeout: {deepspeed_config.get('nccl_timeout', 'N/A')} ms")
        print(f"[DeepSpeed] Offload Optimizer: {offload_optimizer} ({offload_optimizer_device})")
        print(f"[DeepSpeed] Offload Param: {offload_param} ({offload_param_device})")
        print(f"[DeepSpeed] Overlap Comm: {overlap_comm}")
        print(f"[DeepSpeed] Mixed Precision: {ds_mixed_precision}")
        if int(stage) >= 3 and overlap_comm:
            print(
                "[DeepSpeed][WARN] ZeRO-3 + overlap_comm=True 易触发 NCCL all_gather 与计算重叠死锁；"
                "建议配置 deepspeed.overlap_comm=false。"
            )

        # ZeRO offload 优化器状态时须用 DeepSpeedCPUAdam，否则 0.18+ 抛 ZeRORuntimeException
        # （torch.optim.AdamW 与 CPU offload 不兼容）。
        # 注意：DeepSpeedCPUAdam 需要 GCC 9+，若节点编译器版本不足 JIT 编译会失败。
        # 如果 DeepSpeedCPUAdam 不可用（GCC 版本过低），fallback 到标准 torch.optim.AdamW。
        if offload_optimizer:
            try:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                optimizer = DeepSpeedCPUAdam(
                    params,
                    lr=cfg.lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=cfg.weight_decay,
                )
            except (ImportError, OSError, RuntimeError) as _e:
                print(f"[DeepSpeed] DeepSpeedCPUAdam unavailable ({_e}), falling back to torch.optim.AdamW")
                optimizer = torch.optim.AdamW(
                    params,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    fused=False,
                )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                fused=False,
            )

        # deepspeed CLI 启动时子进程内通常尚未 init_process_group，须由 DeepSpeed 初始化
        # （与 scripts/deepspeed_zero3_smoke.py 中 dist_init_required=True 一致）。
        # 注意：deepspeed bf16.enabled=true 会在 initialize() 时将模型参数保持在 bf16，
        # 不需要手动 model.to(bf16)；输入数据 cast 在训练循环里做（见 forward 处）。

        # ZeRO-3：MaskEncoder（scatter_reduce / Voronoi 路径）在子层逐个 hook 时，易出现
        # 部分 rank 已离开该模块、其余 rank 仍卡在内部 → NCCL 与下一子模块 hook 交错死锁。
        # 将整段 MaskEncoder 标为 leaf，参数在进/出模块时一次性 gather/release（见 DeepSpeed #4966）。
        _set_z3_leaf = None
        try:
            from deepspeed.utils import set_z3_leaf_modules as _set_z3_leaf
        except ImportError:
            try:
                from deepspeed.utils.set_z3_leaf_modules import (
                    set_z3_leaf_modules as _set_z3_leaf,
                )
            except ImportError:
                try:
                    from deepspeed.zero import set_z3_leaf_modules as _set_z3_leaf
                except ImportError:
                    pass
        if _set_z3_leaf is not None:
            from hpp_sam.model.mask_hra_leaf import MaskEncoderHRALeaf

            try:
                _set_z3_leaf(model, [MaskEncoderHRALeaf])
            except Exception as _z3e:
                print(f"[DeepSpeed][WARN] set_z3_leaf_modules(MaskEncoderHRALeaf) failed: {_z3e}")
            else:
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(
                        "[DeepSpeed] ZeRO-3 leaf: MaskEncoderHRALeaf "
                        "(mask_encoder+HRA single gather; fixes M≠ rank skew)"
                    )

        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=deepspeed_config,
            dist_init_required=True,
        )

        # DeepSpeed 模式下 accelerator 仅用于日志和部分工具
        accelerator = Accelerator(
            project_config=project_config,
            log_with=cfg.log_with,
            mixed_precision=ds_mixed_precision,
        )

        # 准备数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            **cfg.train_dataloader,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
        )
        train_dataloader = accelerator.prepare(train_dataloader)

        if cfg.val_freq > 0:
            val_dataloader = DataLoader(
                val_dataset, **cfg.val_dataloader, worker_init_fn=worker_init_fn
            )
            val_dataloader = accelerator.prepare(val_dataloader)

        # 创建 DeepSpeed 兼容的 scheduler (由 DeepSpeed 管理，这里创建占位符)
        class DeepSpeedSchedulerPlaceholder:
            def step(self):
                pass
            def get_last_lr(self):
                return [cfg.lr]
        scheduler = DeepSpeedSchedulerPlaceholder()

        accelerator.print(
            f"[DeepSpeed] ZeRO-{stage} initialized with CPU offload "
            f"(optimizer={offload_optimizer}, param={offload_param})"
        )

    # ============================================================ #
    # FSDP 显存均摊模式 (DeepSpeed 不可用时的备选)
    # ============================================================ #
    elif use_fsdp:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        fsdp_config = cfg.fsdp
        offload_params = fsdp_config.get("cpu_offload_optimizer", False)
        backward_prefetch = fsdp_config.get("backward_prefetch", "backward_pre")
        forward_prefetch = fsdp_config.get("forward_prefetch", False)

        # Wrapping 策略：顶层子模块按 **对象 identity** 整包 wrap（各 rank 一致）。
        #
        # 旧实现用 module._module_name / "pc_encoder" 字符串匹配，但 nn.Module 默认没有 _module_name，
        # hasattr 为 False 时退化为 type(module).__name__（如 PointCloudEncoder），永远匹配不到逻辑名，
        # 导致 pc_encoder / mask_decoder 等从未按「整子模块」成为 FSDP 根，分片边界与 gradient checkpoint、
        # bf16 reduce 等组合时易在多卡下触发非法显存访问（NCCL watchdog 报 illegal memory access）。
        #
        # 上游 Point-SAM（https://github.com/zyc00/Point-SAM）训练仅用 DDP + accelerator.no_sync，无 FSDP；
        # 若仍不稳定，可在配置中设 fsdp.enabled: false 与上游对齐。
        _fsdp_wrap_roots = frozenset(
            m
            for m in (
                getattr(model, "pc_encoder", None),
                getattr(model, "point_encoder", None),
                getattr(model, "mask_encoder", None),
                getattr(model, "mask_decoder", None),
                getattr(model, "hyper_prompt_branch", None),
            )
            if m is not None
        )

        def module_name_auto_wrap_policy(module, recurse, nonwrapped_numel, **kwargs):
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy as _tf_wrap,
            )
            from hpp_sam.model.transformer import TwoWayAttentionBlock
            layer_types = {TwoWayAttentionBlock}
            try:
                from timm.models.vision_transformer import Block as TimmViTBlock
                layer_types.add(TimmViTBlock)
            except ImportError:
                pass
            if recurse:
                return False
            if module in _fsdp_wrap_roots:
                return True
            return _tf_wrap(module, recurse, nonwrapped_numel=nonwrapped_numel, transformer_layer_cls=layer_types)

        auto_wrap_policy = partial(module_name_auto_wrap_policy, nonwrapped_numel=None)

        # backward_prefetch 与 forward_prefetch 保持与原配置一致
        from torch.distributed.fsdp import BackwardPrefetch
        backward_prefetch_map = {
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST,
        }
        bw_prefetch = backward_prefetch_map.get(backward_prefetch, BackwardPrefetch.BACKWARD_PRE)

        # 与 cfg.mixed_precision 及 accelerate launch --mixed_precision 对齐（FSDP 曾硬编码 bf16）
        _mp = str(cfg.get("mixed_precision", "no") or "no").lower()
        if _mp in ("bf16", "bfloat16"):
            _fsdp_mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
            _accel_mp = "bf16"
        elif _mp in ("fp16", "float16"):
            _fsdp_mp = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float16,
            )
            _accel_mp = "fp16"
        else:
            _fsdp_mp = None
            _accel_mp = "no"

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision_policy=_fsdp_mp,
            cpu_offload=offload_params,
            backward_prefetch=bw_prefetch,
            forward_prefetch=forward_prefetch,
        )
        accelerator = Accelerator(
            project_config=project_config,
            log_with=cfg.log_with,
            mixed_precision=_accel_mp,
            fsdp_plugin=fsdp_plugin,
        )

        if offload_params:
            model = model.cpu()
        else:
            model = model.to(accelerator.device)

        use_fused_optimizer = False
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=False)
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        if cfg.val_freq > 0:
            val_dataloader = accelerator.prepare(val_dataloader)
        accelerator.print(
            f"[FSDP] FULL_SHARD (Accelerate), auto_wrap=module_name, cpu_offload={offload_params}, "
            f"backward_prefetch={backward_prefetch}, forward_prefetch={forward_prefetch}, "
            f"mixed_precision={_accel_mp}, fused=False"
        )

    else:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
            log_with=cfg.log_with,
            mixed_precision=cfg.get("mixed_precision", "no"),
        )
        offload_params = False
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=False)
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        if cfg.val_freq > 0:
            val_dataloader = accelerator.prepare(val_dataloader)
        accelerator.print("[optimizer] fused=False (DDP mode)")

    # criterion = Criterion()
    criterion = hydra.utils.instantiate(cfg.loss)

    # Initialize learning diagnostics and adaptive regularization
    diagnostics = None
    adaptive_reg = None
    if HAS_LEARNING_DIAGNOSTICS:
        try:
            diagnostics = LearningDiagnostics(
                patience=cfg.get("diagnostics_patience", 10),
                overfit_threshold=cfg.get("overfit_threshold", 0.3),
                collapse_threshold=cfg.get("collapse_threshold", 0.5),
                entropy_threshold=cfg.get("entropy_threshold", 0.3),
                min_epochs=cfg.get("min_epochs_before_intervention", 50),
            )
            adaptive_reg = create_adaptive_regularizer(model)
            accelerator.print("[Learning Diagnostics] Initialized successfully")
        except Exception as e:
            accelerator.print(f"[Learning Diagnostics] Failed to initialize: {e}")
            diagnostics = None
            adaptive_reg = None

    accelerator.print(OmegaConf.to_yaml(cfg))

    if cfg.log_with:
        accelerator.init_trackers(
            project_name=cfg.get("project_name", "pointcloud-sam"),
            config=hparams,
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    if cfg.log_with == "wandb":
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            file_path = os.path.join(wandb_tracker.run.dir, "full_config.yaml")
            with open(file_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            wandb_tracker.run.save(file_path)
        except:
            pass

    # Define validation function
    @torch.no_grad()
    def validate():
        model.eval()
        epoch_ious = defaultdict(list)

        if accelerator.is_main_process:
            if cfg.log_with == "wandb":
                pbar = tqdm(total=len(val_dataloader), miniters=10, maxinterval=60)
            else:
                pbar = tqdm(total=len(val_dataloader))
        else:
            pbar = _NoOpPbar()

        for data in val_dataloader:
            outputs = model(**data, is_eval=True)
            gt_masks = data["gt_masks"].flatten(0, 1)

            for i_iter in range(len(outputs)):
                if i_iter == 0:
                    all_masks = outputs[0]["masks"]  # [B*M, C, N]
                    all_ious = compute_iou(
                        all_masks, gt_masks.unsqueeze(1).expand_as(all_masks)
                    )
                    best_iou = all_ious.max(dim=1).values
                    epoch_ious["best"].extend(best_iou.tolist())
                iou = compute_iou(outputs[i_iter]["prompt_masks"], gt_masks)
                epoch_ious[i_iter].extend(iou.tolist())

            # Running means for progress bar
            metrics_run = {}
            for k, vals in epoch_ious.items():
                if len(vals) == 0:
                    continue
                if k == "best":
                    metrics_run["IoU_best"] = float(np.mean(vals))
                else:
                    metrics_run[f"iou({k})"] = float(np.mean(vals))
            metrics_run.update(compute_benchmark_from_epoch_ious(epoch_ious))
            pbar.set_postfix(metrics_run)
            pbar.update(1)

        pbar.close()

        metrics = {}
        for k, vals in epoch_ious.items():
            if len(vals) == 0:
                continue
            if k == "best":
                metrics["IoU_best_multimask_iter0"] = float(np.mean(vals))
            else:
                metrics[f"iou({k})"] = float(np.mean(vals))
        metrics.update(compute_benchmark_from_epoch_ious(epoch_ious))
        # Surrogate scalar loss for learning_diagnostics (lower is better)
        iou_vals = [v for k, v in metrics.items() if isinstance(k, str) and k.startswith("iou(")]
        if iou_vals:
            metrics["iou"] = float(np.mean(iou_vals))
            metrics["loss"] = 1.0 - metrics["iou"]
        return metrics

    # ---------------------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------------------- #
    # step：仅统计「成功完成 backward」的微步，用于梯度累积边界（与 DataLoader 消耗批次数解耦）
    step = 0
    global_step = 0  # 成功训练步的日志步（与 optimizer.step 次数含义见下方 else 分支）
    start_epoch = 0
    # 断点续传：仅在 checkpoint 记录的 epoch 内跳过 [0, next_batch_idx)（勿用全局 step 与 batch_idx 比较）
    resume_skip_epoch = None
    resume_next_batch_idx = 0
    loop_counter = 0  # 每消耗一个 DataLoader batch +1（含 skip），用于 batch_save_freq

    # Restore state
    # resume_state 必须放在 project_dir 根目录，不能放在 checkpoints/：
    # accelerate.save_state 会 os.listdir(checkpoints) 并按路径末尾数字排序；
    # resume_state.pt 无数字，会触发 IndexError（见 accelerator.py save_state）。
    project_dir_path = Path(accelerator.project_dir)
    ckpt_dir = project_dir_path / "checkpoints"
    resume_state_file = project_dir_path / "resume_state.pt"
    resume_state_file_legacy = ckpt_dir / "resume_state.pt"

    if accelerator.is_main_process and resume_state_file_legacy.exists():
        if not resume_state_file.exists():
            shutil.move(str(resume_state_file_legacy), str(resume_state_file))
            print(f"[Checkpoint] 已从 checkpoints/ 迁移 resume_state.pt -> {resume_state_file}")
        else:
            try:
                resume_state_file_legacy.unlink()
            except OSError:
                pass
    accelerator.wait_for_everyone()

    _resume_to_load = resume_state_file if resume_state_file.exists() else None
    if _resume_to_load is None and resume_state_file_legacy.exists():
        _resume_to_load = resume_state_file_legacy

    def _has_accelerate_checkpoint() -> bool:
        return ckpt_dir.exists() and bool(list(ckpt_dir.glob("checkpoint_*")))

    def _has_deepspeed_checkpoint() -> bool:
        p = ckpt_dir / "deepspeed"
        if not p.is_dir():
            return False
        try:
            return any(p.iterdir())
        except OSError:
            return False

    # -------------------------------------------------------------------------
    # 断点恢复顺序（关键）：
    # 1) 若存在 Accelerate 的 checkpoint_*，必须先 load_state，恢复权重/优化器/调度器。
    # 2) 若再存在 resume_state.pt，用其覆盖 epoch、next_batch_idx、global_step 等游标。
    #
    # 旧逻辑是「有 resume_state 就只读元数据、绝不 load_state」——resume_state 在训练中
    # 会周期性写入，导致重启后几乎总有该文件，从而**永远跳过权重恢复**，出现「进度在走、
    # 指标像假学习/冻结」或训练从随机权重接着记 wandb 步数等严重问题。
    # -------------------------------------------------------------------------
    # 断点恢复: DeepSpeed 使用自己的 checkpoint 格式
    # -------------------------------------------------------------------------
    if use_deepspeed:
        # DeepSpeed 使用 deepspeed.save_checkpoint / load_checkpoint
        ds_checkpoint_dir = ckpt_dir / "deepspeed"
        if ds_checkpoint_dir.exists():
            load_path, _ = model.load_checkpoint(str(ds_checkpoint_dir))
            if accelerator.is_main_process:
                print(f"[DeepSpeed] Loaded checkpoint from {ds_checkpoint_dir}")
        else:
            if accelerator.is_main_process:
                print("[DeepSpeed] No checkpoint found, starting from scratch")
        accelerator.wait_for_everyone()

    elif _has_accelerate_checkpoint():
        # 所有 rank 均执行 load_state（Accelerate 对各 rank 独立验证，防止 rank-specific 状态不一致）
        # FSDP 下需 wait_for_everyone 同步；非 FSDP DDP 无需额外同步
        if use_fsdp:
            accelerator.load_state(input_dir=str(ckpt_dir))
            accelerator.wait_for_everyone()
        else:
            accelerator.load_state(input_dir=str(ckpt_dir))
        if accelerator.is_main_process:
            print(f"[Checkpoint] Loaded Accelerate state (weights/optimizer/scheduler) from {ckpt_dir}")

    # 是否与 resume_state 配套的权重目录存在且已尝试加载（无权重则不得按游标跳过 batch，否则「只跑 tqdm 不训练」）
    _weights_restored = (
        _has_deepspeed_checkpoint() if use_deepspeed else _has_accelerate_checkpoint()
    )

    if _resume_to_load is not None:
        # 加载断点续传状态（游标与计步器；权重已在上方 load_state）
        if accelerator.is_main_process:
            print(f"Loading resume cursor from {_resume_to_load}")
        resume_state = torch.load(_resume_to_load, map_location="cpu")
        start_epoch = resume_state.get("epoch", 0)
        global_step = resume_state.get("global_step", 0)
        step = resume_state.get("micro_step", resume_state.get("batch", 0))
        loop_counter = resume_state.get("loop_counter", resume_state.get("batch", 0))
        fmt = resume_state.get("format_version", 1)
        if fmt >= 2 and "next_batch_idx" in resume_state:
            resume_skip_epoch = resume_state["epoch"]
            resume_next_batch_idx = int(resume_state["next_batch_idx"])
        else:
            # 旧版把全局 step 误存为 batch，与 batch_idx 比较会在 step>=len(dataloader) 时跳过整 epoch
            resume_skip_epoch = resume_state.get("epoch", start_epoch)
            resume_next_batch_idx = int(resume_state.get("batch", 0))

        if not _weights_restored:
            # 仅有周期性写入的 resume_state.pt、无 DeepSpeed/Accelerate 权重时，沿用游标会空跑若干 batch
            if accelerator.is_main_process:
                print(
                    "[Checkpoint] 仅有 resume_state.pt，未找到已保存的权重 checkpoint（DeepSpeed: "
                    f"{ckpt_dir / 'deepspeed'}；Accelerate: {ckpt_dir}/checkpoint_*）。"
                    "已忽略游标中的 epoch/batch 跳过与计步，从 epoch 0、batch 0 重新训练。"
                )
            start_epoch = 0
            global_step = 0
            step = 0
            loop_counter = 0
            resume_skip_epoch = None
            resume_next_batch_idx = 0
        elif accelerator.is_main_process:
            print(
                f"Resuming from epoch {start_epoch}, next_batch_idx={resume_next_batch_idx} "
                f"(skip_epoch={resume_skip_epoch}), micro_step={step}, global_step={global_step}"
            )
        del resume_state
        torch.cuda.empty_cache()
    elif _has_accelerate_checkpoint():
        # 无 resume_state：仅根据最新 checkpoint 目录名 + 调度器推断起点（旧行为）
        global_step = scheduler.scheduler.last_epoch // accelerator.state.num_processes
        get_epoch_fn = lambda x: int(x.name.split("_")[-1])
        last_ckpt_dir = sorted(ckpt_dir.glob("checkpoint_*"), key=get_epoch_fn)[-1]
        start_epoch = get_epoch_fn(last_ckpt_dir) + 1
        if accelerator.is_main_process:
            print(
                f"[Checkpoint] No resume_state.pt; inferred start_epoch={start_epoch}, "
                f"global_step≈{global_step} from scheduler / {last_ckpt_dir.name}"
            )

    last_train_metrics_snap: dict = {}
    # 训练条上 loss 的指数滑动平均，便于看出是否在下降（单 batch 的 acc/iou 波动大且 tqdm 默认精度低）
    train_loss_ema = None
    train_loss_ema_beta = 0.99
    # ZeRO-3 下不允许 skip，故移除 consecutive_cuda_skips、known_bad_batches 等跟踪变量。
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        val_metrics_this_epoch = None
        _hang_probe(cfg, f"epoch_start epoch={epoch}")

        if accelerator.is_main_process:
            if cfg.log_with == "wandb":
                # wandb 会抓 stdout，降低 tqdm 刷新频率（不影响 postfix 里的数值含义）
                pbar = tqdm(total=len(train_dataloader), miniters=10, maxinterval=60)
            else:
                pbar = tqdm(total=len(train_dataloader))
        else:
            pbar = _NoOpPbar()

        train_len = len(train_dataloader)
        train_iter = iter(train_dataloader)
        # 每个 epoch 最后一次成功 forward 的输出（供 epoch 末 diagnostics，避免 outputs 未定义）
        last_train_outputs = None

        # 断点恢复与本 epoch 对齐：
        # - next_batch_idx > train_len：旧版把全局 step 写进 batch，尝试取模
        # - next_batch_idx >= train_len：本 epoch 在保存点已结束（含 == train_len 的 off-by-one），
        #   若仍按 batch_idx < next_batch_idx 跳过则会永远 skip 且无法清除 resume_skip_epoch
        if resume_skip_epoch is not None and epoch == resume_skip_epoch:
            if resume_next_batch_idx > train_len:
                old_nb = resume_next_batch_idx
                resume_next_batch_idx = resume_next_batch_idx % train_len
                if accelerator.is_main_process:
                    print(
                        f"[WARN] resume 中 next_batch_idx={old_nb} > train_len={train_len}，"
                        f"疑似旧版 resume_state.pt；已取模为 {resume_next_batch_idx}。"
                        f"若仍异常请删除 {resume_state_file} 后重开训练。"
                    )
            if resume_next_batch_idx >= train_len:
                if accelerator.is_main_process:
                    print(
                        f"[WARN] resume 中 next_batch_idx={resume_next_batch_idx} >= train_len={train_len}，"
                        f"表示保存时本 epoch 已全部消费；跳过本 epoch，进入下一 epoch。"
                    )
                resume_skip_epoch = None
                resume_next_batch_idx = 0
                pbar.close()
                continue

        for batch_idx in range(train_len):
            try:
                _hang_probe(cfg, "dataloader_next_begin", batch_idx=batch_idx)
                data = next(train_iter)
                _hang_probe(cfg, "dataloader_next_end", batch_idx=batch_idx)
                # DeepSpeed ZeRO-3：禁止部分 rank 先取下一 batch 并开始 forward，而其他 rank
                # 仍卡在上一次 forward 的集体通信里（日志里常见 rank1 已到 forward_begin，
                # 其余 rank 仍在 fwd.iter*）。取数后强制对齐。
                if use_deepspeed:
                    accelerator.wait_for_everyone()
            except StopIteration:
                break
            except RuntimeError as e:
                # ZeRO-3 下不允许 skip：直接抛异常，交由上层处理。
                raise

            # 断点续传：仅在 checkpoint 记录的 epoch 内跳过 [0, next_batch_idx)
            if resume_skip_epoch is not None and epoch == resume_skip_epoch:
                if batch_idx < resume_next_batch_idx:
                    # 跳过阶段不跑 forward，tqdm 默认会保留上一 postfix，易被误认为「指标冻住」
                    if accelerator.is_main_process:
                        pbar.set_postfix_str(
                            f"resume_skip {batch_idx}/{resume_next_batch_idx} (no train yet)",
                            refresh=False,
                        )
                    pbar.update(1)
                    loop_counter += 1
                    if use_deepspeed:
                        accelerator.wait_for_everyone()
                    continue
                resume_skip_epoch = None

            flag = (step + 1) % cfg.gradient_accumulation_steps == 0

            # ZeRO-3：不再有任何 skip 路径，每个 batch 必须完整执行 forward+backward+step，
            # 否则会导致各 rank backward/step 调用次数不一致，NCCL 集体通信死锁。

            # FSDP / DeepSpeed 禁用 no_sync。
            ctx = nullcontext if (flag or use_fsdp or use_deepspeed) else accelerator.no_sync

            # DeepSpeed bf16：显式 cast 输入数据为 bf16。
            if use_deepspeed and ds_mixed_precision == "bf16":
                data = {k: v.to(torch.bfloat16) if torch.is_tensor(v) and v.dtype in (torch.float32, torch.half) else v for k, v in data.items()}

            with ctx(model):
                _hang_probe(cfg, "forward_begin", batch_idx=batch_idx)
                outputs = model(**data, hang_debug=_hang_debug_payload(cfg, batch_idx))
                _hang_probe(cfg, "forward_end", batch_idx=batch_idx)
                gt_masks = data["gt_masks"].flatten(0, 1)  # [B*M, N]
                loss, aux = criterion(outputs, gt_masks)

                # 强制检查 loss 有限性，不 skip。
                if not torch.isfinite(loss).all():
                    loss_value = loss.detach().cpu()
                    if accelerator.is_main_process:
                        print(
                            f"[ERROR] 非有限 loss，batch_idx={batch_idx} micro_step={step}: "
                            f"loss={loss_value}。不允许 skip，ZeRO-3 需要所有 rank 同步。"
                        )
                    raise RuntimeError(f"Nonfinite loss: {loss_value}. ZeRO-3 requires all ranks to be in sync.")

                # DeepSpeed 训练路径
                if use_deepspeed:
                    _loss_for_bw = (
                        loss.to(torch.bfloat16)
                        if loss.dtype == torch.float32
                        else loss
                    )
                    _hang_probe(
                        cfg,
                        "deepspeed_backward_begin",
                        batch_idx=batch_idx,
                        extra=f"grad_acc_boundary={flag}",
                    )
                    model.backward(_loss_for_bw / cfg.gradient_accumulation_steps)
                    _hang_probe(cfg, "deepspeed_backward_end", batch_idx=batch_idx)
                    if flag:
                        _hang_probe(cfg, "deepspeed_step_begin", batch_idx=batch_idx)
                        model.step()
                        _hang_probe(cfg, "deepspeed_step_end", batch_idx=batch_idx)
                # FSDP / DDP 训练路径
                else:
                    _hang_probe(
                        cfg,
                        "accelerator_backward_begin",
                        batch_idx=batch_idx,
                        extra=f"grad_acc_boundary={flag}",
                    )
                    accelerator.backward(loss / cfg.gradient_accumulation_steps)
                    _hang_probe(cfg, "accelerator_backward_end", batch_idx=batch_idx)
                    if flag:
                        _hang_probe(cfg, "optimizer_step_begin", batch_idx=batch_idx)
                        if cfg.max_grad_value:
                            accelerator.clip_grad_norm_(
                                model.parameters(), max_norm=float(cfg.max_grad_value)
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        _hang_probe(cfg, "optimizer_step_end", batch_idx=batch_idx)
                        if cuda_sync_after_step and torch.cuda.is_available():
                            torch.cuda.synchronize()

                # Compute metrics（每个 batch 必定成功执行到这里，因为 skip 逻辑已删除）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                with torch.no_grad():
                    metrics = dict(loss=loss.item())
                    for i_iter in range(len(aux)):
                        pred_masks = aux[i_iter]["best_masks"] > 0
                        is_correct = pred_masks == gt_masks
                        acc = is_correct.float().mean()
                        fg_acc = is_correct[gt_masks == 1].float().mean()
                        bg_acc = is_correct[gt_masks == 0].float().mean()
                        metrics[f"acc({i_iter})"] = acc.item()
                        metrics[f"fg_acc({i_iter})"] = fg_acc.item()
                        metrics[f"bg_acc({i_iter})"] = bg_acc.item()

                        iou = aux[i_iter]["iou"].mean()
                        metrics[f"iou({i_iter})"] = iou.item()

                        for k, v in aux[i_iter].items():
                            if k.startswith("loss"):
                                metrics[f"{k}({i_iter})"] = v.item()

                    if aux[0].get("reg_loss") is not None:
                        metrics["loss/reg_loss"] = aux[0]["reg_loss"].item()
                    if aux[0].get("radius_reg") is not None:
                        metrics["loss/radius_reg"] = aux[0]["radius_reg"].item()
                    if aux[0].get("radius_div") is not None:
                        metrics["loss/radius_div"] = aux[0]["radius_div"].item()

                    bench_train = compute_benchmark_from_aux(aux)
                    metrics.update(bench_train)

                    metrics["debug/coords_abs_sum"] = float(
                        data["coords"].float().abs().sum().cpu()
                    )
                    metrics["debug/pred_fg_ratio_iter0"] = float(
                        (aux[0]["best_masks"] > 0).float().mean().cpu()
                    )

                    if hasattr(model, "hyper_prompt_branch"):
                        try:
                            metrics["hra/curvature"] = (
                                model.hyper_prompt_branch.get_curvature()
                            )
                        except Exception:
                            pass

                    for i_iter, out in enumerate(outputs):
                        if out.get("radius") is not None:
                            r = out["radius"].detach().float()
                            metrics[f"hra/radius_mean_iter{i_iter}"] = r.mean().item()
                            metrics[f"hra/radius_std_iter{i_iter}"] = r.std(unbiased=False).item()
                            metrics[f"hra/radius_min_iter{i_iter}"] = r.min().item()
                            metrics[f"hra/radius_max_iter{i_iter}"] = r.max().item()
                        if out.get("temperature") is not None:
                            t = out["temperature"].detach().float()
                            metrics[f"hra/temp_mean_iter{i_iter}"] = t.mean().item()
                            metrics[f"hra/temp_std_iter{i_iter}"] = t.std(unbiased=False).item()
                            metrics[f"hra/temp_min_iter{i_iter}"] = t.min().item()
                            metrics[f"hra/temp_max_iter{i_iter}"] = t.max().item()
                        if out.get("prompt_masks") is not None:
                            pm = out["prompt_masks"].detach().float()
                            metrics[f"mask/fg_ratio_iter{i_iter}"] = (
                                (pm > 0).float().mean().item()
                            )
                        if i_iter < len(aux) and aux[i_iter].get("iou") is not None:
                            si = aux[i_iter]["iou"].detach().float()
                            metrics[f"mask/iou_min_iter{i_iter}"] = si.min().item()
                            metrics[f"mask/iou_std_iter{i_iter}"] = si.std(
                                unbiased=False
                            ).item()
                        if out.get("prompt_coords") is not None:
                            metrics[f"prompt/num_prompts_iter{i_iter}"] = (
                                out["prompt_coords"].shape[1]
                            )

                    last_train_metrics_snap = {}
                    for _k, _v in metrics.items():
                        try:
                            last_train_metrics_snap[_k] = float(_v)
                        except (TypeError, ValueError):
                            pass

                # tqdm：当前 batch 的瞬时 acc/iou（非 epoch 均值）。batch_size=1 时离散化明显，
                # 且默认显示位数少，看起来会「不变」；看 loss / loss_ema / W&B 曲线更可靠。
                loss_now = float(metrics["loss"])
                if train_loss_ema is None:
                    train_loss_ema = loss_now
                else:
                    train_loss_ema = (
                        train_loss_ema_beta * train_loss_ema
                        + (1.0 - train_loss_ema_beta) * loss_now
                    )
                sub_metrics = {
                    k: round(float(v), 4)
                    for k, v in metrics.items()
                    if k.startswith("acc") or k.startswith("iou")
                }
                sub_metrics["bench"] = format_benchmark_for_postfix(bench_train)
                sub_metrics["loss"] = round(loss_now, 5)
                sub_metrics["loss_ema"] = round(train_loss_ema, 5)
                _lrs = scheduler.get_last_lr()
                sub_metrics["lr"] = round(float(_lrs[0]) if _lrs else 0.0, 8)
                sub_metrics["c_abs"] = round(metrics["debug/coords_abs_sum"], 1)
                sub_metrics["p_fg"] = round(metrics["debug/pred_fg_ratio_iter0"], 4)
                pbar.set_postfix(sub_metrics)

                # WandB: radius / temperature histograms (same cadence as 3D vis)
                if (
                    cfg.log_with == "wandb"
                    and (global_step + 1) % (cfg.get("vis_freq", 1000)) == 0
                ):
                    for i_iter, out in enumerate(outputs):
                        if out.get("radius") is not None:
                            r = out["radius"].detach().float().reshape(-1).cpu().numpy()
                            metrics[f"hra/radius_hist_iter{i_iter}"] = wandb.Histogram(r)
                        if out.get("temperature") is not None:
                            t = (
                                out["temperature"]
                                .detach()
                                .float()
                                .reshape(-1)
                                .cpu()
                                .numpy()
                            )
                            metrics[f"hra/temp_hist_iter{i_iter}"] = wandb.Histogram(t)

                # Visualize with wandb
                if (
                    cfg.log_with == "wandb"
                    and (global_step + 1) % (cfg.get("vis_freq", 1000)) == 0
                ):
                    pcds = get_wandb_object_3d(
                        data["coords"],
                        data["features"],
                        gt_masks,
                        [aux[0]["best_masks"] > 0, aux[-1]["best_masks"] > 0],
                        [outputs[0]["prompt_coords"], outputs[-1]["prompt_coords"]],
                        [outputs[0]["prompt_labels"], outputs[-1]["prompt_labels"]],
                    )
                    metrics["pcd"] = pcds

                if cfg.log_with:
                    accelerator.log(metrics, step=global_step)

                global_step += 1
                step += 1
                last_train_outputs = outputs

            # 每个 batch 跑完后 pbar 前进 + checkpoint 保存
            pbar.update(1)
            loop_counter += 1
            if use_deepspeed:
                accelerator.wait_for_everyone()

            # 每 N 个「已消耗的 DataLoader batch」保存一次断点（与梯度微步 step 解耦）
            batch_save_freq = cfg.get("batch_save_freq", 50)
            if batch_save_freq > 0 and loop_counter % batch_save_freq == 0:
                if accelerator.is_main_process:
                    project_dir_path.mkdir(parents=True, exist_ok=True)
                    nb = batch_idx + 1
                    # 刚跑完的是 batch_idx；若已是本 epoch 最后一个 batch，应用「下一 epoch 起点」语义，
                    # 避免出现 next_batch_idx==train_len 导致永远 batch_idx < next_batch_idx
                    if nb >= train_len:
                        ckpt_epoch = epoch + 1
                        ckpt_next = 0
                    else:
                        ckpt_epoch = epoch
                        ckpt_next = nb
                    resume_state = {
                        "format_version": 2,
                        "epoch": ckpt_epoch,
                        "next_batch_idx": ckpt_next,
                        "global_step": global_step,
                        "micro_step": step,
                        "loop_counter": loop_counter,
                    }
                    torch.save(resume_state, resume_state_file)
                    print(
                        f"[Checkpoint] Saved resume state: epoch={ckpt_epoch}, next_batch_idx={ckpt_next}, "
                        f"micro_step={step}, global_step={global_step}, loop_counter={loop_counter}"
                    )
            
            if global_step >= cfg.max_steps:
                break

        pbar.close()

        # 保存最终断点状态
        if accelerator.is_main_process:
            project_dir_path.mkdir(parents=True, exist_ok=True)
            resume_state = {
                "format_version": 2,
                "epoch": epoch + 1,
                "next_batch_idx": 0,
                "global_step": global_step,
                "micro_step": step,
                "loop_counter": loop_counter,
                "training_mode": "deepspeed" if use_deepspeed else ("fsdp" if use_fsdp else "ddp"),
            }
            torch.save(resume_state, resume_state_file)

        # Save full model state (epoch-level checkpoint)
        if (epoch + 1) % cfg.get("save_freq", 1) == 0:
            if use_deepspeed:
                # DeepSpeed 使用自己的 checkpoint 格式
                ds_ckpt_dir = ckpt_dir / f"deepspeed_epoch_{epoch + 1}"
                ds_ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_checkpoint(str(ds_ckpt_dir), tag=f"epoch_{epoch + 1}")
                if accelerator.is_main_process:
                    print(f"[DeepSpeed] Saved checkpoint to {ds_ckpt_dir}")
            else:
                accelerator.save_state()

        if cfg.val_freq > 0 and (epoch + 1) % cfg.val_freq == 0:
            torch.cuda.empty_cache()
            # DeepSpeed 不支持 no_sync，需要使用 model.eval() 模式
            if use_deepspeed:
                with torch.no_grad():
                    val_metrics_this_epoch = validate()
            else:
                with accelerator.no_sync(model):
                    val_metrics_this_epoch = validate()
            torch.cuda.empty_cache()
            if cfg.log_with:
                val_log = {("val/" + k): v for k, v in val_metrics_this_epoch.items()}
                accelerator.log(val_log, step=global_step)

        # Learning Diagnostics: Check for overfitting and apply interventions
        if diagnostics is not None and adaptive_reg is not None:
            try:
                train_metrics = last_train_metrics_snap
                val_metrics = val_metrics_this_epoch
                
                # Record radius and curvature from last successful training batch
                if last_train_outputs is not None:
                    for out in last_train_outputs:
                        if out.get("radius") is not None:
                            # Get curvature from hyper_prompt_branch if available
                            curvature = None
                            if hasattr(model, 'hyper_prompt_branch'):
                                try:
                                    curvature = model.hyper_prompt_branch.get_curvature()
                                except Exception:
                                    pass
                            diagnostics.record_radius(out["radius"], curvature=curvature)
                            break
                
                # Get current LR
                current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else cfg.lr
                
                # Compute diagnostics
                diag_results = diagnostics.record(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=current_lr,
                )
                
                # Log diagnostics to wandb
                if cfg.log_with:
                    diag_metrics = {
                        f"diagnostics/{k}": v 
                        for k, v in diag_results.items()
                    }
                    accelerator.log(diag_metrics, step=global_step)
                
                # Check if intervention is needed
                should_intervene, reason = diagnostics.should_intervene()
                if should_intervene:
                    # Apply regularization interventions
                    interventions = adaptive_reg.intervene()
                    
                    # Log intervention
                    if cfg.log_with:
                        intervention_log = {
                            f"intervention/{k}": v 
                            for k, v in interventions.items()
                        }
                        intervention_log["intervention/reason"] = reason
                        accelerator.log(intervention_log, step=global_step)
                    
                    accelerator.print(f"[Epoch {epoch}] Intervention triggered: {reason}")
                    accelerator.print(f"[Epoch {epoch}] Actions: {interventions['actions']}")
                    
                    # Adjust learning rate if needed
                    # DeepSpeed 使用自己的学习率管理，不直接修改 optimizer.param_groups
                    if adaptive_reg.should_reduce_lr():
                        new_lr = adaptive_reg.get_adjusted_lr(cfg.lr)
                        if use_deepspeed:
                            # DeepSpeed: 使用 model.get_lr() 获取当前学习率
                            # 注意: DeepSpeed 学习率由配置文件控制，运行时调整有限
                            try:
                                current_ds_lr = model.get_lr()
                                accelerator.print(f"[Epoch {epoch}] DeepSpeed current LR: {current_ds_lr}")
                            except AttributeError:
                                pass
                            accelerator.print(f"[Epoch {epoch}] DeepSpeed LR adjustment requires config update (new_lr={new_lr})")
                        else:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            accelerator.print(f"[Epoch {epoch}] Learning rate reduced to {new_lr}")
                        
                    # Reset counters after intervention
                    diagnostics.reset_counters()
                    
            except Exception as e:
                accelerator.print(f"[Learning Diagnostics] Error during diagnostics: {e}")

        if global_step >= cfg.max_steps:
            break

    accelerator.end_training()


@torch.no_grad()
def get_wandb_object_3d(xyz, rgb, gt_masks, pred_masks, prompt_coords, prompt_labels):
    pcds = []
    xyz = xyz[0].cpu().numpy()  # [N, 3]
    rgb = (rgb[0].cpu().numpy() * 0.5 + 0.5) * 255  # [N, 3]
    gt_mask = gt_masks[0].cpu().numpy()  # [N]

    input_pcd = np.concatenate([xyz, rgb], axis=1)
    pcds.append(wandb.Object3D(input_pcd))

    gt_pcd = np.concatenate([xyz, gt_mask[:, None]], axis=1)
    pcds.append(wandb.Object3D(gt_pcd))

    # Only visualize the first sample
    for i, pred_mask in enumerate(pred_masks):
        pred_mask = pred_mask[0].cpu().numpy()
        # pred_pcd = np.concatenate([xyz, pred_mask[:, None]], axis=1)
        xyz2 = np.concatenate([xyz, prompt_coords[i][0].cpu().numpy()])
        pred_mask = np.concatenate([pred_mask, prompt_labels[i][0].cpu().numpy() + 2])
        pred_pcd = np.concatenate([xyz2, pred_mask[:, None]], axis=1)
        pcds.append(wandb.Object3D(pred_pcd))

    return pcds


def _cleanup_distributed() -> None:
    """避免 rank 异常退出时其他 rank 报 TCPStore / destroy_process_group 警告。"""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_distributed()
