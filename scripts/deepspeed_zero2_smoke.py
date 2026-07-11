#!/usr/bin/env python3
"""
ZeRO-2 多卡冒烟测试：验证 deepspeed launch + ZeRO-2 前向/反向能否跑通。
ZeRO-2 比 ZeRO-3 更稳定，更不容易出错。

用法：
  deepspeed --num_gpus=4 scripts/deepspeed_zero2_smoke.py --steps 30
  deepspeed --num_gpus=1 scripts/deepspeed_zero2_smoke.py --steps 10  # 单卡测试
"""
from __future__ import annotations

import argparse
import os
import sys

# ---- 最早期：禁用所有 DeepSpeed C++/CUDA extension 的 JIT 编译 ----
os.environ["DS_JIT"] = "0"
os.environ["DS_BUILD_GDS"] = "0"

# ---- argparse 在所有 import 之前 ----
parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-2 smoke test")
parser.add_argument("--steps", type=int, default=30, help="优化步数")
parser.add_argument(
    "--no-wall-clock",
    action="store_true",
    help="关闭 wall_clock_breakdown（日志更短）",
)
parser.add_argument(
    "--no-cpu-offload",
    action="store_true",
    help="禁用优化器 CPU offload（默认开启）",
)
parser.add_argument(
    "--no-jit",
    action="store_true",
    help="禁用 DeepSpeed JIT 编译（DS_JIT=0），用于排查问题",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=int(os.environ.get("LOCAL_RANK", "0")),
    help="DeepSpeed launcher 注入，不要手动指定",
)
args = parser.parse_args()

# ---- 早期打印 ----
print(
    f"[smoke] START pid={os.getpid()} "
    f"LOCAL_RANK={os.environ.get('LOCAL_RANK','?')} "
    f"WORLD_SIZE={os.environ.get('WORLD_SIZE','?')} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
    flush=True,
)

import time
import traceback

import torch

print(
    f"[smoke] torch={torch.__version__} "
    f"cuda={torch.version.cuda} "
    f"device_count={torch.cuda.device_count()}",
    flush=True,
)

if torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"[smoke] GPU{i}={torch.cuda.get_device_name(i)}", flush=True)

import deepspeed

print(f"[smoke] import deepspeed {deepspeed.__version__} OK", flush=True)


class _TinyModel(torch.nn.Module):
    """简单的 MLP 模型，用于 ZeRO-2 测试"""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4096, 8192),
            torch.nn.GELU(),
            torch.nn.Linear(8192, 8192),
            torch.nn.GELU(),
            torch.nn.Linear(8192, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).sum()


def _ds_config(*, cpu_offload: bool, wall_clock: bool) -> dict:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    z: dict = {
        "stage": 2,  # ZeRO-2（不同于 ZeRO-3 的 stage: 3）
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50_000_000,
    }
    if cpu_offload:
        z["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    return {
        "train_batch_size": ws,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1,
        "wall_clock_breakdown": wall_clock,
        "zero_optimization": z,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
    }


def _fail_if_no_usable_cuda(local_rank: int) -> None:
    """在 set_device 之前失败，避免仅看到 launcher 的 return code=1 / sigkill_handler。"""
    import socket

    if not torch.cuda.is_available():
        print(
            "[smoke] FATAL: torch.cuda.is_available()=False（本机无可用 CUDA 运行时/驱动，"
            "或当前节点未分配 GPU）。不要在登录节点直接 deepspeed --num_gpus；"
            "请 sbatch/salloc 到带 GPU 的计算节点后再跑。",
            flush=True,
        )
        print(
            f"[smoke] hint: host={socket.gethostname()} "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
            flush=True,
        )
        sys.exit(1)
    n = torch.cuda.device_count()
    if n < 1:
        print(
            "[smoke] FATAL: torch.cuda.device_count()==0（常见：登录节点无 NVIDIA 驱动/GPU，"
            "或 Slurm 作业未申请 --gres=gpu）。",
            flush=True,
        )
        print(
            f"[smoke] hint: host={socket.gethostname()} "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
            flush=True,
        )
        sys.exit(1)
    if local_rank < 0 or local_rank >= n:
        print(
            f"[smoke] FATAL: LOCAL_RANK={local_rank} 超出可见 GPU 数 {n}（launcher 与作业实际可见卡数不一致）。",
            flush=True,
        )
        sys.exit(1)


def main() -> None:
    if args.no_jit:
        os.environ["DS_JIT"] = "0"
        print("[smoke] DS_JIT=0（禁用 JIT 编译）", flush=True)

    print(
        f"[smoke] env: LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')} "
        f"LD_PRELOAD={os.environ.get('LD_PRELOAD','')} "
        f"DS_BUILD_GDS={os.environ.get('DS_BUILD_GDS','')}",
        flush=True,
    )

    cpu_offload = not args.no_cpu_offload
    wall_clock = not args.no_wall_clock

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _fail_if_no_usable_cuda(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[smoke] rank={local_rank} building model...", flush=True)
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)
    cfg = _ds_config(cpu_offload=cpu_offload, wall_clock=wall_clock)

    print(f"[smoke] rank={local_rank} calling deepspeed.initialize... "
          f"cpu_offload={cpu_offload}", flush=True)
    try:
        engine, opt, _, _ = deepspeed.initialize(
            model=model,
            optimizer=opt,
            config=cfg,
            dist_init_required=True,
        )
        print(f"[smoke] rank={local_rank} deepspeed.initialize OK", flush=True)
    except Exception as init_err:
        print(
            f"[smoke] FATAL deepspeed.initialize FAILED rank={local_rank}: {init_err}",
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)

    engine.train()
    t0 = time.perf_counter()
    for step in range(args.steps):
        x = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
        try:
            loss = engine(x)
            engine.backward(loss)
            engine.step()
        except Exception as step_err:
            print(
                f"[smoke] FATAL step={step} FAILED rank={local_rank}: {step_err}",
                flush=True,
            )
            traceback.print_exc()
            sys.exit(1)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if local_rank == 0:
        dt = time.perf_counter() - t0
        print(
            f"[ZeRO-2 smoke] OK | steps={args.steps} | cpu_offload={cpu_offload} | "
            f"wall_clock_breakdown={wall_clock} | {dt:.2f}s total",
            flush=True,
        )


if __name__ == "__main__":
    main()
