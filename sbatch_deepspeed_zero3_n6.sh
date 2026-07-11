#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 在 n4（默认 xgpua800n4）上申请 4×GPU，跑 DeepSpeed 环境报告 + ZeRO-3 冒烟测试。
# （文件名含 n6 为历史命名；默认节点见下方 SLURM_NODELIST。）
#
# 提交：
#   chmod +x sbatch_deepspeed_zero3_n6.sh scripts/run_deepspeed_zero3_smoke_on_node.sh
#   ./sbatch_deepspeed_zero3_n6.sh
#
# 换分区 / GPU 类型（sinfo：xgpua800n4/n5/n6 均在 gpua8001t）:
#   export SLURM_PARTITION=gpua8001t SLURM_GPU_TYPE=a800
# 换节点:
#   export SLURM_NODELIST=xgpua800n5
#   # 或 xgpua800n6
#
# 链接错误 "collect2: error: ld returned 1 exit status"
#            "undefined reference to `dlopen', `dlclose', `dlsym', `dlvsym'"
#   -> DeepSpeed 官方确认的 libcufile.so / libdl 链接顺序 bug（Issue #6461）
#   -> 修复：run 脚本会自动设置 LDFLAGS="-Wl,--no-as-needed -ldl"（见脚本内注释）
#
# 与 hpp_memory_efficient 对齐（优化器 CPU offload）:
#   export ZERO3_SMOKE_CPU_OFFLOAD=1
#   ./sbatch_deepspeed_zero3_n6.sh
#
# 仅环境报告:
#   export ZERO3_SMOKE_DS_REPORT_ONLY=1
#   ./sbatch_deepspeed_zero3_n6.sh
#
# 若在 Linux 上报 bash\r：仓库根目录执行 git add --renormalize '*.sh' && git status
# 或: sed -i 's/\r$//' sbatch_deepspeed_zero3_n6.sh scripts/run_deepspeed_zero3_smoke_on_node.sh
#
# import deepspeed 报 device=3,num_gpus=3：先试 pip uninstall -y triton；冒烟脚本会按 torch 实际可见卡数起 deepspeed。
# 强制期望卡数（仅告警）: export ZERO3_SMOKE_NUM_GPUS=4
# 降级单卡冒烟（调试 SIGKILL）: export ZERO3_SMOKE_SINGLE_GPU=1
# conda run 绕过 conda linker wrapper（终极方案）: export ZERO3_SMOKE_USE_CONDA_RUN=1
# 禁用 JIT（排查 JIT 编译 SIGKILL）: export ZERO3_SMOKE_NO_JIT=1
# 禁用 CPU offload（纯 GPU）: export ZERO3_SMOKE_NO_CPU_OFFLOAD=1
# torchrun launcher（fork 模式，绕过 spawn SIGKILL）: export ZERO3_SMOKE_LAUNCHER=torchrun
# 关键修复：
#   1. run 脚本已默认设置 DS_SKIP_CUDA_CHECK=1（绕过 nvcc 12.2 vs torch cu126 版本不匹配）
#   2. 已默认开启 --enable_each_rank_log=logs/rank_logs_*/（每个 rank 的真实错误会写到日志）
#   3. 已设置 LD_PRELOAD=/lib64/libdl.so.2 + LD_LIBRARY_PATH=/lib64（解决 libcufile.so dlopen 符号缺失）
#   4. 已移除 --module（避免 ModuleNotFoundError）
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PARTITION="${SLURM_PARTITION:-gpua8001t}"
GPU_TYPE="${SLURM_GPU_TYPE:-a800}"
NODELIST="${SLURM_NODELIST:-xgpua800n4}"
MEM="${SLURM_MEM:-96G}"
CPUS="${SLURM_CPUS:-32}"

mkdir -p logs

echo "Submitting ZeRO-3 smoke: partition=$PARTITION nodelist=$NODELIST gres=gpu:${GPU_TYPE}:4"

# HPP_REPO_ROOT：避免计算节点上 batch 脚本在 /var/spool/slurmd/scripts/ 执行时
# BASH_SOURCE 指向 spool，误把仓库根推成 /var/spool/slurmd（子进程找不到 .py）。
sbatch \
  --job-name=ds-zero3-smoke \
  --partition="$PARTITION" \
  --nodelist="$NODELIST" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres="gpu:${GPU_TYPE}:4" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="${SLURM_TIME:-00:30:00}" \
  --output=logs/slurm-ds-zero3-smoke-%j.out \
  --error=logs/slurm-ds-zero3-smoke-%j.err \
  --export=ALL,HPP_REPO_ROOT="$ROOT" \
  "$ROOT/scripts/run_deepspeed_zero3_smoke_on_node.sh"
