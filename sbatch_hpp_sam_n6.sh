#!/usr/bin/env bash
# 仓库根目录提交 Slurm 作业：n6 类节点上 1 / 4 / 8 卡（单任务 + 多进程）
#
#   ./sbatch_hpp_sam_n6.sh 1
#   ./sbatch_hpp_sam_n6.sh 4
#   ./sbatch_hpp_sam_n6.sh 8
#
# 默认：分区 gpua8001t（队列名；xgpua800n6 是节点名，不能当 --partition）
# 等价手动 4 卡: sbatch --partition=gpua8001t --gres=gpu:a800:4 --ntasks-per-node=1 --cpus-per-task=32 --mem=96G train_hpp_sam.sh
# 换 4090 n6：export SLURM_PARTITION=gpu40901t SLURM_GPU_TYPE=4090
# 强制某节点：export SLURM_NODELIST=xgpua800n6
# 内存：n6 约 100GB，默认申请 96G；大内存节点可 export SLURM_MEM=256G

# 分布式后端选项: deepspeed (推荐), fsdp, ddp
# 默认使用 deepspeed，它有更好的 CPU 内存管理
# Conda 环境名（须与 train_hpp_sam.sh / scripts/init_hpp_env.sh 一致）:
#   export HPP_CONDA_ENV=hpp
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 生成唯一时间戳：确保 Slurm 日志路径在作业提交时就固定下来，
# 而非等 %j（job ID）在计算节点上才展开——那样的话作业开始前本地根本不会有日志文件。
# 格式：logs/slurm-<date>-<time>.out / .err
_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
mkdir -p logs
export TRAIN_OUT_LOG="logs/slurm-${_TIMESTAMP}.out"
export TRAIN_ERR_LOG="logs/slurm-${_TIMESTAMP}.err"
echo "[sbatch wrapper] Slurm logs will go to: $TRAIN_OUT_LOG / $TRAIN_ERR_LOG"

NGPU="${1:-}"
if [[ ! "$NGPU" =~ ^(1|4|8)$ ]]; then
  echo "Usage: $0 1|4|8"
  echo "  SLURM_PARTITION (default: gpua8001t)  SLURM_GPU_TYPE (default: a800)"
  echo "  SLURM_NODELIST (optional)  SLURM_MEM (default: 96G)"
  echo "  DIST_BACKEND (default: deepspeed) - Options: deepspeed, fsdp, ddp"
  echo "    deepspeed: Recommended, better CPU memory management"
  echo "    fsdp: PyTorch FSDP (CPU memory management less refined)"
  echo "    ddp: Pure DDP (high memory usage, no sharding)"
  exit 1
fi

PARTITION="${SLURM_PARTITION:-gpua8001t}"
GPU_TYPE="${SLURM_GPU_TYPE:-a800}"
MEM="${SLURM_MEM:-96G}"

case "$NGPU" in
  1) CPUS=32 ;;
  4) CPUS=32 ;;
  8) CPUS=64 ;;
esac

mkdir -p logs
export NUM_GPUS="$NGPU"
export DIST_BACKEND="$DIST_BACKEND"

SBATCH_EXTRA=()
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  SBATCH_EXTRA+=(--nodelist="$SLURM_NODELIST")
fi

echo "Submitting: partition=$PARTITION mem=$MEM gres=gpu:${GPU_TYPE}:${NGPU} ntasks-per-node=1 cpus-per-task=$CPUS NUM_GPUS=$NUM_GPUS DIST_BACKEND=$DIST_BACKEND ${SLURM_NODELIST:+nodelist=$SLURM_NODELIST}"
echo "Slurm stdout -> $TRAIN_OUT_LOG"
echo "Slurm stderr -> $TRAIN_ERR_LOG"

sbatch \
  --partition="$PARTITION" \
  --mem="$MEM" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres="gpu:${GPU_TYPE}:${NGPU}" \
  --cpus-per-task="$CPUS" \
  --output="$TRAIN_OUT_LOG" \
  --error="$TRAIN_ERR_LOG" \
  --export=ALL \
  "${SBATCH_EXTRA[@]}" \
  "$ROOT/train_hpp_sam.sh"
