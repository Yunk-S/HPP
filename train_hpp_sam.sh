#!/bin/bash
# =============================================================================
# HPP-SAM Unified Training Script for SLURM HPC
#
# 推荐（根目录调度脚本，n6 上 1/4/8 卡）:
#   ./sbatch_hpp_sam_n6.sh 1
#   ./sbatch_hpp_sam_n6.sh 4
#   ./sbatch_hpp_sam_n6.sh 8
#
# 分区 / GPU 等与集群不一致时，在运行 sbatch_hpp_sam_n6.sh 前 export，例如:
#   export SLURM_PARTITION=gpua8001t
#   export SLURM_GPU_TYPE=a800
#
# 手动 sbatch（命令行会覆盖下面 #SBATCH；分区必须是队列名 gpua8001t，勿写 gpua800n6 节点名）:
#   sbatch --partition=gpua8001t \
#     --gres=gpu:a800:4 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=32 \
#     --mem=96G \
#     train_hpp_sam.sh
#
# 其它 Hydra / 数据路径仍可用环境变量:
#   CONFIG=hpp_voronoi ./sbatch_hpp_sam_n6.sh 4
#
# Conda 环境名（默认 hpp；勿再用 pointsam 旧环境，易混 cu130/torkit3d ABI）:
#   export HPP_CONDA_ENV=hpp
# =============================================================================

# Conda 环境（与 scripts/init_hpp_env.sh 一致）
HPP_CONDA_ENV="${HPP_CONDA_ENV:-hpp}"

# -------------------------- SLURM Parameters --------------------------------
# 注意：#SBATCH 由 sbatch 在提交前解析，不会做 ${VAR} 展开。
# 以下为「单机单任务」默认；多卡请用根目录 ./sbatch_hpp_sam_n6.sh 或 sbatch 命令行覆盖。
# 分区名须为 Slurm 真实 partition（xgpua800n6 在 gpua8001t，不是 gpua800n6）。
#SBATCH --job-name=hpp_sam_train
#SBATCH --partition=gpua8001t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00

#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=yunkun.shi23@student.xjtlu.edu.cn

# ============================================================
# 日志：由 sbatch_hpp_sam_n6.sh 传入的 --output / --error（TRAIN_OUT_LOG / TRAIN_ERR_LOG）
# 捕获本脚本 stdout/stderr。**勿**再 exec tee 写同一路径，否则 Slurm 与 tee 各写一遍 → 双份日志。
# 非 Slurm（直接 bash train_hpp_sam.sh）时无 TRAIN_*，输出仍走终端。
# ============================================================
set +x 2>/dev/null || true
mkdir -p logs
_TRAIN_TS="${TRAIN_TS:-$(date '+%Y%m%d-%H%M%S')}"
_TRAIN_OUT_LOG="${TRAIN_OUT_LOG:-logs/slurm-${_TRAIN_TS}.out}"
_TRAIN_ERR_LOG="${TRAIN_ERR_LOG:-logs/slurm-${_TRAIN_TS}.err}"
echo ">>> [train_hpp_sam.sh] Slurm/日志: stdout=${TRAIN_OUT_LOG} stderr=${TRAIN_ERR_LOG}（若由 sbatch 提交应与之一致）"
echo ">>> [train_hpp_sam.sh] Job: ${SLURM_JOB_NAME:-?} ID: ${SLURM_JOB_ID:-?} Node: ${SLURM_JOB_NODELIST:-?}"
echo ">>> [train_hpp_sam.sh] WDIR: $(pwd)  Host: $(hostname -s)  PID: $$"
echo "============================================"

# Conda 环境（与 scripts/init_hpp_env.sh 一致）
HPP_CONDA_ENV="${HPP_CONDA_ENV:-hpp}"

# -------------------------- Configuration ----------------------------------
# 训练配置选择（通过环境变量覆盖）
# 可选值:
#   - hpp               : Giant骨干（大规模，推荐8卡）
#   - hpp_memory_efficient : EVA02 Base + 显存优化（默认）
#   - hpp_voronoi       : Voronoi分词器（最新优化）
CONFIG="${CONFIG:-hpp_memory_efficient}"

# 数据集配置
DATASET="${DATASET:-partnext}"  # partnext, partnet, partnet+partnext
DATA_ROOT="/gpfs/work/aac/yunkunshi23/HPP-SAM/data/partnext/dataset"

# 混合精度设置（通过环境变量覆盖）
# 可选: bf16, fp16, no
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# -------------------------- Job Info ----------------------------------------
echo "============================================"
echo "HPP-SAM Unified Training Script"
echo "============================================"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "GPUs/Node:    $SLURM_JOB_GPUS"
echo "CPUs/Task:    $SLURM_CPUS_PER_TASK"
echo "Config:       $CONFIG"
echo "Dataset:      $DATASET"
echo "Precision:    $MIXED_PRECISION"
echo "============================================"

# -------------------------- Environment --------------------------------------
# PyTorch cu126 wheel 自带 CUDA 运行时；训练一般只需节点驱动。
# cuda/12.8 module 主要用于 nvcc 编译扩展（见 init_hpp_env）；加载后可能污染 LD_LIBRARY_PATH，
# 与 wheel 内 cudart 混用曾导致 GPU 枚举异常。默认不加载；确需 PATH 里有 nvcc 时:
#   export HPP_LOAD_CUDA_TOOLKIT_MODULE=1
module purge
if [[ "${HPP_LOAD_CUDA_TOOLKIT_MODULE:-0}" == "1" ]]; then
  module load cuda/12.8.0-none-none-xmhtcei
fi
module load anaconda3/2023.09-0-none-none-3te2njg

eval "$(conda shell.bash hook)"
conda activate "$HPP_CONDA_ENV"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$HPP_CONDA_ENV" ]]; then
    echo "ERROR: 未能激活 conda 环境 \"$HPP_CONDA_ENV\"（当前 CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-空}）。"
    echo "  请先: conda create -n $HPP_CONDA_ENV python=3.10 -y"
    echo "  再按仓库 scripts/init_hpp_env.sh 装依赖。"
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================
# 关键修复 A：移除 conda compiler_compat linker wrapper
#
# 问题：conda 环境自带 compiler_compat/ld wrapper，在子进程 JIT 编译时被调用
#       来链接 CUDA 库（如 libcufile.so），但它与 CUDA 12.6 不兼容：
#       undefined reference to `dlopen', `dlclose', `dlsym', `dlvsym'...
# 症状：collect2: error: ld returned 1 exit status
# 解决：移除 conda 的 linker wrapper，让系统 linker (/usr/bin/ld) 处理链接
# ============================================================
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CONDA_COMPILER_COMPAT_LD="${CONDA_PREFIX}/compiler_compat/ld"
  if [[ -f "$CONDA_COMPILER_COMPAT_LD" ]]; then
    echo ">>> [fix-ld] 检测到 conda compiler_compat linker wrapper"
    echo ">>> [fix-ld] 路径: $CONDA_COMPILER_COMPAT_LD"
    if [[ -f "${CONDA_COMPILER_COMPAT_LD}.bak" ]]; then
      echo ">>> [fix-ld] 备份已存在，跳过备份"
    else
      cp "$CONDA_COMPILER_COMPAT_LD" "${CONDA_COMPILER_COMPAT_LD}.bak"
      echo ">>> [fix-ld] 备份到: ${CONDA_COMPILER_COMPAT_LD}.bak"
    fi
    rm -f "$CONDA_COMPILER_COMPAT_LD"
    echo ">>> [fix-ld] 已移除，后续 JIT 编译将使用系统 linker"
  fi
fi

# ============================================================
# 关键修复 B：设置 LDFLAGS 让链接器正确链接 libcufile.so
#
# 根因（DeepSpeed 官方 Issue #6461/#6593）：
#   - libcufile.so（CUDA GPU Direct Storage 库）依赖 libdl.so 提供的
#     dlopen/dlsym/dlclose/dlerror/dlvsym 符号
#   - conda linker wrapper 和部分系统 linker 调用链没有自动加入 -ldl
#   - 链接时出现 "undefined reference to `dlopen'" 等错误，collect2 返回 1
#
# 修复（官方确认有效）：
#   - LDFLAGS="-Wl,--no-as-needed -ldl"
#     -Wl,--no-as-needed  : 强制链接 libdl，不被 "看起来没直接引用" 而跳过
#     -ldl                 : 链接动态加载器库（提供 dlopen 系列符号）
#
# 注意：deepspeed.initialize() 即使 DS_JIT=0 也会触发 C++/CUDA 扩展的
#       JIT 编译（_install_cuda_extensions 无条件尝试构建未安装的 op），
#       所以 LDFLAGS 是训练前必须设置的。
# ============================================================
export LDFLAGS="-Wl,--no-as-needed -ldl"
export CFLAGS="${CFLAGS:-} -I/usr/include"
echo ">>> [fix-ld] 设置 LDFLAGS=$LDFLAGS (让 libcufile.so 正确链接 dlopen/dlsym)"
echo ">>> [fix-ld] 注意：deepspeed.initialize() 即使 DS_JIT=0 也会触发 JIT 编译"

# ============================================================
# 关键修复 C：LD_PRELOAD libdl + LD_LIBRARY_PATH 含 /lib64（与 ZeRO-3 烟测一致）
#
# 现象：libcufile.so 动态加载/链接阶段 undefined reference to dlopen 等 → collect2 失败。
# 解决：预加载系统 libdl.so.2，并保证链接器搜索路径含 /lib64（RHEL 类节点）。
# ============================================================
if [[ -n "${LD_PRELOAD:-}" ]]; then
  echo ">>> [fix-ld-preload] 重置 LD_PRELOAD（避免与 conda 注入冲突）: ${LD_PRELOAD}"
  unset LD_PRELOAD LD_PRELOAD_32
fi
_LDLIB=""
for _ldpath in \
  "/usr/lib/x86_64-linux-gnu/libdl.so.2" \
  "/lib64/libdl.so.2" \
  "/lib/x86_64-linux-gnu/libdl.so.2"; do
  if [[ -f "$_ldpath" ]]; then
    _LDLIB="$_ldpath"
    break
  fi
done
if [[ -n "$_LDLIB" ]]; then
  export LD_PRELOAD="$_LDLIB"
  echo ">>> [fix-ld-preload] LD_PRELOAD=$LD_PRELOAD"
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="/lib64"
  else
    export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH:/lib64" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')"
  fi
  echo ">>> [fix-ld-preload] LD_LIBRARY_PATH（已去重并含 /lib64）=$LD_LIBRARY_PATH"
else
  echo ">>> [fix-ld-preload] 警告：未找到 libdl.so.2，跳过 LD_PRELOAD（若 DeepSpeed JIT 失败请检查节点路径）"
fi

# PyTorch 显存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export ACCELERATE_FSDP_WARNING=0

# DeepSpeed 环境变量
export DS_AIO_BLOCK_SIZE=1048576
export DS_IN_ADEFAULT_MODE=1
# 禁用 DeepSpeed 的异步 IO 以提高稳定性
export DS_NO_ASYNC_LOADING=1
# 跳过 GDS / libcufile 相关构建链（与 Issue #6461/#6593、本仓库烟测一致）
export DS_BUILD_GDS=0
# 节点 nvcc 与 torch wheel 的 CUDA 主版本不一致时，避免 DeepSpeed 扩展编译前置检查直接失败
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

# NCCL 超时：ProcessGroupNCCL watchdog 超时，单位为毫秒（ms）。
# 调试时缩至 300000ms（5分钟）快速定位问题；确认稳定后可改回 1800000ms（30分钟）。
# 注意：PyTorch 2.x 忽略环境变量 NCCL_TIMEOUT，真实超时由 DeepSpeed deepspeed_config['nccl_timeout'] 控制。
export NCCL_TIMEOUT=300000

# 如果使用 NVMe offload，创建临时目录
if [ -d "/tmp" ]; then
    export DS_NVME_PATH="/tmp/deepspeed_nvme_${SLURM_JOB_ID:-$$}"
    mkdir -p "$DS_NVME_PATH"
fi

echo "============================================"
echo "Environment Info:"
echo "  CWD:        $(pwd)"
echo "  Python:     $(which python)"
echo "  PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:       $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPUs:       $(python -c 'import torch; print(torch.cuda.device_count())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "  GPU Name:   $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")')"
fi
echo "============================================"

# -------------------------- Preflight（避免旧 pointsam：cu130、torkit3d 符号、DeepSpeed+torch 不匹配）---
echo "============================================"
echo "Preflight: torch / torkit3d / (deepspeed)"
echo "============================================"
PREFLIGHT_DS=0
[[ "$DIST_BACKEND" == "deepspeed" ]] && PREFLIGHT_DS=1
if ! python - "$PREFLIGHT_DS" <<'PY'
import sys
need_ds = int(sys.argv[1])

def die(msg):
    print("ERROR:", msg)
    sys.exit(1)

import torch
if not hasattr(torch.amp, "custom_fwd"):
    die(
        "torch.amp 无 custom_fwd：当前 PyTorch 过旧，无法满足 DeepSpeed 0.13+。"
        "请安装 torch>=2.2（推荐按 scripts/init_hpp_env.sh 使用 torch 2.11+cu126）。"
    )

ver, tc = torch.__version__, torch.version.cuda or ""
if "+cu130" in ver or tc.startswith("13"):
    print(
        "WARN: PyTorch 为 CUDA 13 构建。集群若无 cuda/13 toolkit，"
        "torkit3d 用 nvcc 12.x 编译时会报「detected CUDA vs PyTorch 不一致」。"
        "推荐: cu126 wheel；编译 torkit3d 时再 module load cuda/12.8（见 init_hpp_env.sh）。"
    )

if need_ds:
    try:
        import deepspeed  # noqa: F401
    except Exception as e:
        die(f"import deepspeed 失败（DIST_BACKEND=deepspeed）: {e}")

try:
    from torkit3d.ops.sample_farthest_points import sample_farthest_points  # noqa: F401
except Exception as e:
    die(
        "torkit3d CUDA 扩展无法加载（常见：旧 wheel 与当前 torch ABI 不一致）。"
        f" 详情: {e}\n"
        "  修复: 在带 nvcc 的环境（如 module load cuda/12.8）下执行\n"
        "    pip uninstall -y torkit3d\n"
        "    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0;8.6' pip install --no-build-isolation -v ./third_party/torkit3d"
    )

dsv = ""
if need_ds:
    import deepspeed
    dsv = deepspeed.__version__
print(f"Preflight OK | torch {ver} | cuda {tc} | deepspeed {dsv or '(skipped)'}")
PY
then
    echo "Preflight 失败，已中止。请按上方 ERROR 修复环境后再 sbatch。"
    exit 1
fi
echo "============================================"

# -------------------------- WANDB --------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARN: WANDB_API_KEY 未设置，wandb 可能无法登录。"
fi

# -------------------------- Accelerate / DeepSpeed Config --------------------------------
NUM_NODES=1
# 单机单任务多 GPU：SLURM_GPUS_ON_NODE 常为分配卡数；若无则使用提交时注入的 NUM_GPUS
NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-1}}"
MAIN_PORT=29500

MASTER_ADDR=$(hostname -s)

# 选择分布式训练后端
# 选项:
#   - deepspeed  : DeepSpeed ZeRO-3 (推荐，已针对 CPU 内存优化)
#   - fsdp       : PyTorch FSDP (CPU 内存管理不如 DeepSpeed 精细)
#   - ddp        : 纯 DDP (无分片，显存占用大)
DIST_BACKEND="${DIST_BACKEND:-deepspeed}"

echo "============================================"
echo "Distributed Training Config:"
echo "  NUM_NODES:     $NUM_NODES"
echo "  NUM_PROCESSES: $NUM_PROCESSES"
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_PORT:   $MAIN_PORT"
echo "  DIST_BACKEND:  $DIST_BACKEND"
echo "============================================"

# -------------------------- Dataset Config ------------------------------------
case "$DATASET" in
    partnext)
        TRAIN_DATASET="partnext"
        VAL_DATASET="partnext_val"
        ;;
    partnet)
        TRAIN_DATASET="partnet"
        VAL_DATASET="partnet_val"
        ;;
    partnet+partnext)
        TRAIN_DATASET="partnet+partnext"
        VAL_DATASET="partnext_val"
        ;;
    *)
        echo "ERROR: Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# -------------------------- Training Command ----------------------------------
echo "============================================"
echo "Training Configuration:"
echo "  Config:           $CONFIG"
echo "  Train Dataset:   $TRAIN_DATASET"
echo "  Val Dataset:      $VAL_DATASET"
echo "  Data Root:        $DATA_ROOT"
echo "  Mixed Precision:  $MIXED_PRECISION"
echo "  Dist Backend:     $DIST_BACKEND"
echo "============================================"

# 构建通用参数
COMMON_ARGS="train.py \
    --config $CONFIG \
    dataset@train_dataset=$TRAIN_DATASET \
    dataset@val_dataset=$VAL_DATASET \
    train_dataset.path=$DATA_ROOT \
    val_dataset.path=$DATA_ROOT \
    pretrained_ckpt_path=null"

# 根据分布式后端选择启动命令
case "$DIST_BACKEND" in
    deepspeed)
        echo "[DeepSpeed] Using DeepSpeed ZeRO-3 backend"
        # import deepspeed 已在上方 Preflight 中校验（DIST_BACKEND=deepspeed 时）

        # DeepSpeed 使用 deepspeed.launch 启动
        TRAIN_CMD="deepspeed \
            --num_gpus=$NUM_PROCESSES \
            $COMMON_ARGS \
            log_with=wandb"
        ;;
    fsdp)
        echo "[FSDP] Using PyTorch FSDP backend"
        
        # FSDP 使用 accelerate launch
        TRAIN_CMD="accelerate launch \
            --main_training_function main \
            --num_processes $NUM_PROCESSES \
            --num_machines $NUM_NODES \
            --machine_rank 0 \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MAIN_PORT \
            --mixed_precision $MIXED_PRECISION \
            --dynamo_backend no \
            $COMMON_ARGS \
            log_with=wandb"
        ;;
    ddp)
        echo "[DDP] Using pure DDP backend (no sharding, high memory usage)"
        
        # 纯 DDP 使用 accelerate launch
        TRAIN_CMD="accelerate launch \
            --main_training_function main \
            --num_processes $NUM_PROCESSES \
            --num_machines $NUM_NODES \
            --machine_rank 0 \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MAIN_PORT \
            --mixed_precision $MIXED_PRECISION \
            --dynamo_backend no \
            $COMMON_ARGS \
            log_with=wandb"
        ;;
    *)
        echo "ERROR: Unknown DIST_BACKEND: $DIST_BACKEND"
        echo "Valid options: deepspeed, fsdp, ddp"
        exit 1
        ;;
esac

echo "============================================"
echo "Executing Command:"
echo "$TRAIN_CMD"
echo "============================================"

eval "$TRAIN_CMD"
train_rc=$?

echo "============================================"
echo "Training Finished at: $(date)"
echo "Exit Code: $train_rc"
echo "============================================"
exit $train_rc
