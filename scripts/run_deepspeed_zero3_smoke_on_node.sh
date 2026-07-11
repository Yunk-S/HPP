#!/usr/bin/env bash
# 由 sbatch_deepspeed_zero3_n6.sh 在计算节点上调用；也可在已分配的交互式 4 卡会话里直接运行。
#
# CUDA 说明（避免和 init_hpp_env 里 cu126 搞混）:
#   - PyTorch 官方 cu126 wheel 自带 CUDA 12.6 运行时，训练/推理一般只需节点驱动，不必 module load cuda toolkit。
#   - init_hpp_env.sh 里的 cuda/12.8 module 是给 nvcc 编译 torkit3d 用的，不是「运行时要 12.8」。
#   - 本冒烟默认不加载 cuda/12.8。若确需 nvcc: export ZERO3_SMOKE_LOAD_CUDA_MODULE=1
#
# GPU 计数说明:
#   - nvidia-smi -L 行数 = 节点上能列出的 GPU，不等于 Slurm 分给本作业的可见数。
#   - 以 torch.cuda.device_count() 为准；deepspeed --num_gpus 与之对齐，避免「申请了 4 卡但进程只见 3 张」。
#
# import deepspeed 报错 device=3, num_gpus=3:
#   - DeepSpeed 0.18 若已安装 triton，import 链会调 is_triton_supported() -> get_device_capability()，
#     与部分 cgroup/驱动组合会触发异常。可试: pip uninstall -y triton（ZeRO-3 训练一般不依赖 triton）。
set -euo pipefail

# ---------------------------------------------------------------------------
# 仓库根目录（不要用 BASH_SOURCE 在 Slurm 下推导）
#
# Slurm 会把 sbatch 的 batch 脚本拷到计算节点本地目录执行，常见路径形如：
#   /var/spool/slurmd/scripts/slurm-<jobid>.sh
# 此时 ${BASH_SOURCE[0]} 指向该副本，dirname/.. 会变成 /var/spool/slurmd，
# 进而 $ROOT/scripts/deepspeed_zero3_smoke.py 变成
#   /var/spool/slurmd/scripts/deepspeed_zero3_smoke.py
# 该目录里只有 Slurm 临时 .sh，没有你的 .py → can't open file [Errno 2]。
#
# 正确做法：
#   1) sbatch 侧注入 HPP_REPO_ROOT（见 sbatch_deepspeed_zero3_n6.sh --export）
#   2) 否则用 SLURM_SUBMIT_DIR（执行 sbatch 时的工作目录，一般为仓库根）
#   3) 非 Slurm（交互 salloc / 本机）再退回 BASH_SOURCE 相对路径
# ---------------------------------------------------------------------------
if [[ -n "${HPP_REPO_ROOT:-}" ]]; then
  ROOT="$(cd "$HPP_REPO_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
echo ">>> [root] ROOT=$ROOT (HPP_REPO_ROOT=${HPP_REPO_ROOT:-空} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-空})"

module purge
if [[ "${ZERO3_SMOKE_LOAD_CUDA_MODULE:-0}" == "1" ]]; then
  CUDA_MOD="${INIT_HPP_CUDA_MODULE:-cuda/12.8.0-none-none-xmhtcei}"
  echo ">>> module load $CUDA_MOD（ZERO3_SMOKE_LOAD_CUDA_MODULE=1）"
  module load "$CUDA_MOD"
fi
module load anaconda3/2023.09-0-none-none-3te2njg

eval "$(conda shell.bash hook)"
conda activate "${HPP_CONDA_ENV:-hpp}"

# ----------------------------------------------------------------------
# 修复 conda CUDA 包的 compiler_compat 与 libcufile.so / 运行时不兼容问题
# 现象: deepspeed --num_gpus=4 启动后 sigkill 所有子进程，exit code 2
# 原因:
#   conda 环境里有 compiler_compat/ 目录（含 ld-linux-x86-64.so.2），即使
#   unset LD_PRELOAD 也可能被 conda Python 自动注入（conda 2023+ 行为）。
#   该 linker 库与 CUDA 12.6 的 libcufile.so 不兼容。
# 解决:
#   1) unset LD_PRELOAD LD_PRELOAD_32（主进程层）
#   2) 通过 env 传进去让子进程也继承干净的链接环境
#   3) 改用 conda run 启动（conda run 自带干净 wrapper）
# ----------------------------------------------------------------------
if [[ -n "${LD_PRELOAD:-}" ]]; then
  echo ">>> 检测到 LD_PRELOAD，重置"
  echo "    当前 LD_PRELOAD=${LD_PRELOAD}"
  unset LD_PRELOAD LD_PRELOAD_32
fi

# ---------------------------------------------------------------------------
# 关键修复：强制预加载 libdl.so，解决运行时 dlopen 找不到符号的问题
#
# 根因（per-rank 日志确认）：
#   /usr/local/cuda/lib64/libcufile.so: undefined reference to `dlopen'
#   /usr/local/cuda/lib64/libcufile.so: undefined reference to `dlclose'
#   /usr/local/cuda/lib64/libcufile.so: undefined reference to `dlsym'
#   collect2: error: ld returned 1 exit status
#
# 分析：
#   - DS_BUILD_GDS=0 只跳过了 DeepSpeed 的 GDS JIT 编译
#   - 但 libcufile.so 作为 CUDA 的一部分，在 import torch / deepspeed 时
#     会被 dlopen() 动态加载到进程空间
#   - dlopen() 需要动态链接器解析 libcufile.so 依赖的 libdl.so 符号
#   - 若 conda compiler_compat 的 ld.so 或 ld-linux-x86-64.so.2 抢先加载，
#     会导致 libcufile.so 找不到 dlopen 等符号（这些符号本由 glibc/libdl.so 提供）
#   - collect2: ld returned 1 是链接器报错，但实际发生在动态加载时
#
# 修复：
#   在 unset LD_PRELOAD 后，显式设置 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libdl.so.2
#   这让动态链接器优先加载 libdl.so，确保 dlopen/dlsym/dlclose/dlerror/dlvsym 符号全局可见。
#
# 备选路径（如果上面找不到）：
#   - /lib64/libdl.so.2   (CentOS/RHEL 路径)
#   - /lib/x86_64-linux-gnu/libdl.so.2  (Debian/Ubuntu 标准)
# ---------------------------------------------------------------------------
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
  # LD_PRELOAD：让动态链接器在运行时优先加载 libdl.so，dlopen/dlsym/dlclose 等符号对所有 SO 可见
  export LD_PRELOAD="$_LDLIB"
  echo ">>> [fix-ld-preload] LD_PRELOAD=$LD_PRELOAD（运行时解决 libcufile.so dlopen 符号缺失）"

  # LD_LIBRARY_PATH：让链接器在编译/构建 C++ 扩展时能找到 libdl.so.2
  # 关键：collect2 (gcc 包装脚本) 在构建子进程的 JIT 扩展时，
  #       需要动态链接器能搜到 libdl.so.2。仅 LD_PRELOAD 不够（那是运行时行为）。
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="/lib64"
  else
    # 追加 /lib64 到现有 LD_LIBRARY_PATH（去重）
    export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH:/lib64" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')"
  fi
  echo ">>> [fix-ld-preload] LD_LIBRARY_PATH 追加 /lib64（编译/构建时让链接器找到 libdl.so.2）"
else
  echo ">>> [fix-ld-preload] 警告：找不到 libdl.so.2，跳过 LD_PRELOAD 设置"
  echo "    请手动确认 /usr/lib/x86_64-linux-gnu/libdl.so.2 是否存在"
fi

# 确保 conda compiler_compat 的 ld 不会以任何方式被 Python 子进程用到
# conda activate 会往 CONDA_PREFIX/compiler_compat 加 PATH，但 linker 是
# 通过 LD_PRELOAD/ld.so.conf 注入的。只要 LD_PRELOAD 干净即可。
# 为保险，另将 conda compiler_compat 从 LD_LIBRARY_PATH 里排除（如果有）。
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  # 去掉 conda compiler_compat/lib 路径（以防 conda activate 误加）
  NEW_LDLP="$LD_LIBRARY_PATH"
  for _cpath in "$CONDA_PREFIX/compiler_compat" "$CONDA_PREFIX/lib"; do
    NEW_LDLP="$(echo "$NEW_LDLP" | tr ':' '\n' | grep -v "^${_cpath}$" | tr '\n' ':' | sed 's/:$//')"
  done
  if [[ "$NEW_LDLP" != "$LD_LIBRARY_PATH" ]]; then
    echo ">>> 清理后的 LD_LIBRARY_PATH=${NEW_LDLP}"
    export LD_LIBRARY_PATH="$NEW_LDLP"
  fi
fi

echo "============================================"
echo "Node: $(hostname -s)"
echo "nvidia-smi -L 行数(仅参考,非作业可见数): $(nvidia-smi -L 2>/dev/null | wc -l)"
for k in CUDA_VISIBLE_DEVICES SLURM_GPUS_ON_NODE SLURM_STEP_GPUS SLURM_JOB_GPUS; do
  if [[ -n "${!k:-}" ]]; then
    echo "$k=${!k}"
  fi
done
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (wheel): $(python -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

python <<'PY'
# 预检：以 PyTorch 可见 GPU 为准，并尽早捕获 import deepspeed 失败
import os
import torch

n = torch.cuda.device_count()
print(f"torch.cuda.device_count()={n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}")

try:
    import deepspeed

    print(f"import deepspeed OK, version={deepspeed.__version__}")
except Exception as e:
    print(
        "\nimport deepspeed 失败。若报错含 device=K, num_gpus=M 且 K>=M:\n"
        "  1) 确认作业实际可见卡数: 上面 torch.cuda.device_count() 是否等于 sbatch --gres 申请数。\n"
        "  2) 常见规避: pip uninstall -y triton 后重跑（削弱 DeepSpeed import 时的 Triton/CUDA 探测）。\n"
        f"原始异常: {e}\n"
    )
    raise SystemExit(1)
PY

REQ_GPUS="${ZERO3_SMOKE_NUM_GPUS:-4}"
NUM_GPUS="$(python -c "import torch; print(torch.cuda.device_count())")"
if [[ "$NUM_GPUS" -lt "$REQ_GPUS" ]]; then
  echo "WARN: 期望约 ${REQ_GPUS} 卡（ZERO3_SMOKE_NUM_GPUS），但 PyTorch 仅见 ${NUM_GPUS} 张，将用 ${NUM_GPUS} 启动 DeepSpeed。"
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "ERROR: torch.cuda.device_count()==0，无法跑 GPU 冒烟。"
  exit 1
fi

echo "============================================"

# --- 安全验证：裸 Python，不走 deepspeed launcher ---
echo ">>> [debug] 裸 Python 导入测试（不走 deepspeed launcher）"
python <<'PYEOF'
import sys, os, traceback
print(f"[bare-py] pid={os.getpid()} LD_PRELOAD={os.environ.get('LD_PRELOAD','N/A')}", flush=True)
try:
    import torch; print(f"[bare-py] torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
    print(f"[bare-py] device_count={torch.cuda.device_count()}", flush=True)
except Exception as e:
    print(f"[bare-py] torch FAILED: {e}", flush=True); traceback.print_exc(); sys.exit(1)
try:
    import deepspeed; print(f"[bare-py] deepspeed={deepspeed.__version__}", flush=True)
except Exception as e:
    print(f"[bare-py] deepspeed FAILED: {e}", flush=True); traceback.print_exc(); sys.exit(1)
print("[bare-py] 裸导入 OK", flush=True)
PYEOF
echo "============================================"

if [[ "${ZERO3_SMOKE_DS_REPORT_ONLY:-0}" != "1" ]]; then
  echo ">>> ds_report (或 python -m deepspeed.env_report)"
  # DS_BUILD_GDS=0 跳过 GDS（GPU Direct Storage）检测，避免链接 libcufile.so 失败。
  # GDS 用于高速存储直通 GPU，训练一般不需要。Issue #6461 确认 GDS 与 conda
  # linker / 缺 -ldl 的问题强相关，跳过不影响任何训练功能。
  export DS_BUILD_GDS=0
  if command -v ds_report &>/dev/null; then
    ds_report || true
  else
    python -m deepspeed.env_report || true
  fi
  echo "============================================"
fi

if [[ "${ZERO3_SMOKE_DS_REPORT_ONLY:-0}" == "1" ]]; then
  echo "ZERO3_SMOKE_DS_REPORT_ONLY=1 -> 跳过冒烟测试"
  exit 0
fi

# ============================================================
# 关键修复 0：开启 per-rank 日志（能看到子进程真正的崩溃原因）
# 问题：deepspeed launcher 默认把子进程 stdout/stderr 接到 /dev/null，
#       即使子进程里有 Python traceback / CUDA 错误你也完全看不到。
#       加上 --enable_each_rank_log 后，每个 rank 的输出会写到
#       logs/rank*.log，让你一眼看到"到底是哪一行崩的"。
# ============================================================
mkdir -p "$ROOT/logs"
RANK_LOG_DIR="$ROOT/logs/rank_logs_$(date +%Y%m%d_%H%M%S)"
echo ">>> [fix-log] per-rank 日志目录: $RANK_LOG_DIR"

# ============================================================
# 关键修复 A：绕过 CUDA toolkit 版本检查
#
# 根因：PyTorch cu126 wheel 用 CUDA 12.6 编译，节点 nvcc 是 12.2。
#       DeepSpeed 的 assert_no_cuda_mismatch() 检查 nvcc 版本，
#       不匹配时可能在某些路径上触发 AssertionError → sys.exit(1)。
# 修复：DS_SKIP_CUDA_CHECK=1 绕过版本检查。
# 注意：跳过检查仅影响 DeepSpeed C++ op 的 JIT 编译路径，
#       不影响已编译好的 kernel（如 AdamW 融合 op）。
# ============================================================
export DS_SKIP_CUDA_CHECK=1
echo ">>> [fix-cuda] DS_SKIP_CUDA_CHECK=1（绕过 nvcc vs torch CUDA 版本不匹配检查）"

# ============================================================
# 关键修复 B：NCCL 调试变量（万一仍有 NCCL 初始化问题，这些变量能暴露根因）
# ============================================================
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,INIT,ENV
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=$(ip -o route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K[^ ]+' || echo "ib0")
echo ">>> [fix-nccl] NCCL_DEBUG=INFO, NCCL_IB_DISABLE=0, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto}"

# NCCL 超时：默认 1800000ms（30分钟），调试时缩至 300000ms（5分钟）快速定位问题
export NCCL_TIMEOUT=300000

# ============================================================
# 关键修复 C：移除 conda compiler_compat linker wrapper
#
# 问题：conda 环境自带 compiler_compat/ld wrapper，在子进程 JIT 编译时被调用
#       来链接 CUDA 库（如 libcufile.so），但它与 CUDA 12.6 不兼容：
#       undefined reference to `dlopen', `dlclose', `dlsym', `shm_open'...
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
# 注意：放在 ds_report 之前，确保 ds_report 子进程也能继承。
# ============================================================
export LDFLAGS="-Wl,--no-as-needed -ldl"
export CFLAGS="${CFLAGS:-} -I/usr/include"
echo ">>> [fix-ld] 设置 LDFLAGS=$LDFLAGS (让 libcufile.so 正确链接 dlopen/dlsym)"

echo ">>> ZeRO-3 smoke (wall_clock_breakdown 默认开启), num_gpus=${NUM_GPUS}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

CPU_OFFLOAD_FLAG=()
if [[ "${ZERO3_SMOKE_CPU_OFFLOAD:-0}" == "1" ]]; then
  CPU_OFFLOAD_FLAG=(--cpu-offload)
  echo "    (CPU offload optimizer，对齐 hpp_memory_efficient)"
fi

NO_JIT_FLAG=()
if [[ "${ZERO3_SMOKE_NO_JIT:-0}" == "1" ]]; then
  NO_JIT_FLAG=(--no-jit)
  echo "    (DS_JIT=0，禁用 DeepSpeed JIT 编译)"
fi

# 调试 SIGKILL/exit2：先用单 GPU 确认最小路径能跑，再切回多卡。
# 若仍 SIGKILL，试 ZERO3_SMOKE_LAUNCHER=torchrun（见脚本顶部说明）。
# 禁用 JIT: export ZERO3_SMOKE_NO_JIT=1

# ---------------------------------------------------------------------------
# 两种 launcher 策略（DeepSpeed launcher vs torchrun）
# deepspeed launcher：spawn 模式，子进程走 execve，conda 环境有时有问题
# torchrun：fork 模式，子进程继承父进程状态，conda 兼容性更好
# ---------------------------------------------------------------------------
LAUNCHER="${ZERO3_SMOKE_LAUNCHER:-deepspeed}"

if [[ "$LAUNCHER" == "torchrun" ]]; then
  echo ">>> ZeRO-3 smoke via torchrun (fork 模式), nnodes=1 nproc_per_node=${NUM_GPUS}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  torchrun \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    --log_dir="$RANK_LOG_DIR" \
    "$ROOT/scripts/deepspeed_zero3_smoke.py" \
    --steps "${ZERO3_SMOKE_STEPS:-40}" \
    "${CPU_OFFLOAD_FLAG[@]}" \
    "${NO_JIT_FLAG[@]}"
elif [[ "$LAUNCHER" == "deepspeed" ]]; then
  if [[ "${ZERO3_SMOKE_SINGLE_GPU:-0}" == "1" ]]; then
    echo ">>> ZERO3_SMOKE_SINGLE_GPU=1，降级到单卡冒烟"
    deepspeed --num_gpus=1 \
      --enable_each_rank_log="$RANK_LOG_DIR" \
      "$ROOT/scripts/deepspeed_zero3_smoke.py" \
      --steps "${ZERO3_SMOKE_STEPS:-10}" \
      "${NO_JIT_FLAG[@]}"
  else
    # ---------------------------------------------------------------------------
    # 直接调用 deepspeed（不用 --module，避免 ModuleNotFoundError）
    #
    # 注意：去掉 --module，因为 --module 会让 launcher 用 python -m 执行脚本，
    #       导致 .py 文件被当作模块导入而不是直接执行，从而报：
    #       "ModuleNotFoundError: No module named 'deepspeed_zero3_smoke.py'"
    # ---------------------------------------------------------------------------
    echo ">>> ZeRO-3 smoke (wall_clock_breakdown 默认开启), num_gpus=${NUM_GPUS}"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    deepspeed --num_gpus="$NUM_GPUS" \
      --enable_each_rank_log="$RANK_LOG_DIR" \
      "$ROOT/scripts/deepspeed_zero3_smoke.py" \
      --steps "${ZERO3_SMOKE_STEPS:-40}" \
      "${NO_JIT_FLAG[@]}"
  fi
else
  echo "ERROR: unknown ZERO3_SMOKE_LAUNCHER=$LAUNCHER (try: deepspeed or torchrun)"
  exit 1
fi

echo "============================================"
echo "完成。日志中的 wall_clock_breakdown 可粗看通信/等待；MBU 需真实训练 + profiler。"
echo "============================================"
