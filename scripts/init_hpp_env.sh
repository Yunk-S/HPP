#!/bin/bash
# =============================================================================
# HPP-SAM 环境初始化（集群 Slurm）
#
# 默认行为（登录节点执行）:
#   bash scripts/init_hpp_env.sh
#   → 自动 sbatch 提交「长时间 + 1 卡」作业，在计算节点完成下载/编译（避免 srun 短时限制）。
#
# 已在交互式长作业内（salloc/srun 已给足 --time）:
#   bash scripts/init_hpp_env.sh --local
#
# 仅被 sbatch 调用（勿手跑）:
#   bash scripts/init_hpp_env.sh --inner
#
# 可调环境变量（提交前 export）:
#   INIT_HPP_PARTITION   默认 gpua8001t
#   INIT_HPP_GRES        默认 gpu:a800:1
#   INIT_HPP_TIME        默认 12:00:00（大 wheel 下载慢可调 24:00:00）
#   INIT_HPP_MEM         默认 96G
#   INIT_HPP_CPUS        默认 16
#   INIT_HPP_CUDA_MODULE 默认 cuda/12.8.0-none-none-xmhtcei（仅 nvcc/编译 torkit3d；PyTorch 仍为 cu126 wheel）
#   INIT_HPP_ANACONDA    默认 anaconda3/2023.09-0-none-none-3te2njg
#   INIT_HPP_CONDA_ENV   默认 hpp
#   INIT_HPP_GCC_MODULE  可选：如 gcc/11.x（若登录/计算节点默认 g++ 过新，nvcc 会在 host_config.h 里 #error）
#   INIT_HPP_MAX_JOBS    默认 4（ninja 并行度，防编译占满内存）
#   TORCH_NVCC_FLAGS     默认含 -allow-unsupported-compiler（绕过「unsupported GNU version」类 #error）
#
# 源策略:
#   - torch / torchvision / torchaudio: 仅官方 cu126（清华无可靠 cu126 三件套，避免 cu130 混入）
#   - 其余与 pyproject 对齐的包: 清华为主 + pypi.org 兜底
#   - pip install -e . 使用 --no-deps，防止镜像侧把 torch 升级成错误 CUDA 构建
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 参数 ----------
DO_INNER=0
DO_LOCAL=0
for a in "$@"; do
  case "$a" in
    --inner) DO_INNER=1 ;;
    --local) DO_LOCAL=1 ;;
  esac
done

# ---------- 自动提交长时间 sbatch（非 Slurm 内且非 --local）----------
if [[ "$DO_INNER" -eq 0 ]] && [[ "$DO_LOCAL" -eq 0 ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$ROOT/logs"
  PARTITION="${INIT_HPP_PARTITION:-gpua8001t}"
  GRES="${INIT_HPP_GRES:-gpu:a800:1}"
  TMEM="${INIT_HPP_TIME:-12:00:00}"
  MEM="${INIT_HPP_MEM:-96G}"
  CPUS="${INIT_HPP_CPUS:-16}"
  echo "============================================"
  echo "当前不在 Slurm 作业内：将提交长时间安装作业（避免登录节点/srun 短时限制）"
  echo "  partition=$PARTITION  gres=$GRES  time=$TMEM  mem=$MEM  cpus=$CPUS"
  echo "  日志: $ROOT/logs/hpp_env_init-<jobid>.out / .err"
  echo "============================================"
  JOB_ID="$(sbatch --parsable \
    --job-name=hpp_env_init \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TMEM" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --output="$ROOT/logs/hpp_env_init-%j.out" \
    --error="$ROOT/logs/hpp_env_init-%j.err" \
    --export=ALL \
    --wrap="bash \"$ROOT/scripts/init_hpp_env.sh\" --inner")"
  echo "已提交作业: $JOB_ID"
  echo "跟踪日志: tail -f $ROOT/logs/hpp_env_init-${JOB_ID}.out"
  exit 0
fi

# ---------- 以下为计算节点 / 长会话内实际安装 ----------
if [[ "$DO_INNER" -eq 1 ]]; then
  echo "[init_hpp_env] --inner 在 Slurm 计算节点执行安装"
fi

CUDA_MOD="${INIT_HPP_CUDA_MODULE:-cuda/12.8.0-none-none-xmhtcei}"
ANACONDA_MOD="${INIT_HPP_ANACONDA:-anaconda3/2023.09-0-none-none-3te2njg}"
CONDA_ENV="${INIT_HPP_CONDA_ENV:-hpp}"

module purge
module load "$CUDA_MOD"
if [[ -n "${INIT_HPP_GCC_MODULE:-}" ]]; then
  echo "Loading GCC module: $INIT_HPP_GCC_MODULE"
  module load "$INIT_HPP_GCC_MODULE"
fi
module load "$ANACONDA_MOD"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$ROOT"

# ============================================================
# 关键修复：设置 LDFLAGS + 移除 conda linker wrapper
#
# 根因（DeepSpeed 官方 Issue #6461/#6593）：
#   - libcufile.so（CUDA GPU Direct Storage 库）依赖 libdl.so 提供的
#     dlopen/dlsym/dlclose/dlerror/dlvsym 符号
#   - conda linker wrapper 和部分系统 linker 调用链没有自动加入 -ldl
#   - 链接时出现 "undefined reference to `dlopen'" 等错误，collect2 返回 1
#
# 修复：
#   A. export LDFLAGS="-Wl,--no-as-needed -ldl"  ← 官方确认有效
#   B. 移除 conda compiler_compat/ld wrapper（防止 conda Python 注入错误 linker）
#
# 重要：torkit3d 编译（Step 5）本身不需要 LDFLAGS，但环境装好后若后续
#       有 DeepSpeed C++ extension JIT 编译需求（如 gds op），此设置可预防报错。
# ============================================================
export LDFLAGS="${LDFLAGS:-} -Wl,--no-as-needed -ldl"
export CFLAGS="${CFLAGS:-} -I/usr/include"
echo ">>> [fix-ld] LDFLAGS=$LDFLAGS"

# 移除 conda linker wrapper（Step 5 之后任何 subprocess 都不应被 conda linker 干扰）
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CONDA_COMPILER_COMPAT_LD="${CONDA_PREFIX}/compiler_compat/ld"
  if [[ -f "$CONDA_COMPILER_COMPAT_LD" ]]; then
    echo ">>> [fix-ld] 移除 conda compiler_compat/ld linker wrapper"
    cp "$CONDA_COMPILER_COMPAT_LD" "${CONDA_COMPILER_COMPAT_LD}.bak" 2>/dev/null || true
    rm -f "$CONDA_COMPILER_COMPAT_LD"
    echo ">>> [fix-ld] 已移除 -> ${CONDA_COMPILER_COMPAT_LD}.bak"
  fi
fi

# 官方 PyTorch CUDA 12.6 wheel（勿加清华为 index-url，避免拉到错误 torch）
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
# 清华 PyPI + 官方兜底（无 torch GPU wheel，不会覆盖已装 torch）
TUNA_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_OFFICIAL="https://pypi.org/simple"

echo "============================================"
echo "环境信息"
echo "  ROOT:          $ROOT"
echo "  Python:        $(which python)"
echo "  nvcc:          $(command -v nvcc || echo 'missing')"
echo "  SLURM_JOB_ID:  ${SLURM_JOB_ID:-'(无)'}"
echo "============================================"

echo "============================================"
echo "Step 0: setuptools（减轻 torch cpp_extension pkg_resources 告警）"
echo "============================================"
python -m pip install "setuptools>=70,<81" \
  -i "$TUNA_INDEX" \
  --extra-index-url "$PYPI_OFFICIAL" \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir

echo "============================================"
echo "Step 1: PyTorch 2.11 + cu126（仅官方索引，两步安装避免 vision 0.21 绑定 torch 2.6）"
echo "============================================"
python -m pip cache purge
python -m pip install "torch==2.11.0" \
  --index-url "$PYTORCH_INDEX" \
  --no-cache-dir
python -m pip install torchvision torchaudio \
  --index-url "$PYTORCH_INDEX" \
  --no-cache-dir

python -c "
import torch
assert 'cu126' in torch.__version__, f'ERROR: {torch.__version__}'
assert torch.version.cuda.startswith('12'), f'ERROR: cuda {torch.version.cuda}'
assert hasattr(torch.amp, 'custom_fwd'), 'ERROR: 需要 torch.amp.custom_fwd（DeepSpeed）'
print('PASS torch', torch.__version__, 'cuda', torch.version.cuda)
"

echo "============================================"
echo "Step 2: 训练依赖（对齐 pyproject.toml，不含 torch/triton 重复指定）"
echo "============================================"
# 与 pyproject [project.dependencies] 一致；torch/torchvision 已装好；triton 由 torch wheel 带入，勿再从镜像单独 pin
python -m pip install \
  "timm>=1.0.3" \
  "hydra-core>=1.3.0" \
  "omegaconf>=2.3.0" \
  "accelerate>=0.25.0" \
  "datasets>=2.14.0" \
  "numpy>=1.24.0" \
  "safetensors>=0.4.0" \
  "scipy>=1.10.0" \
  "wandb>=0.15.0" \
  "einops>=0.7.0" \
  "deepspeed>=0.14.0" \
  "ninja" \
  -i "$TUNA_INDEX" \
  --extra-index-url "$PYPI_OFFICIAL" \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir

python -c "
import torch, deepspeed
assert hasattr(torch.amp, 'custom_fwd')
assert 'cu126' in torch.__version__
print('PASS deepspeed', deepspeed.__version__, '| torch', torch.__version__)
"

echo "============================================"
echo "Step 3: 安装本项目（--no-deps，防止镜像把 torch 升级成 cu130 等）"
echo "============================================"
python -m pip install -e . --no-deps \
  -i "$TUNA_INDEX" \
  --extra-index-url "$PYPI_OFFICIAL" \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir

echo "============================================"
echo "Step 4: 拉取子模块（git submodule 空目录时直接 clone 兜底）"
echo "============================================"
# git submodule 在 GPFS 上偶发静默失败：用 --force + 兜底 clone 保证内容存在
if [[ ! -s "third_party/torkit3d/setup.py" ]] && [[ ! -s "third_party/torkit3d/pyproject.toml" ]]; then
  echo "torkit3d 子模块目录为空，尝试直接 clone..."
  rm -rf third_party/torkit3d
  git clone --recursive https://github.com/Jiayuan-Gu/torkit3d.git third_party/torkit3d
fi
if [[ ! -s "third_party/apex/setup.py" ]] && [[ ! -f "third_party/apex/setup.py" ]]; then
  echo "apex 子模块目录为空，尝试直接 clone..."
  rm -rf third_party/apex
  git clone --recursive https://github.com/NVIDIA/apex.git third_party/apex
fi
# 确认 submodule 注册
git submodule update --init --recursive third_party/torkit3d third_party/apex || true

echo "============================================"
echo "Step 5: 编译 torkit3d（详细报错模式）"
echo "============================================"
# 预检：nvcc 版本、CUDA_HOME、编译器（g++ 过新时 nvcc 会在 CUDA 头文件里 #error，日志常只剩 “#error \\” 多行）
if command -v nvcc &>/dev/null; then
  echo "  nvcc: $(nvcc --version | grep 'release' | head -1)"
else
  echo "WARN: nvcc 不在 PATH（module load cuda/12.8 了吗？）"
fi
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc &>/dev/null; then
  _NVCC_BIN="$(command -v nvcc)"
  export CUDA_HOME="$(dirname "$(dirname "$_NVCC_BIN")")"
  echo "  已推断 CUDA_HOME=$CUDA_HOME（原未设置）"
fi
echo "  CUDA_HOME: ${CUDA_HOME:-未设置}"
echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
echo "  CXX: $CXX"
command -v g++ &>/dev/null && echo "  g++: $(g++ --version | head -1)" || true
echo "  Python: $(which python)"

# 找 CUDA 自带 gcc（版本 >= 9，满足 PyTorch 2.11）
# 集群 cuda/12.8 module 自带 GCC 11.x，用它替代系统默认 gcc（系统 gcc 可能 < 9）
_CUDA_GCC=""
for _p in \
    "/gpfs/spack/opt/spack/linux-icelake/cuda-12.8.0-xmhtcei6cboqmg6jfdvilib6dsjyzoko/bin/gcc" \
    "${CUDA_HOME:-}/bin/gcc"; do
  if [[ -x "$_p" ]]; then
    _CUDA_GCC="$_p"
    break
  fi
done

if [[ -n "$_CUDA_GCC" ]]; then
  echo "  CUDA GCC: $($_CUDA_GCC --version | head -1)  (用于 nvcc -ccbin)"
  export TORCH_NVCC_FLAGS="-ccbin=$_CUDA_GCC"
elif command -v gcc &>/dev/null; then
  _SYS_GCC_VER="$(gcc -dumpversion 2>/dev/null || echo 0)"
  echo "  系统 gcc: $(gcc --version | head -1) (ver=$_SYS_GCC_VER)"
  if [[ "$((_SYS_GCC_VER))" -lt 9 ]]; then
    echo "  WARN: 系统 gcc < 9，PyTorch 2.11 头文件会 #error。"
    echo "  请确保 CUDA 自带 gcc 可用，或 find /gpfs/spack -name gcc -type f | head"
  fi
fi
export MAX_JOBS="${INIT_HPP_MAX_JOBS:-4}"

# 彻底卸载旧 torkit3d（避免旧 .so 与新 torch ABI 不兼容）
python -m pip uninstall -y torkit3d 2>/dev/null || true
rm -rf third_party/torkit3d/build third_party/torkit3d/*.egg-info

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export VERBOSE=1
python -m pip install --no-build-isolation ./third_party/torkit3d 2>&1 | tee "/tmp/torkit3d_build_$$.log"
PIP_RC=${PIPESTATUS[0]}
if [[ "$PIP_RC" -ne 0 ]]; then
  echo ""
  echo "==== torkit3d 编译失败（exit $PIP_RC）===="
  echo "关键报错片段："
  grep -E "error:|Error:|fatal:|unsupported|#error GCC" /tmp/torkit3d_build_$$.log | head -60
  echo "完整日志: /tmp/torkit3d_build_$$.log"
  exit 1
fi

python -c "from torkit3d.ops.sample_farthest_points import sample_farthest_points; print('torkit3d OK')"

echo "============================================"
echo "Step 6: 整体验证"
echo "============================================"
python -c "
import torch, deepspeed, timm, hpp_sam
from torkit3d.ops.sample_farthest_points import sample_farthest_points
assert hasattr(torch.amp, 'custom_fwd')
assert 'cu126' in torch.__version__
print('torch ', torch.__version__, 'cuda', torch.version.cuda)
print('deepspeed', deepspeed.__version__)
print('CUDA avail', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0))
print('ALL OK')
"

echo "============================================"
echo "完成。训练请: conda activate $CONDA_ENV && sbatch train_hpp_sam.sh"
echo "============================================"
