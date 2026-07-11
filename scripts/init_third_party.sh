#!/usr/bin/env bash
# 拉取 Point-SAM 同款子模块并安装 torkit3d / apex（可选）。
# 空目录原因：子模块未 init；需在本仓库根目录执行。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> git submodule update --init --recursive"
git submodule update --init --recursive third_party/torkit3d third_party/apex 2>/dev/null \
  || git submodule update --init --recursive torkit3d apex 2>/dev/null \
  || {
    echo "若仍失败，请检查 .gitmodules 中的 path 是否与本地目录一致。"
    git submodule update --init --recursive
  }

TORKIT=""
if [[ -f third_party/torkit3d/setup.py ]]; then
  TORKIT="third_party/torkit3d"
elif [[ -f torkit3d/setup.py ]]; then
  TORKIT="torkit3d"
fi

if [[ -n "$TORKIT" ]]; then
  echo "==> pip install torkit3d from $TORKIT (需要 CUDA 与编译环境)"
  # Point-SAM README: FORCE_CUDA=1 pip install third_party/torkit3d
  FORCE_CUDA="${FORCE_CUDA:-1}" pip install -v "$TORKIT"
else
  echo "WARN: 未找到 torkit3d/setup.py，请确认子模块已克隆到 third_party/torkit3d 或 torkit3d"
fi

APEX=""
if [[ -f third_party/apex/setup.py ]]; then
  APEX="third_party/apex"
elif [[ -f apex/setup.py ]]; then
  APEX="apex"
fi

if [[ -n "$APEX" ]]; then
  echo "==> 可选：安装 apex（FusedLayerNorm 等）。PyTorch 2.x + FSDP 下本项目默认不用 apex。"
  read -r -p "是否编译安装 apex? [y/N] " ans || true
  if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
    pip install -v --no-build-isolation \
      --config-settings "--build-option=--cpp_ext" \
      --config-settings "--build-option=--cuda_ext" \
      "$APEX"
  fi
else
  echo "WARN: 未找到 apex/setup.py，可跳过（训练不依赖 apex）。"
fi

echo "完成。"
