# HPP-SAM

> **[English](README.md)** | 中文说明

---

## 项目愿景

HPP-SAM 结合 **双曲几何** 与 SAM 风格可提示分割，探索**层级粒度感知的交互式 3D 分割**。

### 核心假设

> 双曲空间的指数容量天然适合编码粗→细的层级分割结构。

### 核心机制

提取 **双曲半径** $r$ 作为粒度的几何代理，通过 $\tau(r)$ 调节注意力锐度。

---

## 架构

```
点云 → 欧氏编码器 → 双曲提示分支 (HRA)
                            ↓
                      r = distance(q_tilde, 0)
                            ↓
                      τ(r) = sigmoid(a·r + b)
                            ↓
          测地线注意力 ⊙ τ(r) → 细粒度分割
```

---

## 快速开始

```bash
git clone https://github.com/Yunk-S/HPP.git
cd HPP
bash scripts/init_third_party.sh
deepspeed --num_gpus=8 train.py --config=hpp
```

---

## ⚠️ 状态

| 方面 | 状态 |
|------|------|
| 代码可运行 | ✅ |
| 显存高效 | ❌ (需优化) |
| 训练稳定 | ⚠️ |
| 创新性 | ✅ (半径→温度→粒度) |

**实验性研究代码。** 显存优化和稳定性修复进行中。

---

## 引用

发表后添加 BibTeX。
