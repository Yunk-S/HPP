# HPP-SAM

> **[English Version](README.md)** | 中文说明

---

## 项目愿景

HPP-SAM 通过将 **SAM 风格的可提示分割** 与 **双曲几何** 相结合，探索**层级粒度感知的交互式 3D 分割**。

### 核心假设

> 交互式点击天然需要多粒度（从粗到细）。
> **双曲空间** 由于其边界附近的指数容量，非常适合编码这种层级结构。

### 核心机制

我们将 **双曲半径** $r$ 作为分割粒度的几何代理，然后通过温度函数 $\tau(r)$ 将其映射为注意力锐度。

---

## 架构概览

```
点云 → 欧氏编码器 → 双曲提示分支 (HRA)
                                        ↓
                              q_tilde ∈ Poincaré Ball
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
# 克隆和安装
git clone https://github.com/Yunk-S/HPP.git
cd HPP
bash scripts/init_third_party.sh

# 训练 (DeepSpeed ZeRO-3)
deepspeed --num_gpus=8 train.py --config=hpp
```

---

## 关键文件

| 文件 | 描述 |
|------|------|
| `train.py` | 主训练脚本 |
| `hpp_sam/model/hyperpoint_sam.py` | 完整模型 |
| `hpp_sam/model/hyper_prompt_branch.py` | HRA + 半径 + 温度 |
| `hpp_sam/model/hyper_cross_attention.py` | 测地线交叉注意力 |
| `hpp_sam/model/hyper_ops.py` | Poincaré ball 运算 |

---

## 数学框架

**指数映射**: $q = \exp_0^c(\alpha \cdot p)$

**HRA (HyperET 风格)**: $\tilde{q} = W \otimes_c q = \exp_0^c(W \log_0^c(q))$

**半径**: $r = d_c(\tilde{q}, 0) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c}\|\tilde{q}\|)$

**温度**: $\tau(r) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \sigma(a \cdot r + b)$

**测地线注意力**: $\alpha_{ij} = \text{softmax}(-\tau(r_i) \cdot d_c(q_i, k_j))$

---

## ⚠️ 状态

| 方面 | 状态 |
|------|------|
| 代码可运行 | ✅ |
| 显存高效 | ❌ (需要优化) |
| 训练稳定 | ⚠️ |
| 创新性 | ✅ (半径 → 温度 → 粒度) |

**这是实验性研究代码。** 显存优化和稳定性修复正在进行中。

---

## 引用

发表后添加 BibTeX。
