# HPP-SAM

> **[中文版 README](README_zh.md)** | English README

---

## Project Vision

HPP-SAM explores **hierarchical granularity-aware interactive 3D segmentation** by combining SAM-style prompting with **hyperbolic geometry**.

### Core Hypothesis

> Interactive clicks naturally require multi-granularity (coarse-to-fine). 
> **Hyperbolic space** is ideal for encoding such hierarchical structure due to its exponential capacity near the boundary.

### Key Mechanism

We extract **hyperbolic radius** $r_i$ as a geometric proxy for segmentation granularity, then map it to attention sharpness via a temperature function $\tau(r_i)$.

---

## Architecture Overview

```
Point Cloud → Euclidean Encoder → Hyperbolic Prompt Branch (HRA)
                                                    ↓
                                        q_tilde ∈ Poincaré Ball
                                                    ↓
                                        r = distance(q_tilde, 0)
                                                    ↓
                                        τ(r) = sigmoid(a·r + b)
                                                    ↓
                         Geodesic Attention ⊙ τ(r) → Fine-grained Segmentation
```

### Mixed-Manifold Design

| Component | Space | Rationale |
|-----------|-------|-----------|
| Point Cloud Encoder | Euclidean | Natural for KNN/Voronoi grouping |
| Prompt Branch + HRA | Hyperbolic | Learnable radius-based granularity control |
| Cross Attention | Hyperbolic (logits) / Tangent (values) | Hybrid design |
| Mask Decoder | Euclidean | Output prediction |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Yunk-S/HPP.git
cd HPP
bash scripts/init_third_party.sh

# Train (DeepSpeed ZeRO-3)
deepspeed --num_gpus=8 train.py --config=hpp
```

---

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `hpp_sam/model/hyperpoint_sam.py` | Full model |
| `hpp_sam/model/hyper_prompt_branch.py` | HRA + radius + temperature |
| `hpp_sam/model/hyper_cross_attention.py` | Geodesic cross-attention |
| `hpp_sam/model/hyper_ops.py` | Poincaré ball operations |

---

## Mathematical Framework

**Exponential Map**: $q = \exp_0^c(\alpha \cdot p)$

**HRA (HyperET-style)**: $\tilde{q} = W \otimes_c q = \exp_0^c(W \log_0^c(q))$

**Radius**: $r = d_c(\tilde{q}, 0) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c}\|\tilde{q}\|)$

**Temperature**: $\tau(r) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \sigma(a \cdot r + b)$

**Geodesic Attention**: $\alpha_{ij} = \text{softmax}(-\tau(r_i) \cdot d_c(q_i, k_j))$

---

## ⚠️ Status

| Aspect | Status |
|---------|---------|
| Code runs | ✅ |
| Memory efficient | ❌ (needs optimization) |
| Training stable | ⚠️ |
| Numerical stable | ⚠️ |
| Novelty | ✅ (radius → temperature → granularity) |

**This is an experimental research codebase.** Memory optimization and stability fixes are ongoing. See internal `修复检查.md` for details (not public).

---

## Citation

Will add BibTeX after publication.
