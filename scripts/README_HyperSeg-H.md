# HyperSeg-H engineering integration

The frozen report is implemented as an additional path in the existing repository.
No benchmark accuracy or SOTA claim is made by the synthetic tests.

## Shared implementation and compatibility

`decoder/models/PointFeatureEnhancer.py` and `CrossAttentionDecoder.py` remain the
canonical implementations. The old training module paths re-export the same classes;
all old constructors, forward calls, parameter names, scale FiLM and dropout remain.
The `official-compatibility` enhancer intentionally retains the public code's
non-batch-first attention. `batch-first-corrected` treats tensors as `[B,N,C]`;
use the same option for every compared baseline. Corrected attention has quadratic
point-count cost, so size the batch for 10,000 points rather than reusing batch 40.

Legacy config defaults retain `model_variant: legacy`, `control_signal: scale`,
`hierarchy_enabled: false` and the old training standardization. The new protocol
CLI always applies unit-sphere normalization. It must not be presented as a proven
replica of the public evaluator: the repository provides a demo/training path rather
than a complete benchmark evaluator, and its legacy normalization differs from the
frozen report. Verify official point sampling, target enumeration, aggregation,
normalization and boundary-prompt choices before publishing an A1 comparison.

The new gate decoder propagates the updated prompt between blocks. Prompt-to-point
attention adds `log(a)` before softmax; point-to-prompt output is multiplied by `a`
after projection/dropout, so gating still matters with a single prompt token.
The old ungated decoder preserves its original block behavior exactly.

## Geometry and control semantics

All geometry returns FP32 even inside autocast. Query radius is the **Euclidean
Poincare-ball radius**, with curvature `-c`; key radius is fixed beyond `rho_max`.
The constructor checks the cone annulus and ball boundary. Defaults: `c=1`,
`rho_min=.15`, `rho_max=.85`, key radius `.95`, cone K `.1`, beta initialized to `5`.
Gamma is positive, trainable and initialized to exactly `1`; beta is positive.

- `scale`: caller passes s = foreground count / N, used by the legacy FiLM baseline.
- `scale-proxy`: caller still passes s; the model explicitly computes g = 1 - s.
- `hierarchy`: caller passes true path-relative g = k/(K-1). This requires
  `hierarchy_enabled: true`; a singleton chain is rejected in A2/C.

The radial map is `rho_min + (rho_max-rho_min) g**gamma`; energy is
`relu(Xi-psi)` and attention gate is `exp(-beta*energy)`. Aperture uses
`asin(K*(1-r²)/r)` with r = sqrt(c)*||q||, following
[Ganea et al., ICML 2018](https://proceedings.mlr.press/v80/ganea18a.html).
Self-coincident embeddings receive angle zero. `g` is path-relative, not a global
semantic depth. Fixed-radius keys may reduce the mechanism to angular selection;
`spherical-hierarchy` provides a matched angular ablation, and `radius-film` uses
the radius as FiLM input without the cone gate. These are separately trained models.

## Cache and manifests

A manifest is a JSON list, with paths relative to its own directory:

```json
[{"model_id": "object-27", "cache_path": "object_0000000.pt"}]
```

Each safe tensor `.pt` cache has:

```python
{
  "model_id": "object-27",
  "points": float32_tensor_N_by_3,
  "features": float32_tensor_N_by_448,  # optional with a configured encoder
  "node_masks": {"root": uint8_binary_N, "part": uint8_binary_N, "leaf": uint8_binary_N},
  "chains": [{"node_ids": ["root", "part", "leaf"], "granularities": [0., .5, 1.]}]
}
```

`granularities` is optional, but validated if supplied. Nodes are stored once.
Hierarchy masks must be nested and nonempty; IDs are explicit and preserved.
The cache builder accepts aligned array paths (see `--help`); `--scale-labels`
converts trusted official `.npy` dictionaries with `coord` and `label` into one
singleton chain per target. It **does not infer a hierarchy from flat labels**.
Features must correspond to the same candidate points in the same order. Existing
features computed with another normalization should be regenerated consistently.
The script consumes already aligned annotations; downloading, remeshing and
translating raw PartNeXt Arrow/mesh annotations are outside this cache adapter.

```bash
python3 scripts/prepare_hyperseg_cache.py --manifest source_arrays.json --output cache
python3 scripts/prepare_hyperseg_cache.py --manifest official_arrays.json --scale-labels --output a1_cache
```

For A2/C, leaf quota samples `min(10, available)` points per unique chain leaf,
then fills uniformly without replacement when enough candidates exist. Quota
overflow raises rather than silently discarding tiny leaves. Sampling below N
uses replacement only for the remaining deficit. A1 uses uniform sampling.
Evaluation sampling is deterministic. Boundary prompts maximize the public
center/interior score using chunked nearest-background distances. A2/C chooses
one prompt inside the deepest surviving node and shares it across all ancestors.
Variable chains are padded with explicit validity masks; object-balanced sampling
supports non-contiguous/string object IDs.

## Run

Run commands from the repository root, or use absolute script paths.

```bash
# Exclude overlaps from ALL training sources against ALL held-out sources.
python3 scripts/check_id_overlap.py --manifest train.json \
  --test-manifest partobjaverse_test.json --test-manifest partnete_test.json --output runs/id_audit

# A1: the scale value is converted to 1-s inside the model.
python3 scripts/hyperseg_h.py train --track A1 \
  --config training/decoder_train/code/configs/hyperseg_h_a1.yaml \
  --manifest a1_train.json --test-manifest a1_test.json --output runs/a1 \
  --epochs 110 --num-points 10000 --device cuda --amp

# A2: true hierarchy, all levels trained with shared prompt and intermediate g.
python3 scripts/hyperseg_h.py train --track A2 \
  --config training/decoder_train/code/configs/hyperseg_h.yaml \
  --manifest hierarchy_train.json --test-manifest hierarchy_test.json --output runs/a2 \
  --epochs 110 --device cuda --amp

# --checkpoint official_decoder.pt initializes audited existing components.
# A full checkpoint restores model weights; this is a warm start, not optimizer resume.
# For S²AM3D-Scale comparison on A2: use --model-variant legacy --control-signal scale.
# For S²AM3D-Hierarchy: use --model-variant legacy --control-signal hierarchy.
# For other ablations: --model-variant radius-film or spherical-hierarchy.

python3 scripts/evaluate_hyperseg_protocol.py --track A2 \
  --checkpoint runs/a2/latest.pt --manifest hierarchy_test.json --output runs/a2_eval
python3 scripts/evaluate_hyperseg_protocol.py --track C \
  --checkpoint runs/a2/latest.pt --manifest hierarchy_test.json --output runs/c_eval --sweep-steps 21
python3 scripts/audit_checkpoint.py --checkpoint runs/a2/latest.pt --output runs/audit.json
python3 scripts/hyperseg_h.py export --track A2 --checkpoint runs/a2/latest.pt --output runs/hyperseg_h.pt

# Existing training entry point also dispatches to the new CLI:
python3 training/decoder_train/code/main.py --hyperseg train --track A2 \
  --config training/decoder_train/code/configs/hyperseg_h.yaml \
  --manifest hierarchy_train.json --test-manifest hierarchy_test.json --output runs/a2
```

Training requires held-out manifests; `excluded_overlap_ids.txt` and
`train_deoverlapped.json` are saved before training. Full checkpoints retain actual
training IDs and evaluation rejects overlaps. Official old checkpoints lack such
provenance: audit their training IDs separately. Exact object-ID equality is used;
source-specific aliases must be canonicalized upstream, without guessing from
filenames. Pass every training source in the merged manifest and every test source
via repeated `--test-manifest` flags.

A1 disables containment/sandwich terms. A2/C use BCE+Dice, all ordered valid-level
containment pairs, per-level positive/negative geometric means and adjacent-level
intermediate-g sandwich loss. Loss weights are configurable. Results report both
target-mean and object-mean IoU, per-level IoU, adjacent-pair containment violation
rates, whole-chain ancestor consistency, and diagnostic cone coverage. Track C
additionally saves continuous g sweeps with mask fraction, rho, psi and violations.
The threshold defaults to .7; any alternative must be fixed using validation only.
No test-distribution calibration is performed.

Training is currently single-device, with optional AMP and finite gradient checks.
The legacy DDP trainer remains available. The new path uses AdamW with fixed LR;
match epochs, optimizer/schedule, augmentation, features/backbone policy, batch size
and initialization explicitly across experimental comparisons. No undocumented
augmentation is applied by the hierarchy cache loader.

## Encoder and checkpoint limitations

Cached features are the tested, dependency-light path. `hyperseg_h/backbone.py`
is a lazy adapter for the existing official PVCNN + triplane extractor, using
points as the default encoder input features, as in the official demo. It requires
the original encoder dependencies/extensions/checkpoint. `backbone.freeze: true`
keeps the encoder frozen and in eval mode; `false` includes it in optimizer updates.
Component audits count **parameter numel** and reject coverage below 95% or shape
mismatches. Full HyperSeg-H checkpoints require every state entry, including the new
gate. Official component initialization reports new gate parameters as fresh.
The interactive demo accepts exported HyperSeg-H checkpoints and labels hierarchy g
explicitly; the interactive UI itself requires its original viewer dependencies.

## Verification

```bash
python3 scripts/test_hyperseg_components.py
python3 scripts/smoke_hyperseg_h.py
python3 -m py_compile hyperseg_h/*.py scripts/*.py decoder/models/*.py \
  decoder/interactive_demo.py training/decoder_train/code/*.py \
  training/decoder_train/code/models/*.py
```

Tests include Git-HEAD numerical legacy parity, shared class identity, batch
independence in corrected mode, endpoints/radius monotonicity, singular and boundary
geometry, FP32 autocast outputs, finite CPU/CUDA AMP gradients, M=1 gate behavior,
multilayer prompt updates, losses/validity, leaf quotas, ID guards, checkpoint
roundtrip/rejection, and synthetic A1/A2/C train/evaluate/export CLI runs.
Synthetic tests do not substitute for training on PartNeXt or validating benchmark
metrics with official datasets and pretrained checkpoints.
