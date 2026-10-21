# Frozen expert Gate and Memory experiments

This is an **additive, output-level experiment framework**. Fusion A, all original
Backbones, sampling/projection definitions, normalization, Cross-Attention,
checkpoints, training entries, audits and source-code hashes remain unchanged.
Nothing runs on import. Export, fitting and Test evaluation are separate opt-in
commands. No real-data export or new GPU experiment is part of this delivery.

## Files and configuration

| File | Interface |
|---|---|
| `export_fusion_evidence.py` | Strict trained-state restoration, `FrozenEvidence`, deterministic export, opt-in integration regression |
| `routing_data.py` | Versioned mmap evidence, integrity and role checks, 16 observations, window/video datasets, normalization |
| `fusion_router.py` | `Router(config)` with explicit input/output state and stable probability mixtures |
| `train_fusion_routing.py` | `fit` and explicit `eval`; generic class counts, checkpoint/log/metric output |
| `test_fusion_routing.py` | CPU regression, math, role/cache boundaries, causality and synthetic fit smoke |
| `configs/fusion_routing.json` | Shared small-module configuration |
| `configs/fusion_routing_roles.template.json` | Empty role-map template, not a newly generated split |

Default fit configuration: Adam, LR `1e-3`, weight decay `0`, 20 epochs,
8 videos/batch, hidden/state dimension 64, gradient clip 1, seed 0, no scheduler,
no early stopping, no Point auxiliary loss, 15 ECE bins. All three dynamic modes
use the same videos/order/batches, epochs, optimizer settings, observations,
normalization and static initialization. No Memory-only pretraining is performed.
All valid windows are visited once per epoch; optimization steps and parameter
counts are logged. Unrestricted static fitting starts at alpha=eta=0.5.

## Current protocol limitations and roles

The existing base experiment trained on the complete original `train` source
and used `test` video NLL for checkpoint selection. Its Test is therefore **not
an independent untouched holdout**. No new roles or data split are assigned by
this implementation. A usable Gate validation protocol remains a user decision.

Copy the empty role template to a new JSON file and provide approved records:

```json
{
  "schema": 1,
  "dataset_id": "UCF101-DVS",
  "videos": [
    {"source_split": "train", "video_id": "ACTUAL_TRAIN_VIDEO_ID", "role": "gate_train"},
    {"source_split": "train", "video_id": "ANOTHER_APPROVED_TRAIN_VIDEO_ID", "role": "gate_val"},
    {"source_split": "test", "video_id": "ACTUAL_TEST_VIDEO_ID", "role": "test"}
  ]
}
```

These placeholder IDs must be replaced with IDs from the audited
`window_manifest.csv`. The exporter rejects absent IDs, split spoofing, duplicate
roles, and mapping original Test to Gate training/validation. It exports **all
valid windows of selected videos**, or an explicitly requested small complete-
video subset. No windows are re-assigned across roles.

`source_split`, `role`, augmentation and sampling epoch are distinct. Original
Train videos remain marked **expert-fit-seen** even if assigned `gate_val`.
Original Test is marked expert-selection-seen. Provenance is attached to every
cache and evaluation. Its expert-fit exposure is conservatively unknown because
the official upstream Point initializer does not carry a verifiable per-video
training manifest. Generic external evidence can explicitly record unknown
expert exposure; it must not describe it as unseen.
The reader checks exposure against provenance: known fit/selection sources must
stay marked seen; an unseen claim requires an explicit
`verified_fit_unseen_source_splits` / `verified_selection_unseen_source_splits`
declaration. Merely changing a role name cannot establish unseen provenance.

Default fitting refuses validation that is expert-seen or unknown. The optional
`--allow-seen-expert-validation` permits **explicit exploratory** validation on
expert-seen/unknown non-Test videos, records that limitation, and never allows
original Test fitting/selection. It does not make those videos unseen. Do not
enable it without deciding and documenting the protocol. Without approved roles,
use only the CPU/synthetic tests below; no real Gate fit should be started.

## Evidence restoration and identity

Supply a complete trained `best_loss.pt`, `epoch_040.pt` or `last.pt`, not the
official initialization checkpoint. The exporter checks every original source
hash from its config, the existing audit, window manifest and video split hash;
then constructs the same architecture and restores **every tensor strictly**.
Only known `module.` / `fusion.` wrapper prefixes are removed. The original
initializer loader is not mistaken for trained-state restoration. Original source
weight hashes are retained for provenance; full trained state is authoritative.

One unchanged original forward computes Point features/FPS and Fusion logits.
Temporary layer4 hooks also pass each original image map through its original
ResNet avgpool/fc. No token adapter, position encoding or new classification head
is substituted. All expert parameters and BN are frozen/eval; hooks are removed
even on an exception.

Default export: no augmentation, explicit sampling epoch 0, per-window NumPy
sampling seed 0, 2048 chronological points with official normalization and
full-event projections; FPS uses original code, process seed+100000. Real source
split is retained. The existing Test frame cache may be read, never rewritten.
Record batch size, workers, device, world size 1 and software versions. Changing
batch shape, GPU count, input order or CUDA random flow is **not** a promise of
bitwise agreement with historical DDP metrics. Schema 1 export is single-process;
the five heads are still computed together, not in five repeated backbone runs.

Each completed cache has:

```text
manifest.json   schema/order/class mapping, trained checkpoint/config hashes,
                original + export code hashes, data/roles identities, provenance,
                normalization, sampling/FPS/export settings, per-file hashes
rows.jsonl      dataset_id, video_id, window_id, source_split, role,
                start_us, end_us, label_id, expert exposure
logits.npy      float32 [N,5,C], order: fusion,point,cstr3,xt_p,ty_p
stats.npy       float32 [N,4], event statistics in the order below
```

Arrays are mmap-readable, not raw events or large backbone features. Expected
window keys, duplicates, coverage, class ordering, finite values and hashes are
checked. Writes use a unique `.pending-*` directory; only a complete manifest and
atomic directory rename publish the cache. A failed export may leave that pending
directory for diagnosis; it cannot be trained on. Existing caches/outputs are
never overwritten. Cross-cache fitting requires identical expert/data/class/
sampling/roles identity. `--expect-identity expected_identity.json` optionally
pins a fully expected identity rather than just reading a self-consistent cache.

At 101 classes, logits take `2020*N` bytes: approximately 53 MiB for all 27384
Test windows and 208 MiB for 108066 Train windows, plus statistics/metadata.
Use new cache paths, preferably the approved datasets cache area to avoid the
remote-home quota. These commands do not automatically create caches there.

## Gate observations and sequences

`routing_data.OBSERVATIONS` defines the exact 16-element order:

1. For each of five heads in cache order: entropy divided by log(C), then
   Top1-minus-Top2 probability margin (10 elements).
2. Jensen-Shannon divergence between arithmetic-mean Frame probability and
   Point probability, divided by log(2).
3. Fraction of three image-head pairs with the same Top1 prediction (0, 1/3, 1).
4. `log1p(raw_event_count / (sensor_pixels * window_seconds))`.
5. `log1p(raw_event_count)`.
6. Occupied native sensor pixels / all sensor pixels.
7. Point samples / raw event count; values above one indicate replacement.

No labels, true-class probabilities, losses, video/group/fold IDs, complete
semantic vectors or future statistics enter Gate. Labels are used only for loss
and metrics. Normalizer mean/population std are fitted only on `gate_train`;
std is floored at 1e-6. All dynamic variants load the same static normalizer.

Sequence batches sort by real `start_us`. Videos may be shuffled; their windows
are not. `dt` is the true start gap in seconds, with first gap 0. Overlap is
`max(previous_end-current_start,0)/min(previous_duration,current_duration)` and
is returned as metadata, not an extra mode-specific Gate input. Gaps are not
assumed to be 0.5 seconds. State is reset for each video, passed explicitly across
chunks if required, and never updated by padding. No persistent video dictionary
is used. First implementation backpropagates through each complete video.

## Modes and math

All probabilities use temperature 1. Frame is the arithmetic average of three
image probabilities; `pB=alpha*pFrame+(1-alpha)*pPoint`, and
`pFinal=(1-eta)*pFusion+eta*pB`. Stable log-space arithmetic mixtures are used for
training NLL, never average logits or probability products.

| Mode | Interface and behavior |
|---|---|
| `fixed` | `eval --fixed fusion/frame/frame_point/fusion_frame/three_way` |
| `static` | Fit global alpha and eta; set `fusion_frame_only=true` in a separate config to fix alpha=1 |
| `current` | Current-observation 16→64→1 tanh MLP correction |
| `ema` | `eval --checkpoint CURRENT.pt --ema-decay 0.9`; no extra fit/search |
| `gru` | Same current path plus 64-state GRUCell/readout |
| `conditional` | Same current path plus conditioned old-state read and vector write gate |

Fixed `(alpha,eta)`: Fusion `(0.5,0)`, Frame `(1,1)`, Frame+Point `(0.5,1)`,
Fusion+Frame `(1,0.5)`, **three equal paths `(0.5,2/3)`**.

Dynamic variants require the same unrestricted static best-validation checkpoint.
They freeze its b0=logit(alpha0) and eta, and zero-initialize correction readouts,
so all begin at the same static prediction. They cannot import a static fit from
different caches/roles. Static endpoints are clamped to [1e-6,1-1e-6] at logit
initialization. An explicitly exact dynamic alpha endpoint stays constant and is
logged as degenerate; eta<0.01 is also flagged as low dynamic influence.

Conditional state equations: read `f(obs)+r(obs,old_state)` for current alpha;
then `u=sigmoid(MLP(obs,old_state,log1p(dt)))` and
`new_state=(1-u)*old_state+u*tanh(Linear(obs))`. State never directly predicts
classes. GRU and conditional models have comparable parameter counts at dimension
64. Both observe exactly the same cache-derived vector; time-gap conditioning is
explicitly part of the conditional write, not a future feature. EMA starts at
alpha0 and applies `state=decay*old_state+(1-decay)*current_alpha` per valid
window; its decay is a predeclared evaluation setting, never selected on Test.

Training minimizes effective-window mean NLL. To preserve window weighting even
when video batches contain different numbers of windows, each batch gradient is
scaled as `sum_window_NLL * epoch_steps / epoch_windows`; averaging these batch
objectives at fixed weights yields exactly epoch window-mean NLL. No padding or
Point auxiliary constant enters the objective. Logs report unscaled window NLL.

Evaluation mixes each window first, then averages its video's valid probabilities.
The new metric API accepts only normalized log probabilities and supports any
C>=2 with Top-k=min(5,C). It reports window NLL/Top1/Top5, video NLL/Top1/Top5,
video Brier=`mean_video sum_class(p-y)^2` (not divided by C), and video ECE with
15 equal-width Top1-confidence bins (confidence=1 in final bin). Rescue/damage/
net change are relative to SAME-cache Fusion A, at video level. Alpha/eta mean,
std, min/max, p05/median/p95, counts, steps and parameter count are saved.

`fit` selects `best_val_nll.pt` only on explicit `gate_val` video NLL and saves
`last.pt`, optimizer/RNG/config/cache identity, metrics JSONL and validation
predictions. This first batch does not add a resume command. `eval` is a separate
explicit operation and never fits normalization, temperature or EMA. Every run
refuses a nonempty output directory. The original 101-class evaluator is untouched.

## Commands for later use

Run from the project root. The following real-data commands are **not executed**
as part of implementation. `roles.approved.json` must first be populated and the
Gate validation limitation resolved. Cache/output names are new, independent paths.

CPU tests (these use temporary synthetic data and tiny CPU fits):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_fusion_routing -v
```

Opt-in original-interface regression with real trained weights, one batch only:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-eventmamba/bin/python export_fusion_evidence.py \
  --checkpoint experiments/ucf101_filename_v2_fusion_A_seed0_v1/checkpoints/epoch_040.pt \
  --roles roles.approved.json --source-split test --max-videos 1 \
  --batch-size 8 --device cuda:0 --integration-regression
```

Small complete-video export, then optional separate full source exports (omit
`--max-videos` only after approval). The same roles JSON, checkpoint, seeds and
export configuration must be used across source caches:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-eventmamba/bin/python export_fusion_evidence.py \
  --checkpoint experiments/ucf101_filename_v2_fusion_A_seed0_v1/checkpoints/epoch_040.pt \
  --roles roles.approved.json --source-split test --max-videos 2 \
  --output experiments/routing_evidence_smoke_test_v1 --batch-size 8 --device cuda:0
```

When approved, an explicit Train-source export (no automatic split, no augmentation)
is the following. The Test-source counterpart uses `--source-split test` and a
different output `experiments/routing_evidence_approved_test_v1`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-eventmamba/bin/python export_fusion_evidence.py \
  --checkpoint experiments/ucf101_filename_v2_fusion_A_seed0_v1/checkpoints/epoch_040.pt \
  --roles roles.approved.json --source-split train \
  --output experiments/routing_evidence_approved_train_v1 --batch-size 8 --device cuda:0
```

Baseline evaluation example; repeat the explicit command for the five fixed
modes without changing evidence. Using `--role test` is explicit Test reporting,
not an allowed way to choose hyperparameters:

```bash
CUDA_VISIBLE_DEVICES='' .venv-eventmamba/bin/python train_fusion_routing.py eval \
  --cache experiments/routing_evidence_smoke_test_v1 --role test \
  --fixed three_way --output experiments/routing_smoke_three_way_v1
```

After role approval and a new Train-source cache containing both approved
Gate roles, the static fit and common-initialization comparisons are:

```bash
CUDA_VISIBLE_DEVICES='' .venv-eventmamba/bin/python train_fusion_routing.py fit \
  --cache experiments/routing_evidence_approved_train_v1 \
  --config configs/fusion_routing.json --mode static \
  --train-role gate_train --val-role gate_val --output experiments/routing_static_v1

for mode in current gru conditional; do
  CUDA_VISIBLE_DEVICES='' .venv-eventmamba/bin/python train_fusion_routing.py fit \
    --cache experiments/routing_evidence_approved_train_v1 \
    --config configs/fusion_routing.json --mode "$mode" \
    --static-init experiments/routing_static_v1/best_val_nll.pt \
    --train-role gate_train --val-role gate_val --output "experiments/routing_${mode}_v1"
done
```

These commands intentionally **fail by default** if Gate validation videos are
expert-seen/unknown. If that exploratory protocol is explicitly approved, append
`--allow-seen-expert-validation` consistently to static and all dynamic fits.
It cannot bypass the original-Test ban. For an eligible separate validation
cache add `--val-cache ...`, with identical expert/cache identity.

Explicit Test evaluation and predeclared EMA example after all choices are fixed:

```bash
CUDA_VISIBLE_DEVICES='' .venv-eventmamba/bin/python train_fusion_routing.py eval \
  --cache experiments/routing_evidence_approved_test_v1 --role test \
  --checkpoint experiments/routing_current_v1/best_val_nll.pt \
  --output experiments/routing_current_test_v1

CUDA_VISIBLE_DEVICES='' .venv-eventmamba/bin/python train_fusion_routing.py eval \
  --cache experiments/routing_evidence_approved_test_v1 --role test \
  --checkpoint experiments/routing_current_v1/best_val_nll.pt --ema-decay 0.9 \
  --output experiments/routing_current_ema_test_v1
```

For restricted Fusion+Frame static control, copy the shared JSON to a new config
and set `fusion_frame_only` true; use a separate output and `--mode static`.
Do not initialize the unrestricted dynamic alpha comparisons from that restricted
control. No temperature/EMA/architecture search, expert fine-tuning or second
dataset training is included in this first batch.
