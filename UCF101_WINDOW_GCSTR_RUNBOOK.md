# UCF101-DVS window gCSTR pretraining

Original DailyDVS and whole-record UCF code/checkpoints remain unchanged.
Formal training has not been started. A separate real-data, two-GPU smoke
workflow is described below; it does not replace the full-dataset audit.

## Latest user overrides

- Official Train/Test only; no new Validation split. Split source:
  `UCF101-DVS/splits/{train,test}.csv`, with video and class-group disjointness checks.
- Test is evaluated each epoch and used for checkpoint selection. Reports call
  this **Test-selected**, not an independent held-out test result. Window overlap
  remains confined to its source video's split.
- Keep suspicious videos, including known single-polarity/constant-coordinate
  records. Do not repair timestamps; use the existing stable-sorted NPZ after
  checking exact equality with raw MAT. Diagnostic flags are recorded for
  reproducibility but do not exclude or block these records.
- Original protocol filters still apply: full candidate window count <40,
  each retained window >10000 events. They may filter known abnormal long videos.
- Missing/invalid polarity, invalid coordinates, inconsistent source fields or
  missing files still block: they cannot produce the requested representation.
- No early stopping; 50 complete epochs per branch.

## Fixed settings

| Setting | Value |
|---|---|
| Window / stride | 1,000,000 / 500,000 microseconds |
| Alignment / bounds | First event; [start,end); only full windows |
| Window count | Reject source videos with >=40 candidate full windows |
| Window events | >=10001, before any hypothetical point sampling |
| Images | All events in each window; no point sampling |
| Future point branch | 8192 recorded only, no point encoder invoked |
| Branch execution | cstr3 / xt_p / ty_p share each batch; three independent models/optimizers |
| Model | Three separate ImageNet1K_V1 ResNet-18; classifier=101 |
| Input | [3,224,224], 224 temporal bins, ImageNet normalization |
| Sensor | Read per-video metadata; validate raw coordinates |
| Optimizer | Adam, lr=1e-4, weight_decay=0, default betas=(0.9,0.999), eps=1e-8 |
| Schedule / stopping | None / none |
| Epochs / batch | 50 per branch / global 64, normally 32 per GPU |
| Parallelism | Two GPUs, DDP + SyncBatchNorm; all three branches resident on both GPUs |
| Loss | CrossEntropyLoss, label smoothing=0 |
| Augmentation / precision | Train-only event-space spatial / float32, AMP disabled |
| Seed / workers | 0 / 4 per GPU (8 total) |
| Coverage | Every valid training window once each epoch, shuffle, drop_last=False |
| Selection | Minimum Test video NLL; auxiliary maximum Test video Top1 |
| Voting | Average window softmax per video, equal weight per video |
| Ties | Loss→higher Top1→earlier epoch; Top1→lower loss→earlier epoch |
| Hard vote | Auxiliary only, lowest class ID wins ties |

The window-only representation reuses existing histogram/min-max/bilinear resize
helpers, without changing `gcstr_representations.py`. XT/TY use fixed window
boundaries. Both XY polarity timestamp means use that same origin/span, followed
by the existing per-map maximum normalization. Empty polarity maps are zero;
no polarity or synthetic events are inserted. This temporal normalization is an
explicit window adaptation, not bit-identical to the old whole-record renderer.

## New entry points (future use only)

Run from the project root in the project's `.venv-eventmamba` environment.

1. `python ucf101_window_data.py prepare`: audit all raw ZIP/MAT files against
   processed NPZ; preserve official video splits, create audit and window manifest.
   It requires a new output directory and exits nonzero when audit is blocked.
2. `python ucf101_window_data.py cache`: generate only unaugmented Test window
   images. Train images are generated online after event-space augmentation,
   never read from or saved to the fixed image cache. Test cache uses float32 NPY;
   budget `test_windows × 3 × 3 × 224 × 224 × 4` bytes plus filesystem overhead.
3. `python -m unittest test_ucf101_window_gcstr`: protocol unit tests.
4. `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 train_ucf101_window_joint.py --smoke`: explicit small
   Train-only forward/backward/optimizer/checkpoint/resume check for all branches,
   using up to 2*batch_size+1 windows to exercise a non-divisible final batch.
   Saves only under `smoke_joint/`, plus a configuration-bound `joint_smoke_passed.json`.
5. `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 train_ucf101_window_joint.py --train`: formal
   shared-batch independent training, refuses to start without a matching joint
   smoke marker. `--resume` restores all three models and their Adam states from
   `checkpoints/joint_last.pt` at the same completed epoch.

These are documentation commands, not a request to execute them now. No automatic
OOM batch reduction or automatic background launch. If batch/epochs/lr changes,
repeat Smoke Test with matching arguments; do not mix protocols in one directory.

## Outputs and metric conventions

All outputs default to `experiments/ucf101dvs_gcstr_sliding_v1/`:

- `dataset_audit.json`, `video_splits.csv`, `class_to_idx.json`, `window_manifest.csv`.
  Audit includes raw field/polarity verification, source hashes, rejected-video
  IDs/reasons, per-class and per-split coverage, event/window distributions.
- `cache/{view}/{cache_key}.npy`: source content, video/window boundaries,
  sensor/config/code version are hashed; variant is the directory component.
- `config.json`: machine-readable JSON rather than YAML (no extra dependency).
- `checkpoints/{view}/{best_loss,best_top1,last}.pt`: compatible `model_state`,
  `view`, `epoch`, `config`, plus optimizer state and selection history.
- `checkpoints/joint_last.pt`: authoritative atomic all-branch resume snapshot.
- `logs/{view}_metrics.jsonl`: Train/Test window CE/Top1/Top5, video soft-vote
  NLL/Top1/Top5 and auxiliary hard vote, coverage, time, allocated/reserved peak
  CUDA memory, LR, checkpoint paths and SHA-256. Accuracy fractions are in [0,1].
- `reports/{view}_test_video_predictions_{last,best_loss}.csv` and `summary.json`.

Training metrics aggregate predictions made while weights change throughout the
epoch (explicit `online_train_changing_weights` field); Test metrics use fixed
end-of-epoch weights. Window loss averages over windows; video NLL averages over
retained videos after mean-softmax voting. Videos without valid windows are
excluded from that denominator and enumerated with coverage in the audit.

Epoch-seeded random generators and persisted Adam state support epoch-boundary
resume. A process lock prevents simultaneous writers. Code/manifest/config
fingerprints reject incompatible resumes. Code hashes are recorded because this
environment does not currently provide a usable `git` executable.

## Two-GPU implementation

Global batch remains 64; LR remains 1e-4. Train batches are partitioned without
duplicating or dropping windows. An undersized final global tail borrows events
from the preceding batch (window indices only, no event resampling); this gives
both ranks a nonempty shard. Unequal shard sizes use globally weighted CE before
DDP gradient averaging. SyncBatchNorm synchronizes training statistics.
Evaluation is sharded without padding; per-video probability sums, counts and
hard votes are merged across ranks before metrics/selection. Only rank 0 writes
checkpoints and reports, and saved state keys have no `module.` prefix. Peak GPU
memory is logged separately for both ranks. Single-GPU CPU/unit-test helpers
remain available, but formal launch commands above explicitly use two GPUs.

## Approved spatial augmentation

The user subsequently approved `spatial` (supersedes the original `none`). Reuse
`gcstr_augmentation.augment_events`: coordinate dithering ±0.5 pixels, rotation
±15 degrees, independent crop removal 0–10% on each side, then spatial rescaling.
Apply to full raw events in the already-selected training window, before
projection. Preserve original window start and 1-second time span; no temporal
warp or polarity flip. The existing implementation retains pre-crop events if
a crop would be empty; this fallback is explicitly recorded in config.

One augmented version per window per epoch, not fivefold dataset expansion.
Randomness is keyed by seed, epoch, video and window, independent of GPU rank,
worker scheduling and view, so equivalent windows share a transform across
branches and epoch-boundary resumes reproduce it. Raw count >10000 is checked
before augmentation; cropping may leave fewer events, which are all projected.
No 8192-point sampling is performed. Test remains deterministic/unaugmented.

Online training projections add CPU/IO work compared with fixed image cache;
actual double-GPU throughput must be measured in the later Smoke Test. No extra
large event cache is created automatically. Code fingerprints include the shared
augmentation module; old smoke markers are not valid for this changed protocol.

## Isolated subset Smoke Test (2026-10-14)

Prepare only a Train subset using `python ucf101_window_data.py prepare
--smoke-per-class 1 --output experiments/ucf101dvs_gcstr_sliding_smoke_20261014`.
This checks one original MAT/NPZ video per class and produces a manifest marked
`audit_scope=smoke_train_subset`. The trainer explicitly refuses `--train` with
such an audit, even if its Smoke Test succeeds. Official Test payload is unused.

Launch using `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone
--nproc_per_node=2 train_ucf101_window_gcstr.py --smoke
--output experiments/ucf101dvs_gcstr_sliding_smoke_20261014`.
Each branch exercises 129 windows at global batch64, spatial augmentation, four
workers per rank, exact-coverage uneven tail handling, SyncBatchNorm, gradient
updates, checkpoint reload equivalence, Adam resume, and distributed video voting
on the same Train subset. This is an execution check, not a recognition benchmark.
Results are in `smoke_passed.json` and `smoke_console.log` in that directory.

## Shared-batch three-branch update (2026-10-14)

Current entry point: `train_ucf101_window_joint.py`. The previous sequential
entry remains available for comparison and provides shared DDP/metric helpers.
`ThreeViewDataset` reads one video/window and applies one spatial transform,
then generates all three projections. Each GPU normally sees 32 windows with
three images per window. Every branch sees global batch64, not batch192.
Within each batch the branches execute forward/backward/Adam one after another,
so they share data processing but not activations, parameters, losses, or
optimizers. All branches advance through each epoch together. This is not fusion
or joint fine-tuning, nor a claim that three GPU kernels execute concurrently.

The joint Smoke Test uses the existing audited Train subset, checks independent
classifier updates, exact checkpoint prediction restoration, all three Adam
states, uneven batches, and distributed video metrics against a direct
single-rank reference. Artifacts are `smoke_joint/`, `joint_smoke_console.log`
and `joint_smoke_passed.json` inside the isolated smoke directory. The original
sequential smoke artifacts remain intact. Full audit and Test-cache preparation
are still required before formal training.

## Authorized background pipeline

`run_ucf101_window_joint.py` runs full audit → Test cache → matching joint Smoke
Test → formal 50-epoch training, stopping on any nonzero stage exit. It is run
in a detached session; no automatic training restarts or parameter changes.
Status is `logs/ucf101_window_joint/status.json`; stage logs are `audit.log`,
`cache.log`, `smoke.log`, `train.log` in the same directory. The experiment
output is `experiments/ucf101dvs_gcstr_sliding_v1/`.

Monitor from the project root with `watch -n 5 cat logs/ucf101_window_joint/status.json`.
After training begins use `tail -F logs/ucf101_window_joint/train.log`.
