# EventMamba V2-only / DailyDVS-200 (2026-10-11)

## Scope and provenance

This is an adaptation of the official V2 method to **DailyDVS-200**, NOT a
reproduction of the 12-class Daily DVS result and NOT the project's V1 encoder.
No existing model weights are loaded. Official Train: 14,773 recordings; Test:
4,093 recordings. No Validation and no Early Stopping.

Reference repository: https://github.com/rhwxmx/EventMamba
Pinned commit: `7f61fd397cf69f10f0f92a182109d36cd7b6fdbb` (API checked 2026-10-11).
Relevant untouched source snapshots and MIT license: `eventmamba_v2_upstream/`.
Existing V1, SECNet and fusion files/environments are not modified.

## Verified discrepancies and adaptations

1. Official `train_classification.py` imports **V1**, defaults to **150 epochs**,
   batch 32, AdamW lr=0.001/wd=0.0001, cosine eta_min=0. It evaluates Test each
   epoch and selects the best Test checkpoint. We explicitly select V2, train
   **350 epochs as reconfirmed by the user**, never select on Test, and only
   evaluate the final epoch. No validation is introduced.
2. Official `models/modules.py` activates coordinate KNN by default, while its
   README explicitly instructs switching V2 to **feature-space KNN**. We make
   that documented switch. Coordinate FPS and temporal index sorting remain.
3. README DailyDVS dimension `[32,64,128]` is shorthand inconsistent with a
   literal reading of stage outputs. V2 code embeds 3->32 and uses THREE stages
   outputting **64,128,256**, groups **1024,512,256**, K=24. The fourth group in
   the source list is unused. We follow code, not an invented smaller network.
4. Local residual Conv-BN-ReLU, local Attention Pooling, LayerNorm+BiMamba-v2,
   global residual Conv-BN-ReLU, final Attention Pooling, and classifier
   Linear(256,256)->BN->ReLU->Dropout(0.5)->Linear(256,200) follow the source.
   The upstream repeated classifier construction in the initialization loop is
   retained. Mamba layer_idx=0 for all stages is retained. Parameter count:
   **953,676**. Input is `[B,3,2048]` in **t,x,y** order, no polarity.
5. The default unfused Mamba block path is reproduced without imports of unused
   timm/old fused-layernorm modules unavailable locally. Real BiMamba forward
   and backward kernels are used, not a mock or unidirectional substitute.
   The author did not pin a BiMamba fork; README's generic mamba-ssm installation
   is insufficient. We use the existing isolated `.venv-eventmamba`: Torch
   2.0.1+cu118, Vim `dd0358ad1e42701f22afbefa0717cc8825cf9f45` BiMamba fork,
   causal-conv1d 1.4.0. This is a verified-compatible dependency, **not proven
   to be the exact author's dependency revision**. Source/binary hashes are
   stored in checkpoints. No dependencies were upgraded for this experiment.

## Window protocol

Reference: `dataprocess/generate_daily.py` and `uti.py` in the pinned snapshot.

- Read existing processed event NPZ via official split manifests. Check all
  200 labels, duplicate IDs/paths and Train/Test disjointness.
- Use integer microseconds, subtract recording start before any floating-point
  conversion. Window 1,500,000 us, stride 500,000 us. Full windows anchored to
  the first event, half-open `[start,start+1.5s)`. No extra trailing partial
  window; recordings shorter than 1.5s get one whole-recording window.
- This repairs upstream's end-index exclusion and boundary mismatch; the script
  also uses a strict end-before-last test, so exact-boundary cases can differ.
- Each retained window is uniformly sampled to **2048** events, without
  replacement when possible, otherwise with replacement, sorted by index/time.
  Sampling is fixed offline, not redrawn every epoch (matches official HDF5).
- Identity/original windows use sampled timestamps quantized to milliseconds,
  followed by sampled min/max normalization as in generate_daily. Augmented
  windows normalize sampled timestamps without ms quantization. Zero spans are
  guarded. Spatial normalization is **x/320,y/240**, adapting the official
  x/128,y/128 to this sensor. No SECNet/gCSTR spatial augmentation is added.
- Train variants follow official intent: original, random spatial/time crop,
  time reverse, vertical flip, horizontal flip. Official `random_crop` is reused.
  `reverse_T`, `flip_H`, `flip_W` are called by the author but **absent** from the
  repository; we supply explicitly defined semantics: t reversal with reordering
  to chronological order, y->239-y, x->319-x. This part is an adaptation, not
  asserted bitwise reproduction of unavailable functions. Empty crops are
  skipped and counted; degenerate time ranges are guarded.
- Official original-window eligibility is `events > 10000`; augmented training
  variants are produced even when the original fails this threshold. We retain
  that distinction. Test uses originals only. If a recording would have ZERO
  usable windows, use its densest candidate window (or the whole recording if
  none exists), sample/repeat to 2048 and count a fallback. No official test
  recording is silently removed. Empty/invalid recordings fail explicitly.
- All preprocessing choices above are fixed before Test. Every cached window
  stores original sample index, label, window index and augmentation variant.

## Cache and resource management

Only sampled float32 point windows are stored, not full gCSTR images. The
estimated capacity is data-dependent (24 KiB/window plus small metadata); logs
report actual counts and GiB. Train cache builds before optimization. Test
cache is built **only after all training finishes**. Smoke reads only Train.
HDF5 building uses four processes, bounded batches of 32 recordings, file locks,
source/protocol signatures, committed recording/window counters and atomic
rename on completion. An interrupted preparation can resume from its last
committed batch. Do not edit preprocessing code during a partial build.

## Training / checkpoint / evaluation

- Batch size **32 globally**, ordinary float32, seed 0, AdamW defaults except
  lr=1e-3 and wd=1e-4, cosine over all 350 epochs, no warmup/early stopping.
- Label smoothing exactly follows author: target=0.8, each other class=0.2/199.
  This is not exactly PyTorch's built-in `label_smoothing=0.2` convention.
- Drop the shuffled final incomplete training batch as the author does;
  evaluation keeps EVERY window and EVERY sample.
- `last.pt` atomically saves every complete epoch, optimizer, scheduler,
  per-rank Python/NumPy/Torch/CUDA RNG and loader RNG, protocol/source/dependency
  and both official manifest hashes. Restart repeats an interrupted partial
  epoch, not a mid-batch exact continuation. CUDA kernels may be nondeterministic.
- Only `last.pt` and `final.pt` are retained, no best-on-Test checkpoint. Failed
  signature checks stop rather than silently mixing protocols.
- Single GPU is tested. A torchrun DDP path exists: global batch32 / ranks,
  DistributedSampler with drop_last and SyncBatchNorm. Final evaluation runs on
  rank 0 alone to avoid distributed padding or duplicated Test samples.
  Two-GPU DDP passed CUDA smoke on 2026-10-12 after GPU 0 became idle.
  Global-batch32 throughput probes found no speedup; the formal run remains
  single-GPU. See `EVENTMAMBA_V2_DDP_BENCHMARK_2026-10-12.md`.
  Do not switch an existing run's world size on resume; signature rejects it.

### Metric definitions (all accuracies stored as fractions)

| Level | Top-1 | Top-5 | `loss` |
|---|---|---|---|
| Window | argmax logits, each window equal weight | top 5 logits | ordinary CE, window-weighted |
| Original sample | hard Majority Voting of window argmax | up to 5 classes with positive vote counts, ranked by votes | CE of mean window probabilities, each sample equal weight |

Voting ties are broken by the earliest window voting for the tied class
(consistent with Counter's first-seen tie behavior), then class index. Top-5
does NOT fill with zero-vote classes; samples with fewer than 5 voted classes
are counted explicitly. Sample loss is **not** a loss on hard votes: hard votes
do not define a smooth probability distribution. `smoothed_loss` is additionally
reported at both levels. `mean_probability_top1` is an auxiliary clearly named
metric, NOT a replacement for Majority Voting and NOT used to select weights.
Explicit grouping by sample index fixes upstream's incomplete-tail sample
aggregation bug. Aggregate votes, counts, tie order and mean probabilities are
saved to `final_sample_predictions.npz` for audit, not tuning.

## Launch and monitoring

From `/remote-home/chenhaoquan/event_project`, only after smoke checks:

```bash
mkdir -p logs/eventmamba_v2_only
setsid nohup .venv-eventmamba/bin/python -u supervise_eventmamba_v2.py --output-dir checkpoints/eventmamba_v2_only_official_seed0_v1 --cache-dir eventmamba_v2_cache/official_seed0_v1 --epochs 350 --devices 1 > logs/eventmamba_v2_only/supervisor_seed0_v1.log 2>&1 < /dev/null &
```

GPU 0 was at 46,059 MiB and 100% utilization; do not displace its job. GPU 1
has ample headroom. The supervisor records GPU/memory/process status each minute,
warns about stale training metrics, records exit status, and stops with an error
if the run exits without final Test results. It never changes hyperparameters,
automatically restarts, or kills other jobs. Logs are local; no automatic chat
notification is claimed. During cache preparation, the trainer waits on its
cache lock if the standalone preparation process is already building it.

One command for live preparation/training progress:

```bash
tail -F /remote-home/chenhaoquan/event_project/logs/eventmamba_v2_only/prepare_train_seed0_v1.log /remote-home/chenhaoquan/event_project/checkpoints/eventmamba_v2_only_official_seed0_v1/console.log
```

Process/GPU monitoring:

```bash
watch -n 5 "pgrep -af '[t]rain_eventmamba_v2_only.py|[s]upervise_eventmamba_v2.py|[e]ventmamba_v2_data.py'; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv"
```

### Resume (only after confirming old processes exited)

Use the SAME launch, adding
`--resume checkpoints/eventmamba_v2_only_official_seed0_v1/last.pt` to the
supervisor command, append logs with `>>`. Do not pass `--resume` if preparation
was interrupted before a first checkpoint; incomplete cache builds can continue
with the original command. Do not simultaneously start multiple supervisors.
If the final model was saved but evaluation interrupted, resume evaluates that
same final model. Existing `test_final.json` prevents repeat evaluation.

## Files and outputs

- `eventmamba_v2_model.py`, `eventmamba_v2_layers.py`: independent V2 model.
- `eventmamba_v2_data.py`: windows, fixed preprocessing, resumable caches.
- `eventmamba_v2_metrics.py`: exact smoothing and explicit voting metrics.
- `train_eventmamba_v2_only.py`: single/DDP training and final evaluation.
- `supervise_eventmamba_v2.py`: bounded-lifetime background supervision.
- `test_eventmamba_v2_only.py`: data/metric/KNN and official CUDA parity checks.
- `eventmamba_v2_upstream/`: untouched reference files and MIT license.
- `checkpoints/eventmamba_v2_only_official_seed0_v1/`: checkpoints, console log,
  JSONL metrics, supervision, PID record and eventual `test_final.json`.
- `eventmamba_v2_cache/official_seed0_v1/`: window caches and preparation state.

## Interpretation

V1/SECNet baselines used whole-recording 8192-point sampling; V2 uses overlapping
2048-point windows, different augmentations and sample voting. This is a useful
standalone-method baseline, NOT a clean encoder-only controlled ablation.
Compare test metrics on the same official sample split, disclose these protocol
differences, and do not tune against the final Test result.

## Verification record

- 13 tests passed (6 V2 plus 7 SECNet/fusion regression tests).
- Real CUDA parity: parameter names/shapes load strictly against the official
  V2 network with the documented feature-KNN switch; matching eval forward and
  gradients with tolerances for CUDA floating-point reduction. Both models use
  real BiMamba kernels. Default-inactive fused norm/DropPath imports are shimmed
  only in the independent reference test, never replaced by fake mixers.
- Real training-mode smoke: batch32, 2048 events, 4 workers, two optimizer steps
  per tiny epoch, finite gradients; checkpoint restores into epoch2 at lr=0.0005
  for the two-epoch smoke schedule. Final eval covers 21 unaugmented windows and
  all 4 original TRAIN recordings. No official Test events read for smoke.
- Peak CUDA allocated memory: 4842.78 MiB (~4.73 GiB), distinct from reserved
  memory/context reported by nvidia-smi.
- Final smoke artifacts: `checkpoints/eventmamba_v2_only_smoke_final/`.
- Old smoke versions retained separately for diagnostic provenance.
- Existing SECNet encoder SHA256 unchanged:
  `a3dcf0b2ff28bf909f2bd5e6f7923d3f6e38c3b506f566d6f7bcf1adcf4f5682`.
- At initial launch, two-GPU execution was unverified due to occupied GPU 0;
  subsequent 2026-10-12 smoke passed but throughput was lower than single-GPU.
- Formal supervised pipeline launched 2026-10-11 18:47 Asia/Singapore:
  supervisor PID 5207, training-entry PID 5209, Train-cache preparation PID 2111.
  At handoff the cache was still preparing (2,240/14,773 recordings, 69,420
  augmented windows, 1.59 GiB point data). **Formal epoch 1 had not started**;
  the training entry waits on the cache lock and automatically proceeds when
  the signed Train cache is complete. These PIDs/progress are a snapshot only.
