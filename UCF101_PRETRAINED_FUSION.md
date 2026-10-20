# UCF101 pretrained window fusion

## Active configuration: filename V2 (2026-10-16)

This section supersedes the historical V1 notes below. Active loader
`load_pretrained_fusion(experiment, point_checkpoint, freeze_images=True,
freeze_points=False)` now uses `PretrainedEventMambaV2`. It strictly loads all
168 tensors of the newly uploaded filename `best_checkpoint.pth`, with no
excluded keys or LSTM override. The former sliding checkpoint was deleted.

Embedding channels: 32. Stage outputs: 64/128/256. FPS groups: 1024/512/256.
K=24, feature-space KNN, BiMamba-v2. Original final attention and classifier
are retained. Checkpoint epoch 146 corresponds to log epoch 147, sequence
accuracy 90.32385466%, other Test accuracy 82.585%. `eventmamba.txt` records
150 epochs, AdamW, batch 128, initial LR 1e-3, decay 1e-4, 2048 points.
These pretraining settings do not define a fusion optimizer or loss recipe.

The official `generate_by_filename.py` also uses 1s windows and 0.5s stride,
>10000 events/window, <40 candidate windows/video and filename splitting before
window extraction. It is not whole-video point sampling. Its public constant
NUM_POINTS=8192 differs from the uploaded log's num_point=2048. Our approved
default remains 2048 with x/320, y/240 and sampled time min/max normalization.
The shared frame manifest's half-open boundaries are intentionally retained;
upstream boundary edge cases are not reproduced. Shared spatial augmentation
and per-epoch point resampling are fusion adaptations to offline H5 preparation.
Actual dataset sensor dimensions remain 240/180 for frame generation.

The three frame `best_loss.pt` paths documented below remain active. Image
backbones default frozen/eval; adapters/position/fusion modules trainable;
point network trainable unless `freeze_points=True` (also sets point eval).
Outputs: new fusion `logits` and original classifier `point_logits`. Train-mode
point classifier BN in a single process needs at least two samples. The current
DDP trainer uses SyncBatchNorm and exact global batches, including local
singleton tails with at least two global samples. Formal training started on
2026-10-16; see UCF101_FILENAME_V2_FUSION_RUNBOOK.md for the executed recipe
and the reviewed research_exports snapshot for a dated result summary.

Current check (no optimizer step or checkpoint write):

```bash
OMP_NUM_THREADS=2 .venv-eventmamba/bin/python test_pretrained_window_fusion.py
```

Passed: all four actual checkpoints loaded strictly, deterministic real-window
input, exact original V2 classifier agreement with identical FPS RNG, stage
feature/coordinate shapes, training-mode CUDA forward/backward, finite point
and fusion gradients, unchanged frozen image BN and frozen-point eval behavior.
Existing standalone V2 and DailyDVS trainers are unchanged.

Official preprocessing source:
https://github.com/rhwxmx/EventMamba/blob/7f61fd397cf69f10f0f92a182109d36cd7b6fdbb/dataprocess/generate_by_filename.py

## Historical V1 notes (inactive; not instructions for the current checkpoint)

Everything below is preserved historical context from the abandoned sliding V1
attempt. Its loader arguments, deleted checkpoint metadata, LSTM questions and
Smoke command do not apply to current filename V2. Use the active API above.

This independent entry leaves existing DailyDVS, V1/V2 and frame trainers unchanged.
It implements the existing prototype A fusion with 101 outputs, three pretrained
ResNet-18 feature maps and three pretrained EventMamba V1 feature stages.
Fusion layers are newly initialized. No formal training has been launched.

## Sources and unresolved settings

- Frame weights: `experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/checkpoints/{cstr3,xt_p,ty_p}/best_loss.pt`.
- Point weights: uploaded `best_checkpoint.pth`, widths 128/256/512 inferred
  from tensors, 101 classes, checkpoint epoch 297 (log epoch 298).
- Architecture: published V1 at rhwxmx/EventMamba commit
  `7f61fd397cf69f10f0f92a182109d36cd7b6fdbb`, MIT. Coordinate KNN,
  sorted FPS centers, K=24, BiMamba-v2, local and final attention pooling.
- Six `lstm.*` tensors are absent from the published V1 forward. The loader
  refuses them by default. `allow_unused_lstm=True` is an explicit hypothesis,
  not evidence that the author's historical forward ignored LSTM.
- Groups are a required loader argument because weights cannot determine them.
  The integration check uses README's UCF sliding groups 1024/512/256.
- User confirms the split is the author's split. The manifest remains the
  existing TXT-based video split; no new split is generated.

### Official repository history audit (2026-10-16)

Queried the public repository branches/tags and cloned its full available Git
history into a temporary directory. There is one branch (`main`, 7f61fd3), no
tags, 25 commits, and 40 distinct historical Python blobs. A case-insensitive
search of every one of these Python blobs found no `lstm` occurrence. All three
historical revisions of `models/eventmamba_v1.py` likewise contain no LSTM.
This establishes that the available public history does not explain the extra
checkpoint tensors; it does not prove they were unused in private training code.
Groups remain explicit and unverified against the original training run.
References: https://github.com/rhwxmx/EventMamba and
https://github.com/rhwxmx/EventMamba/commits/main/models/eventmamba_v1.py .

## Input contract

User-approved defaults are `point_count=2048` and
`point_normalization='official'`. It returns `events` [3,N] in t,x,y order, normalized
`cstr3`, `xt_p`, `ty_p` [3,224,224], `label`, `video_id`, `window_id`.
Use standard DataLoader collation and `model.forward_batch_outputs(batch)`.

Both branches use precisely the same manifest window (1 s / 0.5 s) and shared
event-level spatial augmentation. All augmented events form the images; the
point branch samples sorted indices with a separate deterministic RNG. Test
point sampling is epoch-independent. No additional window filtering occurs.

`official` point normalization uses x/320, y/240 and sampled time min/max;
`window` uses actual sensor dimensions and fixed manifest time boundaries.
Neither choice changes pretrained frame rendering. The former preserves the
published point preprocessing convention, including its hardcoded dimensions;
the latter is an explicit adaptation. Both the integration check and approved
production defaults use 2048 points and `official`.

The shared manifest intentionally preserves the trained frame window boundaries;
it does not reproduce the upstream window indexing off-by-one behavior.
Feature positions are point-space t,x,y; prototype attention does not perform
pixel-to-point geometric correspondence.

## Loading and validation

`load_pretrained_fusion(experiment, point_checkpoint, groups=...)` loads each
image model strictly and the complete published point network strictly after
any explicitly acknowledged LSTM exclusion. It never loads optimizer state.
Image backbones default to frozen/eval; point backbone defaults to trainable.
`freeze_points=True` also keeps point BN/dropout in eval. Returned outputs are
`logits` (new fusion head) and `point_logits` (pretrained official head);
a future trainer must explicitly decide whether/how to supervise both.

The real-window integration check below is diagnostic, not training. It does
forward/backward without an optimizer step and writes no checkpoint:

```bash
OMP_NUM_THREADS=2 .venv-eventmamba/bin/python test_pretrained_window_fusion.py --allow-unused-lstm
```

Passed on a real Test window: strict compatible tensor loading, deterministic
input, finite CUDA forward/backward, gradients into Mamba and fusion head,
and no gradients into frozen image encoders. This does not establish original
model equivalence or evaluate accuracy. Resolve original LSTM usage and grouping
before formal training or claiming reproduction of the uploaded checkpoint.
