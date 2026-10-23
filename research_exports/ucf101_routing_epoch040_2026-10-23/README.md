# Completed frozen-expert routing experiment

Snapshot assembled 2026-10-23. Expert: trained Fusion A epoch 40; all five
heads are frozen and share the same window. Train fits parameters and input
normalization; original Test selects checkpoints by video NLL.
**This is not an independent held-out Test evaluation.**

Train: 10296 videos / 108066 windows. Test: 2555 videos / 27384 windows.
Export: single GPU0, batch64 windows, workers4, augmentation off, seed0.
Fit: 80 epochs, batch32 videos, Adam 1e-3, weight decay0, no early stopping.
All three dynamic modes initialize from Static epoch10 and reuse its normalizer.

| Method | NLL-selected epoch | Video Top1 | Video Top5 | Video NLL | Rescued / damaged vs Fusion |
|---|---:|---:|---:|---:|---:|
| Fusion (same evidence cache) | expert40 | 96.04697% | — | — | — |
| Static | 10 | 96.24266% | 99.09980% | 0.19723610 | 5 / 0 |
| Current | 58 | 96.32094% | 98.94325% | 0.19598950 | 7 / 0 |
| GRU | 11 | 96.32094% | 98.94325% | 0.19685699 | 7 / 0 |
| Conditional | 21 | 96.32094% | 98.94325% | 0.19640851 | 7 / 0 |

Current has the lowest selected NLL. Memory has not shown additional Top1
improvement in this run. Equal rescue counts do not prove identical rescued
samples. No multi-seed significance claim is made. EMA has not been evaluated.
The historical expert40 score from the original run is not substituted for
the same-cache baseline; random FPS depends on batch/order during export.

Each method directory contains its actual config, all 80 epochs of metrics,
the selected video predictions and completion status. `summary.json` also
contains final-epoch results. `sources.json` records source paths and SHA256;
checkpoint hashes are included but checkpoint binaries are not.
No event arrays, logits caches, weights, credentials, or environments are included.

Implementation details and regression limits are documented in FUSION_ROUTING.md.
THU preprocessing/training code is a separate portable package; its results are
not part of this UCF experiment. Open3D end-to-end validation remains pending.
