# Project continuity and reproducibility

This repository preserves implementation and research artifacts, not chat history. Start with this file when moving to a new Codex session.

## Research direction

The project combines full-event gCSTR image projections with EventMamba point features. B is the base prototype fusion architecture; G adds an independent frame reader and a gated feature residual; GM adds causal cross-window feature memory. G-FixedGate fixes the added gate to 0.5 while retaining the frame reader and residual MLP. Therefore B versus G does not isolate the gate alone.

For THU-CHL, all variants start from the same independently pretrained image branches and EventMamba V1 checkpoint, with newly initialized fusion. They do not initialize from a trained Fusion A model. Historical UCF GM instead started from trained Fusion A epoch 40. The new independent-initialization UCF entry is prepared but has no formal training result; do not mix these protocols.

## Entry points

- `gcstr_feature_memory_ablation.py`: isolated Gate/Memory ablation policies.
- `train_thu_chl_ablation.py`: THU single-GPU training, Top-1 selection, additional NLL checkpoint, epoch-boundary resume.
- `thu_chl_sequence_dataset.py`: chronological video windows and aligned image/point augmentation.
- `thu_chl_ablation_fusion.py`: strict independent-branch loading and identity checks.
- `train_thu_chl_memory_only.py`: isolated M (base fusion + memory residual) training; see `THU_CHL_MEMORY_ONLY.md`. M removes the complete PF supplement, retains the memory writer gate, and is distinct from GM-FeatureOnly. Its CPU tests and strict pretrained-branch loading pass; real CUDA smoke and formal training remain pending.
- `train_gcstr_feature_memory_ablation.py`, `evaluate_gcstr_feature_memory_ablation.py`: UCF ablation training/evaluation.
- `prepare_ucf101_ablations.py`: relocate UCF configs and audit supplied assets.
- `configs/examples/`: portable configuration templates. Replace `ASSET_ROOT` and `LOCAL_PATH` strings with actual paths; these are literal placeholders, not automatically expanded environment variables. Copy THU templates to `configs/thu_chl_ablations/` and UCF templates to `configs/feature_memory_ablations/` for tests and default benchmark commands.

## THU experiment protocol

50 classes; 1,786 Train videos / 7,129 windows; 544 Test videos / all 2,392 nonempty windows. Preserve the official split, including sparse Test windows with explicit point sampling fallback. Test is used for model selection, not an independent held-out evaluation.

Historical-GM-aligned configs are under `configs/examples/thu_chl_ablations/gm_aligned/`: global video batch 48, TBPTT 4, 50 epochs, seed 0, FP32, AdamW, weight decay 1e-4, clip 1, no scheduler or early stopping. Point LR 1e-5, base fusion LR 5e-5, new-module LR 1e-4. Freeze image encoders and Point BN running statistics. Loss is fusion CE + 0.5 Point CE. Select by video Top-1; local code breaks ties with lower NLL and saves a separate NLL-best checkpoint.

Historical GM used three DDP ranks, local batch 16; completed G used global batch 48 on one GPU. Global grouping and update counts match, but random-number streams and floating-point execution are not bitwise identical. `train_thu_chl_ablation.py` does not implement gradient accumulation. Concurrency and memory availability must be checked on the destination machine.

## Results as of 2026-11-03

The current source of truth is [the main/ablation snapshot](research_exports/main_ablation_2026-11-03/README.md), including complete curves, selection rules, four prediction tables and input/checkpoint checksums.

- THU-CHL: G completed 50 epochs, selected epoch 40, 439/544 = 80.6985% video Top-1. GM completed 50 epochs, selected epoch 10, 442/544 = 81.25%. Both evaluate all 2,392 windows. The +3-video difference is one seed with single-GPU/DDP execution differences. B and M have no completed result; G-FixedGate is also incomplete and G-GlobalGate deferred.
- DVSAction: GM completed 50 epochs, selected epoch 47, 51/57 = 89.4737%; 239 Test windows. Independent pretrained branches, new fusion, shared spatial augmentation of distinct raw-image/SNN-point populations. No completed B/G/M ablations.
- UCF101-DVS: historical Fusion A log maximum is 96.1252% at epoch 40; historical GM log maximum is 96.7123% at epoch 47. The latter weights were not retained. Its retained NLL-selected epoch 17 scores 96.2427%. GM initializes from trained Fusion A, so this is not a matched independent-initialization comparison. The new fresh-GM entry is prepared, not formally trained.
- DailyDVS-200: no completed current-window-protocol fusion main experiment or B/G/M/GM ablation. Standalone branch pretraining is outside this result snapshot's scope.
- Historical frozen-expert probability routing is an appendix, not feature-level G/GM. Its G2 three-seed mean is 96.7776% ± 0.0815 percentage points (sample SD), selected by NLL on the original Test.

All listed main results use the original Test for checkpoint selection. No claim of an untouched final evaluation set. A prepared configuration or successful smoke test is not a completed experiment.

New entry points: `train_dvsaction_gm.py`, `train_ucf101_fresh_gm.py`, DVSAction preprocessing under `dvsaction_preprocessing/`, DailyDVS-200 prepared-window adaptation under `dailydvs200_prepared_data.py`, and point pretraining under `train_eventmamba_paper.py`. See their runbooks before reuse.

Raw datasets, weights, experiment directories, machine-specific configs and local handoffs remain outside Git. No task should auto-start merely because this document is read. This publication leaves active training sources unchanged; their local path defaults may differ from portable public versions and must remain unchanged for identity-checked resume.
