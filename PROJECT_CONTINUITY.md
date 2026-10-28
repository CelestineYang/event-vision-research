# Project continuity and reproducibility

This repository preserves implementation and research artifacts, not chat history. Start with this file when moving to a new Codex session.

## Research direction

The project combines full-event gCSTR image projections with EventMamba point features. B is the base prototype fusion architecture; G adds an independent frame reader and a gated feature residual; GM adds causal cross-window feature memory. G-FixedGate fixes the added gate to 0.5 while retaining the frame reader and residual MLP. Therefore B versus G does not isolate the gate alone.

For THU-CHL, all variants start from the same independently pretrained image branches and EventMamba V1 checkpoint, with newly initialized fusion. They do not initialize from a trained Fusion A model. UCF experiments use a different initialization protocol; do not mix their configs with THU.

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

Historical GM used three DDP ranks, local batch 16; current training entry processes global batch 48 on one GPU. Global grouping and update counts match, but random-number streams and floating-point execution are not bitwise identical. `train_thu_chl_ablation.py` does not implement gradient accumulation. Concurrency and memory availability must be checked on the destination machine.

## Preserved evidence and pending work

`research_exports/thu_chl_ablation_2026-10-27/` contains the completed historical GM metrics and verification summary. GM reached 81.25% video Top-1 at epoch 10; its Top-1- and NLL-selected model tensors are identical. Do not report unfinished local ablations as completed results.

UCF ablation preparation includes synthetic CPU tests and real Train-only CUDA smoke checks; see the tests and configuration-generation entry point. Remaining research work includes completing protocol-matched THU baselines and Gate/Memory ablations and comparing selected checkpoints. A plan or prepared config is not proof that a run completed.

Raw datasets, pretrained weights, experiment directories, machine-specific configs and private local handoffs stay outside Git. Keep them backed up separately. No training task should auto-start merely because this document was read.
