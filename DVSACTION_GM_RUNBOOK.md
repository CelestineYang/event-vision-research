# DVSAction GM

Independent initialization: author DVSAction EventMamba V2 checkpoint (10 classes), three independently pretrained image branches chosen by video NLL (cstr3 epoch26, xt_p43, ty_p35). All image parameters and running statistics frozen. Point trainable with BN running statistics frozen. Base fusion randomly initialized; PF and Memory residuals zero initialized. No FusionA input.

Point architecture: embedding32, stages64/128/256, group counts512/256/128, K24. The point checkpoint strictly loads. Group counts are non-tensor hyperparameters, chosen from author configuration and the uploaded training log's 2048_512 experiment name, not inferred from strict tensor loading alone.

Data: 226 Train videos / 926 valid windows; 57 Test videos /239 valid windows. 0.5s windows, 0.25s stride. Point input uses author SNN reconstructed pixels; image input uses raw polarity events in the matching temporal window and 150x150 ROI. Training applies the same per-video rotation/crop parameters separately to both event populations; per-event coordinate jitter is independent, since the populations are different. This explicitly approved fusion adaptation does not rerun SNN on augmented raw events. Evaluation uses exact prepared HDF5 points and cached image projections.

Protocol: seed0,50epochs, single GPU batch48 video lanes,TBPTT4,FP32,AdamW wd1e-4,clip1. Point LR1e-5,existing fusion5e-5,new modules1e-4. Loss fusionCE+0.5PointCE. Memory reset between video groups, detached at TBPTT boundaries. No scheduler or early stopping. Test videoTop1 chooses best (ties lowerNLL); separate bestNLL also saved. Original Test used for selection; no independent validation/test claim.

Entry: train_dvsaction_gm.py. Public template: `configs/examples/dvsaction_gm/GM.json`. Replace paths, verify the dataset identity, set `data_version_confirmed` to true, and save a local working copy as `configs/dvsaction_gm/GM.json`. Sources: dvsaction_gm_fusion.py,dvsaction_gm_data.py,dvsaction_gm_runtime.py. Independent sources and data/code checksums saved with checkpoints; do not edit these dependencies during a run.

Smoke: experiments/dvsaction_gm_smoke_20261031; three complete Train videos,21windows,2lanes,TBPTT4. Two invocations exercised disk resume to epoch2/8steps. Initial logits equal base exactly;408 image/Point BN tensors unchanged after training. Full batch probe uses48 actual Train videos and never initializes the formal model. Evidence: reports/dvsaction_gm_preparation_20261031/.


Completed result (2026-11-03 audit): 50 epochs, Top-1-selected epoch 47, 51/57 videos = 89.4737%, Top-5 100%, video NLL 0.348034. See [the result snapshot](research_exports/main_ablation_2026-11-03/README.md) for all epochs and the selected predictions.

```bash
CUDA_VISIBLE_DEVICES=0 python train_dvsaction_gm.py --config configs/dvsaction_gm/GM.json --output experiments/dvsaction_gm_new_run
```
