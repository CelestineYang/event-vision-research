# DVSAction independent image pretraining

Entry: `train_images.py`, with `image_dataset.py` adapting the prepared data and frame manifests. Three separate ImageNet ResNet-18 models and separate Adam optimizers, each with a newly initialized 10-class head. All image backbone parameters trainable here; freezing belongs to later fusion training. This stage does not load EventMamba or FusionA weights.

Formal run: `experiments/dvsaction_gcstr_three_branch_seed0_20261031`, GPU 4 UUID recorded in `logs/dvsaction/formal_launch_20261031.json`. Settings: 50 epochs, batch 64 windows, workers 4, seed 0, Adam LR 1e-4, weight decay 0, FP32, no scheduler/early stopping. 926 Train windows (226 recordings); 239 Test windows (57 recordings), no dropped last batch. Three branches execute sequentially per shared batch on one GPU.

Train uses full raw event windows cropped into 150x150, shared per-window deterministic epoch-dependent spatial augmentation, then three projections. Evaluation uses the verified unaugmented frame cache. ImageNet normalization is applied in the dataset. IDs are full recording filenames, not split-local numeric IDs.

Original Test is used for selection; no independent validation split, `independent_test=false`. Video scores are arithmetic means of window softmax probabilities. Each branch saves `best_loss.pt` by video NLL, `best_top1.pt` by video Top-1 with lower NLL tiebreak, `last.pt`, metrics and selected predictions. Joint `resume.pt` contains three models, optimizers, RNG and selection state. Resume requires identical config/data/source hashes; use the same command plus `--resume`. Do not alter these source files while training is running.

Real CUDA smoke completed with two videos per split and random initialization, separately from the ImageNet-initialized formal run. Dataset checks cover split identity, deterministic augmentation within an epoch, changed augmentation across epochs and evaluation shapes. All smoke checkpoint heads were 10-class and finite. ImageNet weights are locally cached and their SHA256 is recorded in the formal config.

Monitor from project root:

```bash
tail -f logs/dvsaction/train_three_branch_seed0_20261031.log
```
