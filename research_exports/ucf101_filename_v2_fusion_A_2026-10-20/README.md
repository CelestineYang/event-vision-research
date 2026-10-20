# UCF101 filename V2 + gCSTR A: discussion snapshot

Snapshot exported at 2026-10-20T04:52:05.353276+00:00.
Contains complete fusion epochs **1–118 of 150**, not a completed150-epoch
experiment. Image pretraining contains all50 epochs. The source training can
continue independently; this directory is a static reviewed snapshot.

## Main findings

- Preselected by video Test NLL: epoch6, Top-1
  94.9902%, Top-5
  98.8650%, NLL
  0.239659.
- Peak video Test Top-1 (supplementary statistic): epoch40,
  96.1252%. This is not the NLL-selected model.
- Read `curve_summary.json` for latest metrics and matched-epoch comparisons.
  Accuracy plateaus while Test NLL rises later in training; the figures are
  diagnostic, not a new model-selection rule.

## Read these files first

1. `UCF101_FILENAME_V2_FUSION_RUNBOOK.md`: executed architecture/training recipe.
2. `fusion_config.json` and `fusion_metrics.jsonl`: actual config and full prefix.
3. `image_summary.json`, `image_config.json` and three `*_metrics.jsonl`: baselines.
4. `curves.png` / `curves.pdf`, `curve_summary.json`, `matched_epochs.csv`.
5. `data_protocol_summary.json`, `UCF101_TXTSPLIT_V2_RUNBOOK.md`, and the root
   repository's tracked `train_ucf.txt`, `test_ucf(1).txt`: data protocol.
6. `eventmamba_official_training.txt` and `checkpoint_index.json`: pretraining
   provenance. Official pretraining is not the current fine-tuned point head.
7. Five `*_predictions_best_loss.csv` files: video-level error analysis. Fusion
   and point CSVs both come from the fusion-NLL selected epoch; they are not
   independently selected point-only results. CSVs contain top5 predictions
   and true-class probabilities, not full101-class probability vectors.

## Architecture and questions for the next design

Frozen three gCSTR ResNet-18 backbones produce49 tokens/view. Trainable official
V2 stages produce1024×64,512×128,256×256 point features. Each stage uses16
prototype queries, then attends to each image view; three128-dimensional stage
vectors form the101-class fusion head. A pretrained attention-pooled point head
has an auxiliary CE loss with weight0.5. Image adapters and position embeddings
are trainable. Points and full-event images share a1s/0.5s window/augmentation.

Useful questions: Does prototype compression remove needed information? Does
the point branch add value over an equally trained three-image control? How
can later confidence overfitting be reduced? Propose small, controlled ablations
and measurable hypotheses before expanding the model.

## Interpretation limits

- Test participates in checkpoint selection; it is not independent held-out data.
- Original TXT lists have 1754 shared class/group
  identities, although exact video overlap is zero. This is not group-disjoint.
- Only retained valid-window videos are evaluated (2555 videos /
  27384 windows); filtering changes the evaluation population.
- Fusion warm-starts from task-trained images and pretrained points. Global
  batch16 versus image batch64 means equal epochs are not equal update budgets.
- No matched three-image fusion/ensemble control has yet been established.
- Official normalization uses x/320,y/240 despite actual240×180 dimensions;
  window indexing and training augmentation are recorded adaptations.

No checkpoints, raw dataset payloads, event/frame caches, environments or
authentication material are included. `MANIFEST.json` lists SHA256 provenance
for every exported member. Original experiment files remain unchanged.
