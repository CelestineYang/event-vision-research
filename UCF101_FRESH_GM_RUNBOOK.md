# UCF101 GM：独立预训练分支＋重新初始化融合

用户于2026-10-29确认：之前提到的数据问题实际上是主实验依赖FusionA初始化，不需要更换UCF101数据。沿用现有TXT划分及事件窗口清单：Train 10,296视频/108,066窗口，Test 2,555视频/27,384窗口。保留既有划分的限制，不将此实验描述为独立Test估计。

## 初始化与协议

独立官方filename EventMamba V2 checkpoint严格加载，Point可训练、BN运行统计冻结；三路独立gCSTR best_loss.pt严格加载，图像参数和BN统计冻结，来源epoch分别49/39/41（cstr3/xt_p/ty_p）。图像checkpoint绑定的window_manifest SHA必须与本次数据一致。

基础融合层随机初始化，Gate/Memory独立随机流初始化，PF和Memory输出残差末层置零。没有加载FusionA权重、优化器或其训练后的Point参数。加载器与配置验证均拒绝source_checkpoint/source_sha256字段。

正式配置：global video batch48，TBPTT4，50 epochs，seed0，FP32，AdamW，Point LR1e-5，基础融合LR5e-5，Gate/Memory LR1e-4，weight decay1e-4，clip1，无scheduler/early stopping；主CE+0.5 Point CE；按视频Top-1选模，同分NLL更低者优先，另存NLL最佳。Test用于选模，无独立Val。单GPU运行，无梯度累积。

## 文件

- `ucf101_fresh_gm_fusion.py`：独立branch loader与fresh GM。
- `ucf101_fresh_gm_runtime.py`：配置、数据/源码身份、三视频smoke排程及选模。
- `train_ucf101_fresh_gm.py`：训练和epoch边界恢复，禁止未经数据版本确认的正式运行。
- `benchmark_ucf101_fresh_gm.py`：仅Train的显存探测。
- `configs/ucf101_fresh_gm/GM_batch48.json`：本机正式配置。
- `configs/examples/ucf101_fresh_gm/GM_batch48.json`：公开路径模板。替换路径并核实数据版本后，将data_version_confirmed设为true。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-eventmamba/bin/python -u train_ucf101_fresh_gm.py \
  --config configs/ucf101_fresh_gm/GM_batch48.json \
  --output experiments/ucf101_fresh_gm_batch48_seed0_20261029
```

此目录已有正式任务时不要再次启动。恢复同一任务用--resume，身份校验不允许偷偷更改batch或源码。现有GM.json仅为早期工程smoke配置（batch16），不是正式配置。

## 验收

独立分支严格CPU加载、3项策略测试通过。真实CUDA smoke及epoch2恢复完成：每轮45个Train窗口，共17次更新；408个冻结图像/Point BN tensor逐项不变；零残差初始化logits最大差0。随后正式参数batch48探测处理336个Train窗口，峰值allocated25521 MiB、reserved27472 MiB。正式任务从初始分支权重重新开始，不复用任何smoke状态。

正式任务首先做固定参数Test基线评估，再开始epoch1。验收记录保留在本地实验目录。此文档不是已完成训练或已得到正式结果的声明。

截至 2026-11-03，独立初始化 GM 尚无正式训练结果；上述 smoke 指标只用于工程验收。
