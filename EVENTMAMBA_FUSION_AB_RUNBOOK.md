# EventMamba 融合实验 A / B

验证记录（2026-10-04）：13 项单元及回归测试通过；A、B 均在 GPU 0 上用
8192 点真实数据完成训练与验证各 1 个 batch；A 从 last.pt 恢复到第 2 轮成功。
对应短测试目录是 `gcstr_checkpoints/eventmamba_aux_smoke_v1` 与
`gcstr_checkpoints/eventmamba_local_smoke_v1`。这些小样本指标仅验证流程，不评估精度。
完整训练尚未启动。

先运行 A，再根据验证结果决定是否运行 B。两者均从头初始化点云编码器和融合模块，
加载同一组三路 gCSTR 图像 checkpoint。不要从此前已过拟合的融合 checkpoint 续训。
使用原有 train/val manifest，训练入口不评估官方 test。

## 配置与比较目的

| 配置 | 融合结构 | 点云辅助头 | 目的 |
|---|---|---|---|
| 原基线 | 三层级 prototype 融合 | 无 | 已完成的基线 |
| A | 原三层级 prototype 融合 | 有 | 单独验证点云直接监督的作用 |
| B | 最后一级点云 + 三路 layer3 局部融合 | 有 | 验证坐标对齐融合方案 |

辅助头是 `mean(point_features_last) -> LayerNorm(256) -> Linear(256,200)`，
输入在所有跨表征交互之前，图像不能参与辅助分类。训练目标为：
`objective_loss = fusion_loss + 0.5 * point_loss`。0.5 是这轮预先固定的起始权重，
不是已验证的最优值；A、B 使用相同权重。

B 使用 256 个点特征（投影到 128 维）查询三路 14×14 的 ResNet layer3 特征图，
每点每视图采样 3×3 区域。三视图更新等权平均，以固定 0.1 系数残差加入点特征，
然后平均池化、线性分类。B 同时改变了融合层级和图像特征分辨率，结果代表整个
结构方案，不能单独归因于局部注意力。动态门控和分类 logits 残差留待后续消融。

## 坐标约定

- EventMamba 中心点为 `[t,x,y]`，采样时 x、y 分别除以 W、H。
- gCSTR 行列分别是 XY=(x,y)、XT=(x,t)、TY=(t,y)。
- 恢复整数空间像素坐标；时间用 `floor(t * time_bins)`，末端裁剪到最后一格。
  时间映射受点云 float32 时间精度限制，极接近分箱边界时可能与原始 float64 不同。
- SciPy zoom 默认保持首尾像素中心，先换算到 224×224 输入像素，再除 layer3 的
  stride=16 得到特征坐标。边界使用 border 采样；这是局部对应先验，不是逐事件精确匹配。

## 日志与 checkpoint

- `loss/top1/top5`：融合分类的指标。
- `point_loss/point_top1/point_top5`：纯点云辅助分类指标。
- `objective_loss`：加权训练目标，包括辅助损失。
- `best.pt`：按 `--selection-metric` 选取的模型，默认仍是融合验证 loss。
- `best_top1.pt`：最高融合验证 Top-1 的模型（相同准确率时保留较早的）。
- `last.pt`：最后一轮，可按原配置恢复训练。

本轮 A/B 保留原基线按 loss 早停、patience=10 的规则，便于对照。
辅助 loss 不参与早停判定。若以后改用 `--selection-metric top1`，应作为新的训练协议记录。
续训会校验融合模式、辅助权重和选择规则；不能用 --resume 切换 A/B。
旧 checkpoint 没有 best_top1 记录时，恢复后的准确率峰值只从恢复点开始跟踪。

## 启动 A（先执行这一组）

```bash
cd /remote-home/chenhaoquan/event_project
mkdir -p logs/eventmamba_ab
CUDA_VISIBLE_DEVICES=0 nohup .venv-eventmamba/bin/python -u train_gcstr_fusion.py \
  --point-encoder eventmamba --fusion prototype --point-aux-weight 0.5 \
  --selection-metric loss --manifest gcstr_splits_seed0.csv \
  --output-dir gcstr_checkpoints/eventmamba_aux_full_seed0_v1 \
  --epochs 50 --batch-size 16 --num-workers 4 \
  --learning-rate 1e-4 --weight-decay 1e-4 --patience 10 --seed 0 \
  --point-count 8192 --augmentation spatial --augmentation-count 5 \
  --log-every 20 --device cuda \
  > logs/eventmamba_ab/aux_seed0_v1.log 2>&1 &
echo $!
tail -f logs/eventmamba_ab/aux_seed0_v1.log
```

## B 命令（A 分析之后再执行）

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv-eventmamba/bin/python -u train_gcstr_fusion.py \
  --point-encoder eventmamba --fusion local --point-aux-weight 0.5 \
  --selection-metric loss --manifest gcstr_splits_seed0.csv \
  --output-dir gcstr_checkpoints/eventmamba_local_full_seed0_v1 \
  --epochs 50 --batch-size 16 --num-workers 4 \
  --learning-rate 1e-4 --weight-decay 1e-4 --patience 10 --seed 0 \
  --point-count 8192 --augmentation spatial --augmentation-count 5 \
  --log-every 20 --device cuda \
  > logs/eventmamba_ab/local_seed0_v1.log 2>&1 &
echo $!
tail -f logs/eventmamba_ab/local_seed0_v1.log
```

## 训练结束后诊断 A

```bash
CUDA_VISIBLE_DEVICES=0 .venv-eventmamba/bin/python -u diagnose_eventmamba_points.py \
  --checkpoint gcstr_checkpoints/eventmamba_aux_full_seed0_v1/best_top1.pt \
  --output logs/eventmamba_ab/aux_seed0_v1_point_diagnostic.json
```

B 将 checkpoint/output 换成对应 local 路径即可。诊断只使用每类 2 个的固定验证子集，
点特征与坐标成对替换，不重复 FPS。错配敏感性作为辅助诊断，不作为训练目标。
比较结论应同时看融合验证准确率、辅助头验证准确率与错配结果。
