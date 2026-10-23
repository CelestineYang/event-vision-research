# 第二阶段：冻结专家路由对照与 Memory 诊断

## 本批状态与边界

基于第一阶段已完成的 epoch40 五头缓存和四个 80 轮模型；不改动
`routing_data.py`、`fusion_router.py`、`train_fusion_routing.py`，保留 checkpoint 源码哈希校验。
新增外围评估、配对分析、CPU 回归测试和配置，不修改 THU 便携流水线。

- 已读取并核验服务器真实缓存、sources.json 中列出的 SHA、checkpoint 身份、选模记录、标准化器及实际初始化权重。
- 已完成 CPU 合成验证。随后经用户批准，真实全量 suite 的22项评估及逐视频配对已完成，四个普通最优模型均复现原指标。
- 真实评估：`reports/routing_stage2_full_v1/`；汇总及配对：`reports/routing_stage2_full_v1_analysis/analysis.md`、`summary.json`、`paired.json`。
- Static Fusion+Frame及六次新增多种子训练均已完成80轮；静态对照最优epoch8，NLL=0.1936431。
- Current/GRU/Conditional三种子准确率均为96.3209%，纠错集合一致；详细结果见 `research_exports/ucf101_routing_stage2_2026-10-23/README.md`。
- 可选前缀评估尚未启动；本次未导出逐窗口轨迹，仅保存窗口统计和完整视频概率。
- 真实审计报告：`reports/routing_stage2/artifact_audit_final.json`。
- 测试日志：`reports/routing_stage2/cpu_tests.log`。这些小型报告可随代码分享；不要上传缓存、权重、全量窗口轨迹。

CPU 回归共32项通过（原路由24项 + 本批8项）；其中小型拟合仅使用合成数据。

实际输入：Train 10296 视频 / 108066 窗口，Test 2555 视频 / 27384 窗口。
Static / Current / GRU / Conditional 的选中 epoch 为 10 / 58 / 11 / 21。
动态模型实际 reference alpha=0.7908821、eta=0.0941077、b0=1.3302507；
配置中的 alpha0=0.5、eta=0.5 不是加载后的权重。
Point 上游训练身份仍记录 unknown，不补写未经核实的来源。

## 文件与接口

- `routing_stage2.py`：从 sources 解析真实产物并严格核验；复用原 Router，临时 hook 观察及读出干预。
- `evaluate_fusion_routing_suite.py`：五头、固定集成、四个最优模型、Current+EMA、历史干预、可选前缀。
- `analyze_fusion_routing.py`：按 dataset/source_split/video_id 精确配对，核对类别及窗口身份，输出错误集合及逐视频差异。
- `test_fusion_routing_stage2.py`：合成序列、真实接口小型拟合、篡改拒绝、诊断不改变预测等 CPU 测试。
- `configs/fusion_routing_stage2.json`：同缓存评估矩阵；默认全部有效窗口、batch32 视频、ECE15 bins、EMA0.9。
- `configs/fusion_routing_static_frame_stage2.json`：仅学习 eta 的 Static Fusion+Frame。
- `configs/fusion_routing_stage2_seed1.json` / `seed2.json`：固定专家和原 Static 的动态模块多种子配置。

所有下述命令在项目根目录执行。新评估及分析输出目录必须不存在，拒绝覆盖。
CPU 默认即可；显式 `--device cuda:0` 才使用 GPU。不重新运行编码器、FPS 或导出五头。

## 独立评估命令（本次已执行，重跑须使用新目录）

```bash
.venv-eventmamba/bin/python evaluate_fusion_routing_suite.py evaluate \
  --config configs/fusion_routing_stage2.json \
  --output reports/routing_stage2_full_v1 --device cpu

.venv-eventmamba/bin/python analyze_fusion_routing.py \
  --suite reports/routing_stage2_full_v1 \
  --output reports/routing_stage2_full_v1_analysis
```

默认共 22 个评估项：14 个去重普通模式，4 个已训练模式各加 2 个历史干预。
五头 Fusion/Point/cstr3/xt_p/ty_p 独立评估；Frame 是三个图像头概率平均。
固定 Fusion、Frame 不重复计算。其他固定项为 frame_point、fusion_frame、three_way。
Current+EMA 固定使用同一个 Current epoch58 和 decay0.9，不扫描系数或重新选 epoch。
四个普通最优模型必须复现保存的指标，否则 suite 失败；小整数指标精确一致，浮点指标允许 2e-6 绝对误差。

加 `--traces` 才写逐窗口 JSONL；普通输出保留汇总统计，不把大轨迹塞入训练日志。
每项均有完整 `video_log_probs.npy`、`video_index.json`、指标及文件 SHA。
可选 `--prefixes .25 .5 .75 1`，使用另一个新输出目录；按有效窗口数 ceil 截取，不是按原始时间比例。
所有方法共享前缀，从零状态开始，不按前缀重新选模。

## 诊断解释

窗口路径权重为 w_A=1−eta、w_F=eta×alpha、w_P=eta×(1−alpha)。
当前 w_A≈90.5892%，alpha 只分配其余约9.4108%。这不是 Point/Frame 的网络级因果贡献。
轨迹包含窗口身份、时间、dt、alpha/eta、路径权重、current/memory 修正、状态范数、Conditional 写门分位数。
标签及真实类别概率仅在前向完成后离线记录，不进入观测或状态。

- normal：原因果前向，逐步观察与整段原前向一致。
- reset_each_window：每窗口重置隐藏状态，保留当前观测、预计算 dt 和 Train 标准化器。Conditional 仍有 r(o,0)，GRU 仍处理当前观测。
- zero_memory_readout：仅将加到门控 logit 的读出置零，保持状态更新。GRU 读出含当前更新信息，不能全部称作纯历史贡献。
- Static/Current 为历史干预负对照，应保持不变。Padding 不更新状态、不计入投票；视频之间不传递状态。

配对输出包含预测改变、谁对谁错、相对 Fusion 的纠错/新增错误视频集合、逐视频 NLL 和概率差异；
干预还输出 Top1/NLL/Brier/ECE 差值、窗口 alpha 与最终概率变化。相同 7/0 计数不代表纠正同一批视频。
这是固定 checkpoint 的推理期机制诊断，不是重训结构消融；不得仅由无变化断言 Memory 无价值，或由下降断言专门读写优于 GRU。

协议沿用 original_test_as_val：原 Train 拟合参数及标准化，原 Test 选视频 NLL 最优模型；independent_test=false。
没有独立测试、多种子结论或 OOF 证据。工具从缓存读取 dataset_id/class_to_idx，不写死101类；其他数据集须提供匹配协议的产物与配置。

## Static 对照（已完成，勿重复运行同一输出目录）

下面路径对应已核验 sources.json；迁移服务器时先更新该索引并重新审计，不能把 Git 小型快照当权重。

```bash
ROUTING_BASE=/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3
.venv-eventmamba/bin/python train_fusion_routing.py fit \
  --config configs/fusion_routing_static_frame_stage2.json --mode static \
  --cache "$ROUTING_BASE/train" --val-cache "$ROUTING_BASE/test" \
  --train-role gate_train --val-role gate_val --selection-protocol original_test_as_val \
  --output checkpoints/routing_stage2_static_frame_seed0 --device cpu
```

80 epochs，batch32 视频，Adam lr1e-3，wd0，clip1，seed0，alpha固定1，仅学习eta；
原 Test 视频 NLL 选模。输出独立，不覆盖已完成 unrestricted Static，绝不作为动态模型的新初始化。
拟合完可用原训练入口的 `eval --checkpoint ... --cache "$ROUTING_BASE/test" --role gate_val --output 新目录` 单独复核。

## 多种子（已完成，以下保留复现命令）

seed0直接复用已有结果；seed1/2共6次拟合已完成。重跑必须使用新输出目录：

```bash
ROUTING_BASE=/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3
for routing_seed in 1 2; do
  for routing_mode in current gru conditional; do
    .venv-eventmamba/bin/python train_fusion_routing.py fit \
      --config "configs/fusion_routing_stage2_seed${routing_seed}.json" --mode "$routing_mode" \
      --cache "$ROUTING_BASE/train" --val-cache "$ROUTING_BASE/test" \
      --static-init "$ROUTING_BASE/static/best_val_nll.pt" \
      --train-role gate_train --val-role gate_val --selection-protocol original_test_as_val \
      --output "checkpoints/routing_stage2_${routing_mode}_seed${routing_seed}" --device cpu
  done
done
```

同80轮/batch32/lr1e-3预算，只变动态模块初始化与训练顺序；这是固定专家及 Static 条件下的 Gate/Memory 方差，不是整个系统重训方差。

## 复核命令

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_fusion_routing test_fusion_routing_stage2 -v
.venv-eventmamba/bin/python evaluate_fusion_routing_suite.py audit --output reports/routing_stage2_new_audit.json
```

审计会读取缓存以验证完整性，但不会执行真实模型前向；前缀评估仍需另行显式启动。
