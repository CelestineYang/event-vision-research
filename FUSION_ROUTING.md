# Frozen expert Gate and Memory framework revision 2

## 已完成实验更新（2026-10-23）

下文 revision 2 工程验收及 epoch6/20轮/batch8 命令保留为历史说明，
**本轮实际实验以此节和结果快照为准**：使用 Fusion A epoch40，GPU0 单卡导出，
batch64、workers4；Static/Current/GRU/Conditional 均为 80 轮、batch32 视频、
Adam LR=1e-3。三种动态模式共享 Static 第10轮最优权重和 Train 标准化器。
显式运行配置为 `configs/fusion_routing_epoch040.json`。

完整结果及来源哈希见 `research_exports/ucf101_routing_epoch040_2026-10-23/`。
四种模式均正常结束，80轮连续，25760步；Static/Current/GRU/Conditional 的
NLL选中轮次分别10/58/11/21，视频Top1分别96.2427/96.3209/96.3209/96.3209%。
同缓存 Fusion Top1=96.0470%；Test参与选模，independent_test=false。
EMA和其他固定权重对照尚未进行真实全量评估，不能据此宣称记忆优于当前窗口门控。

真实GPU回归已执行：原模型重复前向也存在float32微小波动，因此Fusion同时检查
original-vs-original和original-vs-wrapper，固定rtol=1e-6/atol=1e-5，
另验概率rtol=1e-5/atol=1e-6及Top1一致。Point和模型状态仍逐位检查；
Train/Test各3批真实batch64通过，workers4输入路径及小规模训练通过。
原UCF专家代码未修改。缓存和权重保存在服务器datasets，Git仅提交小型报告快照。

本阶段采用已确认的 `original_test_as_val` 协议：原 Train 的全部有效视频用于
Gate/Memory 参数和标准化拟合，原 Test 用于验证及按视频 NLL 选择 checkpoint。
不再拆 Train，不重新随机划分，没有额外独立 Test；所有结果记录
`independent_test=false`。默认 API 严格模式仍存在，选择新协议通过配置或
`--selection-protocol original_test_as_val` 完成，不需要另加 seen-expert 覆盖参数。

## 文件和行为变化

| 文件 | 本轮变化 |
|---|---|
| export_fusion_evidence.py | 明确协议贯穿角色、导出身份；保留完整严格 Fusion 恢复、一次 Point/FPS 前向、原图像 avgpool/fc 路径 |
| fusion_evidence_dataset.py | 新增可导入顶层 ExportDataset，支持 spawn；真实来源与 augmentation/采样状态分开，rows 裁剪直接作用于采样 |
| prepare_fusion_routing_roles.py | 新增审计清单与原 TXT 交叉核对、完整角色映射、数量和重叠报告；不重划分 |
| routing_data.py | schema 2，17 维公共观测，窗口/序列共用真实 gap，协议检查，Train-only 标准化 |
| fusion_router.py | 原始静态 logit 直接初始化；状态 buffer 恢复；显式端点与有限 logit 饱和分开；共同 gap 输入 |
| train_fusion_routing.py | 原 Test 验证选模协议、动态直接恢复标准化器、schema 2 checkpoint、预先固定 EMA 系数 |
| analyze_ucf101_complementarity.py | 实际差值方向、unique rescue 数量/比例、空错误 CSV、动态 checkpoint 元数据及当前协议 |
| test_fusion_routing.py | 初始化精度、协议、gap、spawn、分析边界及四种模式的合成训练 |
| configs/fusion_routing.json | 本轮共同配置，明确 original_test_as_val 与 independent_test=false |
| configs/fusion_routing_roles.template.json | 新协议模板 |
| configs/fusion_routing_roles.original_test_as_val.json | 已生成的完整真实有效视频用途映射 |

14 个原 Fusion A 源码文件哈希保持与已完成实验完全一致；原 Backbone、2048 点、
投影、Point 归一化、Cross-Attention、训练入口、权重与数据缓存均保持原样。
本轮没有推送 Git。

## 已确认的数据角色和专家来源

此前四项问题现在落实如下：

1. 原有效 Train 全部映射 gate_train，原有效 Test 全部映射 gate_val，
   source_split 保持 train/test，同一视频全部窗口继承角色。
2. 新协议允许验证来源为专家已见或未知；事实标签保留，不标成 unseen。
   严格模式原有 --allow-seen-expert-validation 含义保持不变，不能凭它允许 Test 选模。
3. Test 可以评估和选 checkpoint，任何协议下都不能进入梯度训练或标准化拟合。
4. 官方 Point 初始化的逐视频拟合来源继续 unknown；已知 Fusion A 在 Train 微调、
   在 Test 选模单独记录。Test 的专家拟合接触为 unknown、选模接触为 seen。
   Point 输出来自当前完整恢复的 Fusion A，未混入独立官方初始化模型的输出。

已生成的真实统计如下；报告在
[role_mapping.json](reports/routing_revision_v2/role_mapping.json)。

| 来源 | 原始视频 | 有效视频 | 有效窗口 | 有效类别 | 无有效窗口视频 |
|---|---:|---:|---:|---:|---:|
| Train | 10819 | 10296 | 108066 | 101 | 523 |
| Test | 2704 | 2555 | 27384 | 101 | 149 |

原始 TXT、当前审计与历史已上传摘要的上述数量一致。缺失/重复记录为 0，
相同视频交集为 0；已有 class/group 交集为 1754，作为已有协议特征报告，
没有新增 group 互斥条件。报告包含排除的视频 ID、已有审计排除信息、
原始/有效类别覆盖、窗口数及清单哈希。生成工具 --check 可只读复核。

首轮共同专家为：
`/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/best_loss.pt`。
已在 CPU 读取元数据并核对 SHA256：
`368edde5c2a196455c8960a31672c8b37f0e64da6cbb9e21c0d356251e974223`。
它对应 epoch 6，视频 NLL 0.239658978845264。所有模式共用它及同一角色映射、
同一五头证据，不改用准确率峰值的 checkpoint。

## 数据缓存和输入

缓存 schema=2、observation_schema=2。五头 logits 的顺序仍为
fusion / point / cstr3 / xt_p / ty_p，float32 [N,5,C]；事件统计 [N,4]，
均可 mmap 读取。保留原子完成标记、文件/清单/源码/权重/类别/角色身份、
有限值、完整性、重复和缺失检查。新 Dataset 模块纳入导出源码身份。
首版 16 维缓存和 Gate checkpoint 明确拒绝，不静默迁移或移除哈希检查。
本次没有执行真实证据缓存导出；旧文件不会被重写。

导出默认 augmentation=none、采样 epoch=0、seed=0，原始 source_split 不伪造。
保持原 2048 点采样与官方归一化、全事件窗口图像投影。单进程模型前向，
可用 spawn DataLoader workers；原 Test 的现有图像缓存只读复用。
记录 batch、workers、随机设置、设备和软件身份；改变 batch/GPU/随机流
不承诺与历史 DDP 指标逐位相同。

观测顺序由 routing_data.OBSERVATIONS 固定并写入缓存与训练配置：

1. 五头依次为归一化预测熵、Top1–Top2 概率 margin，共 10 维。
2. Frame–Point 的 JS 分歧除以 log(2)。
3. 三个图像头的一致配对比例。
4. log1p(每像素每秒事件数)、log1p(事件数)、空间覆盖比例、
   Point 样本数/原事件数，共 4 维。
5. 最后一维为 log1p(dt_seconds)，总计 17 维。

每个视频按真实 start_us 排序后预计算 gap，首个有效窗口为 0；
窗口模式、序列模式、shuffle、不同 batch 都使用同一映射。
gap 先 log1p 一次，再用 Train 的均值/标准差标准化。
Dataset 和 ObservationNormalizer.fit 共用 observation_for/observations；
conditional 不再额外拼接或再次 log1p gap。原始 dt 和 overlap 仍作为元数据返回。
标签、真实类别概率、NLL、视频/group/fold ID、未来统计和完整语义向量不进入 Gate。

## 路由模式和训练

概率公式保持：
pF=(pXY+pXT+pTY)/3，
pB=alpha*pF+(1-alpha)*pPoint，
pFinal=(1-eta)*pFusion+eta*pB。
使用算术概率混合和稳定 log-space NLL，温度固定为 1。

| 模式 | 接口 |
|---|---|
| fixed | eval --fixed fusion/frame/frame_point/fusion_frame/three_way |
| static | 学习 alpha、eta；fusion_frame_only=true 可固定 alpha=1 |
| current | 17→64→1 当前观测 MLP，无隐藏历史 |
| ema | 对 current checkpoint 显式 eval --ema-decay 0.9 |
| gru | 当前观测更新 GRU，然后读出门控 |
| conditional | 当前观测读取旧状态生成门控，再条件化写入供后续窗口使用 |

固定三路径等权仍是 alpha=1/2、eta=2/3。
current/gru/conditional 从同一 unrestricted static checkpoint 初始化：
直接复制 router_state.alpha_logit 到 base_logit；
从同 dtype 的 eta_logit.sigmoid() 设置 fixed_eta；
fixed_alpha、EMA 初始值、padding 回退值同步。
这些 buffer 不进入优化器。动态修正输出零初始化，完整 log-probability
和 NLL 与静态组合一致。有限 logit 即使 sigmoid 浮点饱和也不被标记为
显式精确端点；用户直接指定的精确 0/1 仍按端点处理。恢复以真实 state_dict
为准，记录静态 checkpoint 哈希、角色/缓存身份及初始化来源。
eta 很小时继续记录影响很小的退化情况。

默认 Adam，LR=1e-3，weight decay=0，20 epochs，每批 8 个视频，
hidden/state=64，gradient clip=1，seed=0，无早停。参数量：
static 2，current 1217，GRU 17218，conditional 17090。
三种动态模式具有相同的 17 维输入、静态起点、标准化器、数据、训练预算、
选模规则。static 在 Train 拟合一次标准化器；动态直接加载，不重复拟合。

每个 epoch 覆盖所有有效训练窗口；不同长视频的 batch 目标按
sum_window_NLL * epoch_steps / epoch_windows 缩放，保持窗口权重。
padding 不参与损失、不更新状态、不参与投票；视频间状态重置，
状态显式传入传出，不使用全局字典。评估先混合窗口概率，再在视频内平均。

记录窗口/视频 Top1/Top5、窗口 NLL、视频 NLL、
Brier=视频平均的类别平方误差和、15 个等宽置信度 bin 的视频 ECE，
相对同缓存 Fusion A 的纠错/新增错误/净变化、门控分布、样本数、
优化步数及参数量。Top-k=min(5,C)。按 gate_val（原 Test）的视频 NLL
保存 best_val_nll.pt，同时保存 last.pt、优化器/RNG、完整配置/身份和指标。
没有独立 Test 成绩。EMA 固定使用配置中的 0.9；显式不同系数会拒绝，
本轮不进行温度、EMA 或结构搜索。当前入口仍不提供 resume 命令。

## 工程验收范围

实际执行 CPU 单元测试和新协议的合成小训练，涵盖不同长度视频、
不等间隔、padding、四种可训练模式，以及 fixed 和 EMA 评估。
最终命令实际完成 35 项测试，全部通过（24 项路由/修订测试与 11 项原有回归），
没有将真实权重 CUDA 检查作为 skip 计入通过数。
真实 ExportDataset 的 Train 和 Test-as-Val 路径均测试
spawn + num_workers=1，并与 workers=0 核对。
旧模型 CPU 接口回归使用真实 ResNet 和原融合路径，Point 是 CPU 测试替身。
真实专家 checkpoint 只读取元数据/哈希，没有执行真实权重 CUDA 前向回归，
也没有全量导出、正式 Gate/Memory 训练、专家重训或联合微调。

运行测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest \
  test_fusion_routing test_ucf101_window_gcstr test_ucf101_event_store test_research_snapshot -v
```

## 后续可执行命令

以下真实数据命令本轮均未执行。先在项目根目录设置新输出位置和共同专家。
命令里的 GPU 1 是后续执行时指定的设备，并不表示本轮已占用该卡。

```bash
routing_python=.venv-eventmamba/bin/python
routing_roles=configs/fusion_routing_roles.original_test_as_val.json
routing_checkpoint=/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/best_loss.pt
routing_output=experiments/routing_revision_v2_small

# 真实角色映射已经生成；只读核对原清单、协议和输出一致性。
CUDA_VISIBLE_DEVICES='' "$routing_python" prepare_fusion_routing_roles.py --check

# 在尚未生成的环境可运行以下命令；拒绝覆盖已有文件。
# CUDA_VISIBLE_DEVICES='' "$routing_python" prepare_fusion_routing_roles.py

# 单批真实权重回归，workers=0，写入零个证据缓存。
CUDA_VISIBLE_DEVICES=1 "$routing_python" export_fusion_evidence.py \
  --selection-protocol original_test_as_val --checkpoint "$routing_checkpoint" \
  --roles "$routing_roles" --source-split test --max-videos 1 \
  --batch-size 8 --workers 0 --device cuda:0 --integration-regression

# 两侧各取最多两个完整视频，只做后续小规模工程检查。
for source in train test; do
  CUDA_VISIBLE_DEVICES=1 "$routing_python" export_fusion_evidence.py \
    --selection-protocol original_test_as_val --checkpoint "$routing_checkpoint" \
    --roles "$routing_roles" --source-split "$source" --max-videos 2 \
    --batch-size 8 --workers 0 --device cuda:0 --output "$routing_output/$source"
done

# 五个 fixed 对照在原 Test 的 gate_val 上报告。
for variant in fusion frame frame_point fusion_frame three_way; do
  CUDA_VISIBLE_DEVICES='' "$routing_python" train_fusion_routing.py eval \
    --selection-protocol original_test_as_val --cache "$routing_output/test" \
    --role gate_val --fixed "$variant" --output "$routing_output/fixed_$variant"
done

# static：梯度和标准化来自原 Train，原 Test 视频 NLL 选模。
CUDA_VISIBLE_DEVICES='' "$routing_python" train_fusion_routing.py fit \
  --selection-protocol original_test_as_val --config configs/fusion_routing.json \
  --cache "$routing_output/train" --val-cache "$routing_output/test" \
  --train-role gate_train --val-role gate_val --mode static \
  --output "$routing_output/static"

# 相同静态 checkpoint、数据、标准化和预算。
for mode in current gru conditional; do
  CUDA_VISIBLE_DEVICES='' "$routing_python" train_fusion_routing.py fit \
    --selection-protocol original_test_as_val --config configs/fusion_routing.json \
    --cache "$routing_output/train" --val-cache "$routing_output/test" \
    --train-role gate_train --val-role gate_val --mode "$mode" \
    --static-init "$routing_output/static/best_val_nll.pt" --output "$routing_output/$mode"
done

# EMA 是 current 的预先固定评估选项，沿用其 video-NLL 选中 checkpoint。
CUDA_VISIBLE_DEVICES='' "$routing_python" train_fusion_routing.py eval \
  --selection-protocol original_test_as_val --cache "$routing_output/test" --role gate_val \
  --checkpoint "$routing_output/current/best_val_nll.pt" --ema-decay 0.9 \
  --output "$routing_output/current_ema"
```

小样本输出不能当完整实验成绩。后续全量导出需另用全新目录并去掉
--max-videos；本轮不会自动执行。所有方法必须共用同一版本缓存，
不可用不匹配的角色、专家或采样身份初始化。旧 16 维小模块 checkpoint
需重做小模块实验；无需因此重训原专家。
