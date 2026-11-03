# 主实验与消融结果汇总（2026-11-03）

范围：当前特征级融合主实验、对应消融及历史冻结专家概率路由附表。单独 gCSTR / EventMamba / SECNet 分支预训练、早期整段事件原型和工程 smoke 不作为主结果。下表根据完整训练日志重新选模，预测覆盖率独立核对，不是根据聊天记录抄录。

## 1. 已完成的特征融合实验

数值为**视频级**指标，Top-1/Top-5 使用百分比。除注明 NLL 外，按视频 Top-1 最大、平分时 NLL 最小、再平分取较早轮次。UCF 的 Top-1 行为日志峰值，并不将历史的 NLL 选模改称为 Top-1 选模。

| 数据集 | 方法 | 口径 | 选中/总轮数 | 正确/总视频 | Top-1 (%) | Top-5 (%) | 视频 NLL | 对应权重 |
|---|---|---|---:|---:|---:|---:|---:|---|
| THU-CHL | G | Top-1 最佳 | 40/50 | 439/544 | 80.6985 | 93.5662 | 0.866872 | 本地保留 |
| THU-CHL | GM | Top-1 最佳 | 10/50 | 442/544 | 81.2500 | 94.8529 | 0.809781 | 本地保留 |
| DVSAction | GM | Top-1 最佳 | 47/50 | 51/57 | 89.4737 | 100.0000 | 0.348034 | 本地保留 |
| UCF101-DVS | Fusion A（历史基础融合） | Top-1 日志峰值 | 40/150 | 2456/2555 | 96.1252 | 98.7084 | 0.362060 | 本地保留 |
| UCF101-DVS | Fusion A（历史基础融合） | NLL 最佳 | 6/150 | 2427/2555 | 94.9902 | 98.8650 | 0.239659 | 本地保留 |
| UCF101-DVS | 历史 GM | Top-1 日志峰值 | 47/50 | 2471/2555 | 96.7123 | 98.9824 | 0.327632 | 未保留，仅日志 |
| UCF101-DVS | 历史 GM | NLL 最佳 | 17/50 | 2459/2555 | 96.2427 | 99.0998 | 0.287184 | 本地保留 |

**THU-CHL：**GM 相比 G 提高 **0.5515 个百分点（3/544 个视频）**。两者均使用独立预训练图像/点分支、重新初始化融合，50 轮、seed 0、全局视频 batch 48、TBPTT 4。GM 是三卡 DDP、每卡 16，G 是单卡 48；随机流及浮点执行不能视为完全相同。配对预测中，GM 纠正 G 的 19 个错误，同时使 G 原先正确的 16 个视频变错（共同正确 423、共同错误 86），净增 3；见 [配对计数](paired_thu_g_vs_gm.json)。该结果支持进一步检验 Memory 的增益，但只有一个种子且缺少 B/M，不能直接宣布完整消融成立。GM 的 Top-1 / NLL 最佳权重张量一致，见[已有核验](../thu_chl_ablation_2026-10-27/historical_gm_verified.json)。

**DVSAction：**GM 完成 50 轮，epoch 47 与 50 的 Top-1 相同，47 的 NLL 更低。基础融合重新初始化，三路图像分支冻结，点分支可训练但固定 BN 运行统计。尚无匹配 B/G/M，单个 GM 结果不能证明 Gate 或 Memory 的独立贡献，也不能仅凭此表判断其优于最佳图像分支。

**UCF101-DVS：**历史 GM 加载的是**已经训练的 Fusion A epoch 40**；首轮 batch 16，后续三卡全局 batch 48，再训练 50 轮。Fusion A 训练了 150 轮，采用不同训练协议。两者峰值相差 0.5871 个百分点，但不能作为同初始化、同预算的模块消融。历史 GM epoch 47 的 96.7123% 只有日志，已保留的 NLL 最佳 epoch 17 为 96.2427%。独立初始化的新 GM 已准备好入口和配置，尚无正式结果。

所有主结果均用原始 **Test 选模，没有独立 Val**；它们不是 untouched test 的无偏估计。本快照不混入训练过程中的变化参数在线准确率。

## 2. 消融完成情况与方法定义

| 数据集/协议 | B | G | M | GM | G-FixedGate | G-GlobalGate |
|---|---|---|---|---|---|---|
| THU-CHL，融合重新初始化 | 无完整结果 | 80.6985% | 已实现，未正式训练 | 81.2500% | 无完整结果 | 暂缓 |
| DVSAction，融合重新初始化 | 未训练 | 未训练 | 未训练 | 89.4737% | 未训练 | 未训练 |
| UCF101-DVS，独立初始化新协议 | 无正式结果 | 无正式结果 | 未训练 | 已准备，未正式训练 | 仅 smoke | 仅 smoke |
| DailyDVS-200，当前窗口融合协议 | 未训练 | 未训练 | 未训练 | 未训练 | 未训练 | 未训练 |

UCF 的 G-PointQuery、G-PointComplement、G-FrameComplement、GM-FeatureOnly、GM-GateOnly、GM-FixedUpdate、GM-NoWriteGate 只有工程 smoke，没有可报告的正式训练准确率。DailyDVS-200 的分支预训练进展不填入这里的融合结果。

- **B**：基础 prototype 融合，结构对应 Fusion A；“同结构”不意味着可以把历史不同协议的 Fusion A 数值当作新 B 实验。
- **G**：在 B 上增加完整 PF 补充残差（独立 frame reader、点/图像混合及 Gate）。B→G 不是只打开 Gate；单独评估学习 Gate 应使用 G 对 G-FixedGate。
- **M**：B 加 Memory 残差，删除整个 PF 补充残差。M 仍保留记忆写入门，不能叫作“所有 gate 都关闭”。
- **GM**：同时包含 PF 补充残差和因果跨窗口 Memory。G→GM 用于研究该 Memory 机制，B/G/M/GM 四组应在统一协议下分别训练。

## 3. 数据与训练口径

| 数据集 | Train 视频/窗口 | Test 视频/窗口 | 数据要点 |
|---|---:|---:|---|
| THU-CHL | 1,786 / 7,129 | 544 / **2,392** | 全部非空 Test 窗口，含稀疏点窗；50 类输出，Test 实际含 46 类 |
| DVSAction | 226 / 926 | 57 / 239 | 固定上传清单；0.5 秒窗、0.25 秒步长；原始极性图像与 SNN 重建点云 |
| UCF101-DVS | 10,296 / 108,066 | 2,555 / 27,384 | 固定 TXT 划分；1 秒窗、0.5 秒步长；存在同 class/group 不同片段跨集合 |

THU G/GM 与 DVSAction GM：FP32、AdamW、weight decay 1e-4、梯度裁剪 1；Point / 基础融合 / 新模块 LR 分别为 1e-5 / 5e-5 / 1e-4；损失为 fusion CE + 0.5 Point CE，无调度、无早停。冻结三路图像编码器及 Point BN 运行统计，视频间重置记忆、TBPTT 4。这里“重新训练”指融合从随机/零残差初始化开始，仍使用独立分支预训练权重。

DVSAction 的点云使用作者 SNN 处理，图像来自匹配原始事件。共享空间旋转/裁剪参数分别作用于两类输入，不等价于增强原始事件后重新执行 SNN。THU 是 EventMamba V1；UCF / DVSAction 是 V2，不可互换权重。详见 [data_protocols.json](data_protocols.json) 和各训练入口文档。

## 4. 历史概率路由实验（独立附表）

这些是在固定专家缓存上训练的概率路由器，均使用 2,555 Test 视频、27,384 窗口，按视频 NLL 选模，预算 80 轮；不等同于特征级 B/G/M/GM。第二阶段复用第一阶段的 seed 0 结果，因此以下 22 条报告条目不代表 22 次互不重复的新训练。

| 阶段 | 变体 | 选中 epoch | 视频 Top-1 (%) | 视频 NLL |
|---|---|---:|---:|---:|
| 1 | conditional | 21 | 96.3209 | 0.196409 |
| 1 | current | 58 | 96.3209 | 0.195989 |
| 1 | gru | 11 | 96.3209 | 0.196857 |
| 1 | static | 10 | 96.2427 | 0.197236 |
| 2 | conditional_seed0 | 21 | 96.3209 | 0.196409 |
| 2 | conditional_seed1 | 3 | 96.3209 | 0.198183 |
| 2 | conditional_seed2 | 38 | 96.3209 | 0.197131 |
| 2 | current_seed0 | 58 | 96.3209 | 0.195989 |
| 2 | current_seed1 | 6 | 96.3209 | 0.198855 |
| 2 | current_seed2 | 33 | 96.3209 | 0.194491 |
| 2 | gru_seed0 | 11 | 96.3209 | 0.196857 |
| 2 | gru_seed1 | 3 | 96.3209 | 0.196958 |
| 2 | gru_seed2 | 13 | 96.3209 | 0.195150 |
| 2 | static_frame_seed0 | 8 | 96.2035 | 0.193643 |
| 3 | G1_seed0 | 78 | 96.5166 | 0.190859 |
| 3 | G1_seed1 | 34 | 96.4384 | 0.194672 |
| 3 | G1_seed2 | 19 | 96.4384 | 0.189079 |
| 3 | G2_seed0 | 9 | 96.8689 | 0.180928 |
| 3 | G2_seed1 | 6 | 96.7123 | 0.183979 |
| 3 | G2_seed2 | 7 | 96.7515 | 0.178437 |
| 3 | G3_seed0 | 3 | 96.2818 | 0.199542 |
| 3 | G4_seed0 | 7 | 96.5166 | 0.201206 |

第三阶段：G1 = 全局 eta + 窗口 NLL，G2 = 动态 eta + 窗口 NLL；G3/G4 分别改用视频 NLL。G1 三种子 Top-1 为 **96.4644% ± 0.0452 个百分点**；G2 为 **96.7776% ± 0.0815 个百分点**，这里 ± 是样本标准差。G2 seed 0 为 96.8689%。这些名字中的 G1/G2 不是上文的特征级 G。

同一专家缓存的对照：Fusion 为 96.0470%，三路图像概率集成为 97.1037%，Fusion 与图像等权混合为 97.0254%。G2 改善了 NLL，但 Top-1 尚未超过该图像集成。缓存上的 Fusion 与训练日志 epoch 40 的 96.1252% 来自不同评估/采样路径，不强行合并。

第二阶段，GRU/Conditional 在推理时逐窗重置记忆或置零读出，Top-1 未变但 NLL 有小幅变化。这是**固定权重的推理干预**，不是重新训练的去 Memory 消融，不能据此推断特征级 Memory 普遍无效。明细保存在 [legacy_probability_ablations.json](legacy_probability_ablations.json)，原始多种子快照仍保留在相邻日期目录。

## 5. 文件与可复核性

- [main_results.csv](main_results.csv)、[summary.json](summary.json)：七种选模口径、正确视频数、轮次、权重保留状态和解释限制。
- [ablation_status.csv](ablation_status.csv)：未完成/已完成状态，缺失结果不填零分。
- `curves/`：5 组完整训练/评测曲线，共 350 个 epoch；历史 UCF GM 的阶段记录已按 1–50 合并，未重复计数迁移阶段。
- `predictions/`：THU G、THU GM、DVSAction GM、UCF GM NLL 最佳的精简逐视频预测；核对 ID 唯一、视频数、窗口数和准确率。UCF Fusion A 保留的是日志口径，未冒充新评测。
- [data_protocols.json](data_protocols.json)：实际协议与输入来源身份，机器路径改为占位值。
- [checkpoints.json](checkpoints.json)：现存选中权重的文件大小及 SHA256，权重本体不上传。
- [sources.json](sources.json)：读取的原始指标、配置和预测文件 SHA256；配置经路径脱敏，原文件 SHA 不是脱敏后 JSON 的 SHA。
- [legacy_probability_ablations.csv](legacy_probability_ablations.csv)：历史概率路由的全部 22 条报告记录；包含明确复用的 seed 0 结果。

复现导出（路径需替换为有完整实验产物的本地目录）：

```bash
python export_main_ablation_results.py --project-root LOCAL_PATH/event_project --asset-root LOCAL_PATH/gm_ablation_assets_20261027_v1 --fusion-a-checkpoints LOCAL_PATH/fusion_A_checkpoints --output research_exports/main_ablation_2026-11-03
```

该命令只读取已完成指标并校验/复制轻量结果，不执行模型评测或训练。原始数据、全量预处理缓存、模型权重和后台运行记录保留在本地。公开代码使用显式路径参数；本次整理未改动正在训练的源码或数据。

发布验收：[validation.json](validation.json)。24 项 CPU 测试通过，26 个新增 Python 文件编译检查通过，全部 350 个 epoch 和 3,700 行预测覆盖率检查通过。
