# 下一批开发任务：特征级 Gate + 时序 Memory 融合模型与全量联合训练

> **执行口径：直接推进完整模型，不再把冻结分类概率上的微调实验作为前置关卡。**
>
> 本任务替代上一份 `CODEX_ROUTING_STAGE4_TASK.md` 中拟议的输出层 H1/H2/H3 扩展。前三阶段已经完成的代码、结果与 checkpoint 全部保留，不重做、不覆盖。
>
> 本次一次完成完整模型、序列训练入口和 B/G/GM 三组配置。必要的单元测试与小批 GPU 工程检查通过后，即具备进入完整数据训练的条件；不要求 Gate 单独涨点后才允许训练完整模型。

## 1. 已确定的方向与本次范围

我们要改造的是**原 Fusion A 内部的表示学习与跨窗口信息交互**，不再只学习五个分类头概率的组合权重。

本批同时实现：

1. 保留原 Point-guided Cross-Attention，补充不依赖当前 Point Query 的独立 Frame 特征读取。
2. 在原 prototype 均值池化之前加入特征级分支门控，让它改变分类头接收的表示。
3. 在同一视频的连续窗口之间维护紧凑的融合特征 Memory，辅助当前门控并向当前表示补充历史信息。
4. 使用已有完整 Fusion A 权重初始化，冻结图像骨干，联合更新 Point 编码器、原融合模块、原分类头及新增模块。
5. 用同一套训练流程安排原模型续训 B、无记忆 G、完整 GM 三组，直接进行完整训练集上的受控比较。

**本轮不做：**继续训练冻结 logits 上的 H1—H3；重新跑已经完成的路由实验；堆叠多个 Memory 架构；同时更换点数、投影、窗口协议或损失目标；从零重训全部编码器。

本任务中的新增结构是**本轮建议实现的研究候选**，并非现有代码已经验证的结论。允许 Codex 根据实际接口作等价工程适配，但不要自行替换为概率融合、增加新损失或扩展实验矩阵。

## 2. 已有依据与不变条件

### 2.1 从已有结果得到的实施方向

现有第三阶段结果支持继续研究动态门控，但尚未证明输出层 Memory 有额外价值；Frame 概率平均的准确率仍高于 G2，而 G2 的 NLL 更低。这支持保留独立图像证据、保留原交互，并把下一步放到特征学习，而不是宣称复杂结构已全面超过简单集成。

原 `PrototypeStage.forward()` 的核心为：

```text
Point prototypes
    → 分别读取 XY / XT / TY
    → 拼接后三视角 MLP
    → 与 Point prototypes 相加
    → mean(prototype)
```

本轮在最后的 `mean` 之前展开结构。前三阶段结论、当前实现事实和本轮新增假设必须在文档中区分。

### 2.2 保持以下设置

| 项目 | 本轮约定 |
|---|---|
| 初始专家权重 | 已确认的完整 Fusion A `epoch_040.pt`；从来源索引核对真实路径和哈希 |
| 数据 | 沿用现有 UCF101-DVS TXT 划分、审计清单和 EventStore，不重新划分 |
| 训练/选模 | 原 Train 全部有效窗口训练；原 Test 作为 Val，按视频 NLL 选模 |
| 协议标记 | `selection_protocol=original_test_as_val`；`independent_test=false` |
| Point 输入 | 2048 点，现有官方归一化与 t/x/y 排列，不新增极性通道 |
| Frame 输入 | 窗口全事件生成的 XY/XT/TY，保留现有投影、尺寸和标准化 |
| 原融合规模 | 先保持现有三个 stage、每层16个 prototype、128维及原 attention heads |
| 视频决策 | 有效窗口 softmax 概率平均，不平均 logits |
| 旧路由权重 | 不将 Static/Current/G2 的 MLP 参数移植进新特征 Gate |

旧五头 logits 缓存只用于历史对照与分析，**不能用于训练本轮会变化的内部特征**。新训练必须走事件/投影输入和真实网络前向、反向。

## 3. 模型结构：给出一个明确的首版实现

下述是本轮的默认结构，不再先做多种设计筛选。符号中 t 为视频内窗口，l 为原融合 stage；K=16，d=128。

### 3.1 保留原交互并暴露池化前 token

原 stage 的 Point prototype 和交互结果记作：

\[
P_t^l=\operatorname{CA}_{point}^l(Q_l,F_{point,t}^l),
\]

\[
Z_t^{l,v}=\operatorname{CA}_{view}^{l,v}(P_t^l,V_t^v),
\]

\[
H_{0,t}^l=P_t^l+\operatorname{MLP}_{old}^l([Z_t^{l,XY};Z_t^{l,XT};Z_t^{l,TY}]).
\]

位置编码、adapter、LayerNorm、attention 参数及计算顺序沿用原实现。`H0` 保留 `[B,K,d]`，暂不均值池化。

通过新增类/适配器暴露 token，不直接修改受 checkpoint 哈希保护的原文件。旧模型仍可单独加载、运行和复现。

### 3.2 新增独立 Frame 特征读取

新增 Frame 路径使用原 stage 的**可学习静态 query Q_l**直接读取三路图像 token，不使用当前样本的 P 或 Point features：

\[
\widetilde Z_t^{l,v}=\operatorname{CA}_{frame}^{l,v}(Q_l,V_t^v),
\]

\[
F_t^l=\operatorname{MLP}_{frame}^l([\widetilde Z_t^{l,XY};\widetilde Z_t^{l,XT};\widetilde Z_t^{l,TY}]).
\]

输出仍是 `[B,K,d]`。与原 Point 读取共享静态 query 的索引，便于建立可学习的槽位对应；**不宣称同索引 token 已经具有严格的几何或语义对齐**。

新 Frame attention 和合并 MLP 可以从对应旧模块复制数值初始化，但应是独立、可训练的参数；不要意外与旧模块权重绑定。原 Q_l 只注册一次，避免重复参数及优化器重复计数。

要求在固定图像 token 和模型参数下，改变 Point 输入不改变这条独立 Frame 路径的输出。共享静态可学习 query 不等于依赖当前样本的 Point Query。

图像 backbone 的计算只做一次：原交互、新 Frame 路径和必要的图像分类监测共用该次特征，不重复运行三个 ResNet。

### 3.3 特征级门控与残差补充

完整 GM 先从历史 Memory 读取 `R_t^l`（见下一节）；无 Memory 的 G 将历史上下文设为零。

每个 stage、每个窗口先使用**一个标量分支 Gate**，不一开始就做复杂的逐通道/逐prototype路由：

\[
g_t^l=\sigma\{\phi_l([\operatorname{pool}(\operatorname{LN}(P_t^l)),
\operatorname{pool}(\operatorname{LN}(F_t^l)),
\operatorname{pool}(\operatorname{LN}(R_t^l)),\log(1+\Delta t)])\}.
\]

无有效历史时，历史观测必须显式为零，不能被 LayerNorm bias 变成伪历史。G 与 GM 都获得相同的真实时间间隔信息。

\[
E_t^l=(1-g_t^l)\operatorname{LN}(P_t^l)+g_t^l\operatorname{LN}(F_t^l),
\]

\[
H_t^l=H_{0,t}^l+D_l(E_t^l)+\mathbb{1}_{history}\,W_l^{mem}(\operatorname{LN}(R_t^l)).
\]

`D_l` 为小型投影/MLP，`W_mem` 为小型投影。原 `H0` 残差完整保留，不对它额外加新归一化，不改变旧分类头的输入尺度定义。

- g 调节的是**新增补充路径中** Frame 与 Point 的比例，不是整个模型的总模态因果贡献。
- 不要求 g 偏向 Frame，不强制均衡，不用标签或真实类别概率生成门控。
- 首版 Gate 的末层零初始化，使 g 起步为0.5。
- `D_l` 最后一层和 `W_mem` 输出投影零初始化，使新模型初始预测恢复旧模型。
- **不要同时设置零残差标量和零输出投影**，避免构成无法自行启动的双零梯度路径。
- 零输出投影会使部分上游新模块第一步梯度为零，这是正常初始化行为；需验证数步优化后 Gate/Frame/Memory 能收到梯度。

最后仍按三个 stage 分别均值池化、拼接，送入已有分类头：

\[
z_t=\operatorname{cat}_l\operatorname{mean}_K(H_t^l),\qquad
logits_t=\operatorname{Classifier}_{old}(z_t).
\]

**主输出不再接旧的 α/η 概率混合器。** 本轮要评估的是新表示和分类头的联合学习。

### 3.4 单个共享特征 Memory

首版只维护一个视频级状态 `M:[B,16,128]`，不为三个 stage 各建大型记忆库；同时维护 `history_valid:[B]`。

**读取：**使用当前各 stage 的 `H0` 查询上一窗口状态：

\[
R_t^l=\operatorname{CA}_{read}^l(H_{0,t}^l,M_{t-1}).
\]

视频首窗口没有历史，R及其最终残差贡献必须显式为零。不要对“所有 key 都被屏蔽”的 attention 直接调用后依赖乘零消除 NaN；使用安全的分支/有效样本选择。

**写入：**当前窗口预测完成后，将三个 stage 的 `H_t^l` 作为48个候选 token，添加 stage embedding，以16个可学习记忆 query 做聚合：

\[
C_t=\operatorname{CA}_{write}(Q_{mem}+\operatorname{LN}(M_{t-1}),
\operatorname{cat}_{tokens,l}[\operatorname{LN}(H_t^l)+e_l]).
\]

跨 stage 聚合靠 attention 学习，不直接相加“同索引”的三个 stage token。

\[
U_t=\sigma\{\psi([\operatorname{LN}(M_{t-1}),\operatorname{LN}(C_t),\log(1+\Delta t)])\},
\]

\[
M_t=(1-U_t)\odot M_{t-1}+U_t\odot C_t.
\]

U 首版可以为 slot/channel 级门控。只对有效窗口写入，padding 保持旧状态。状态默认从零开始，读出当前结果之后才设置/更新下一窗口可读状态。

Memory 存储**融合特征**，不是概率、熵或视频标签；也不保存上一批其他视频的状态。它在窗口层面只读过去，不宣称事件级零延迟在线识别。

建议接口显式返回：

```python
outputs, next_state = model.forward_window(batch, state)
# outputs 至少包含 logits、point_logits；可选 feature/gate diagnostics
# state 包含 memory、history_valid，不能藏在全局 video_id 字典中
```

## 4. 三组主实验一次实现、同轮推进

| 配置 | 模型内容 | 训练目的 |
|---|---|---|
| B | 原 Fusion A 续训，不注册无用途新模块 | 排除新训练预算、序列组织和训练设置本身的收益 |
| G | 原交互 + 独立 Frame 读取 + 特征 Gate，无 Memory | 当前窗口结构改造对照 |
| GM | G + 特征 Memory 读写 + 历史条件门控 | 完整主模型 |

三组从同一完整 epoch40 权重初始化，原公共参数逐项一致。G/GM 的公共新增参数也采用相同初值；用独立初始化随机流，避免 GM 多构建模块改变公共参数或采样随机流。

**GM 不需要先等 G 提升才允许启动。** 资源允许时同批运行；否则按 GPU 资源顺序执行，不新增“方法先通过下一阶段审批”的流程。

B→G 同时改变独立 Frame 读取和 Gate，不能将全部增益单独归因于 Gate；该细分消融留待完整模型结果支持后补齐，不阻塞本批主模型训练。

## 5. 序列数据与截断反向传播

### 5.1 沿用原事件存储，不用 logits 训练新模型

复用既有 EventStore/mmap、审计清单、投影实现和未增强 Test 图像缓存。只加载当前正在处理的窗口片段到 GPU，不把整个长视频的全部事件和图像一次搬到 GPU。

数据组织按视频分组、按 `start_us` 排序；视频顺序每轮可打乱，视频内部顺序保持。显式返回窗口身份、原始时间戳、gap、有效 mask、视频开始/结束标记。

Point 仍每窗采2048点，三投影仍使用该窗口全部有效事件。Point 时间归一化不能替代 Memory 的真实窗口间隔。

### 5.2 训练采用连续片段，但保留跨片段状态

建议首版 `tbptt_steps=4`：同一视频的连续4个有效窗口构成一段反向图，在段内保留 Memory 梯度，段末反传和优化；进入该视频下一段前将状态 `detach`，**不重置状态**。

这属于截断反向传播：跨片段保留历史数值、截断梯度；段间优化后状态由之前的参数产生，该近似必须在运行说明中记录。

每个视频开始重置、视频结束释放，不能把一个视频的状态给另一个视频。首版不做随机跳跃片段采样、不额外重复 warm-up 事件、不跨视频训练状态。

每个 epoch 的每个有效窗口恰好作为目标监督一次；B/G 使用与 GM 相同的窗口次序、片段划分、有效样本和优化步安排，不因为无 Memory 就改回另一套随机窗口训练预算。

变长视频的 padding 不进入编码器、BN、loss或投票。一个时刻若有效视频数很少，要安全处理 singleton，不为凑 batch 复制真实样本。全 padding 时刻不进行空网络前向。

### 5.3 视频内几何增强一致

当前原增强先消耗与事件数量有关的随机数，再抽旋转/裁剪。**给不同窗口相同 seed，并不保证旋转/裁剪参数一致。**

在新序列数据层显式拆分“抽取几何参数”和“应用参数”：

```python
params = sample_spatial_params(seed, epoch, video_id)
events = apply_spatial_params(window_events, params, event_jitter_rng)
```

同一 epoch 同一视频的所有片段共享旋转/裁剪几何参数，维持现有参数范围；事件级 jitter 和 Point 采样使用独立随机流。不要宣称重叠事件的 jitter 必然逐点一致。

同窗 Point 与三投影从同一份增强后事件构建。Test不增强。B/G/GM 同步使用这一增强规则，不将它的影响单算作 Memory 收益。

尽量在新文件中实现，不修改旧增强源码的历史行为。空窗口、过滤及边界情况沿用明确且可审计的处理，不静默丢弃监督窗口。

## 6. 权重加载、可训练范围与训练预算

### 6.1 加载完整训练权重，不回到预训练初始化

先构造并严格恢复原 Fusion A，包括 Point、原 attention/adapters、图像参数和分类头，再安装新增模块。

仅处理明确的 `module.`/`fusion.` 前缀；逐项校验原参数，新增参数名单明确列出。不能用无检查的 `strict=False` 吞掉公共权重不匹配。

原 checkpoint、原源码哈希、原清单保持不变；记录新实现版本、权重来源、新增模块初始化和新配置。使用独立目录保存新 checkpoint。

### 6.2 冻结与联训

| 模块 | 本轮状态 |
|---|---|
| 三路 ResNet backbone及原图像FC | 冻结、eval；不更新参数或运行统计 |
| 图像 token adapters与位置映射 | 可训练 |
| EventMamba V2及原Point分类头 | 可训练，使用较小学习率 |
| 原prototype、跨注意力、融合MLP、融合分类头 | 可训练 |
| 新Frame读取、特征Gate、Memory | 可训练 |

原代码已对图像 token adapter 与 backbone 的梯度范围作了区分，不要为了冻结 ResNet 把 adapter 也放进 `no_grad`。

为适配较小的时序局部 batch，首版三组统一冻结 Point 中已有 BatchNorm 的 running statistics；Point BN 的 affine 参数仍可训练。每次 `model.train()` 后重新落实该策略：仅将 BN 子模块切到 eval，其余 Point 模块保留训练模式及 `requires_grad=True`。注意 eval 本身不等于冻结参数梯度；不要把整个 Point 分支误冻结，也不要直接把旧 SyncBN 逻辑复制进变长序列入口。

这一 BN 策略与历史训练不同，但 B/G/GM 共同使用，必须记录，不能把历史 checkpoint 分数当成本轮同预算 B 的替代品。

### 6.3 训练目标

保留原任务目标：

\[
\mathcal L=CE(logits,y)+0.5\,CE(point\_logits,y).
\]

Point 辅助头仍只依赖 Point 特征，不接入图像或历史。新主损失使用 logits 的 CE，不沿用冻结概率混合的训练方式。不要额外新增 Frame 辅助损失或蒸馏损失；本轮保持目标的可归因性。

按有效窗口累计 CE，确保变长视频/尾批没有无意的重加权。单卡可沿用已验证的 `sum_window_loss * steps_per_epoch / total_train_windows` 目标系数，前提是 epoch 排程和步数已确定；B/G/GM 使用同一约定。若增加累积梯度，显式计算有效样本系数，不能每段随意均值后等权叠加。

选模只看主输出的**视频 NLL**，不把 Point 辅助损失混入选模指标。

### 6.4 建议的首轮运行配置

以下是本轮建议起点，不是已经验证的最优配置；先通过短 GPU 资源检查固定，再让三组共同执行，不能按各自 Test 表现临时修改。

```yaml
seed: 0
epochs: 50
video_batch_size_per_gpu: 2
tbptt_steps: 4
optimizer: AdamW
lr_point: 1.0e-5
lr_existing_fusion: 5.0e-5
lr_new_modules: 1.0e-4
weight_decay: 1.0e-4
gradient_clip: 1.0
point_aux_weight: 0.5
precision: FP32
scheduler: none
early_stopping: false
freeze_image_backbones: true
freeze_point_bn_running_stats: true
selection_metric: video_nll
```

50轮均指覆盖完整有效训练集，并非小样本试验。显存不够时优先统一降低 video batch 或片段长度，并记录实际优化步及有效窗口数；不要只给某组减少样本或私自开启 AMP。

不照搬旧输出路由的CPU/80轮配置。已有 GPU Mamba、FPS 与真实事件前向必须使用项目验证过的环境，不能用 CPU 假模块替代正式训练。

## 7. GPU执行与资源安排：不让分布式开发拖慢主模型

首版优先保证**单卡真实训练正确**。若两张卡可用且每组能在单卡运行，可让 B 和 GM 各占一张卡，随后安排 G；每组都训练完整数据，不能把独立实验错误地拆成两份数据。

不把新增完整 DDP 框架作为本轮前置条件。若项目已有可直接复用且经过序列验证的 DDP 支持，可以提供双卡模式，但必须验证：视频/片段分配、相同步数与同步调用、尾批有效样本加权、无重复监督、跨rank不共享Memory。

**不能直接沿用旧随机窗口 sampler，也不能在不同 rank 上用不同次数的 SyncBN/DDP 前向。** 未验收的双卡模式不写为已支持；优先使用两卡跑独立实验来推进。

保留现有事件读取加速。首版不要求先建立新的巨量 token 缓存；Point 与融合层会变化，不能使用脱离当前参数的旧特征替代训练。任何额外图像特征缓存优化都不应阻塞完整模型的正确训练入口。

## 8. 评估、日志与恢复

评估按完整视频有序遍历，首窗重置，后续窗口保留状态；`no_grad` 下可分块处理，但不能每块重置Memory。最终仍对该视频全部有效窗口概率平均。

所有 B/G/GM 使用同一固定评估采样设置和窗口覆盖。历史96.04697%等数字来自特定缓存/FPS随机流，不作为新在线推理必须逐位复现的目标；先记录本轮初始化/在线原模型基线，正式比较以本轮同流程 B/G/GM 为准。

日志最少记录：

- 主/Point窗口CE、窗口Top1、视频Top1/Top5、视频NLL、Brier、固定定义ECE；训练在线指标与固定参数评估分开。
- 各stage g分布、新残差相对H0的范数、Memory读出与状态范数、写门分布；没有历史的窗口单独核验。
- 各优化器参数组学习率、梯度范数、实际监督窗口/视频数、优化步数、GPU显存峰值和窗口吞吐。
- 各组相对B、已有Frame与Fusion+Frame参考的纠错/新增错误；不因NLL改善就宣称Top1全面超过。

保存 `best_video_nll.pt`、`last.pt`、配置与源码身份；主选模规则一致，Top1仅辅助记录。提供真正可用的 epoch 边界 resume：包含新模型全部状态、优化器、数据epoch、随机状态与累计步数，非空目录拒绝意外覆盖。

epoch边界恢复时新视频状态从零开始，不恢复前一epoch的Memory。若未实现中途片段续训，不宣称支持；发生中断从已完成epoch的checkpoint恢复即可。

## 9. 必要验收：一次完成后进入全量训练准备

本批只保留保证正确性的工程检查，不新增多轮“先证明每个小模块有效”的试验关卡。

### CPU/合成测试

1. 原权重严格加载，只有明确的新参数未来源于checkpoint；图像冻结，公共初始参数一致。
2. 新增残差为零时，G/GM的主logits和Point logits恢复原模型；包含单窗及多窗。
3. 独立Frame路径不依赖当前Point特征；Point辅助头不依赖Frame或Memory。
4. 多次小步优化后，原融合、Point、新Frame/Gate以及GM读写路径均有预期梯度；不把首步零梯度误判为失败。
5. 修改未来窗口不影响过去输出；修改当前写入只能影响后续，不改变已经计算的当前预测。
6. 首窗口无伪历史、跨视频隔离、padding不写入、不做全mask NaN attention。
7. 不更新参数时，整段前向与相同状态传递的分块前向一致；TBPTT detach保留数值且截断跨段图。
8. 视频增强几何一致，同窗Point/Frame同步；gap从真实时间戳计算，不因片段开始被改为0。
9. 完整epoch排程无重复/丢失目标窗口；B/G/GM覆盖与步数一致。
10. 视频投票、CE/辅助权重、尾批缩放、保存恢复和架构身份校验正确。

### 一次小批真实GPU验收

使用真实Point/CUDA算子与少量Train视频，完成至少两个连续片段的前向、反向、优化、状态detach及保存恢复；检查单样本尾批、显存和有限值。小批仅是工程检查，不产生方法有效性的结论。

原模型与新增零残差模型的输出使用已经确认的重复性基线和明确FP32容差，不回到Fusion logits零容差要求。参数/buffer不被改动的回归仍严格检查；测试本身的优化步骤与冻结/不变检查要分开。

**这些验收通过后直接准备 B/G/GM 的全量正式训练，不再要求先跑冻结概率版H1—H3、先补更多种子或先做额外输出层诊断。**

## 10. 文件组织与交付

建议采用下面的新增文件，已有可复用组件直接导入；名称可按项目规范等价调整：

| 建议文件 | 职责 |
|---|---|
| `gcstr_feature_memory_fusion.py` | 原token交互包装、独立Frame、特征Gate、Memory；B/G/GM模型构造 |
| `gcstr_video_sequence_dataset.py` | 按视频排程、片段流式读取、显式几何增强、mask与gap |
| `train_gcstr_feature_memory.py` | 真实输入联合训练、TBPTT、参数组、验证、checkpoint/resume |
| `evaluate_gcstr_feature_memory.py` | 完整视频有序评估、普通预测与必要诊断 |
| `test_gcstr_feature_memory.py` | CPU、数学/梯度/状态测试及独立GPU smoke入口 |
| `configs/feature_memory/{B,G,GM}.json` | 三组共同预算、共同来源、结构开关 |
| `FEATURE_GATE_MEMORY_RUNBOOK.md` | 结构、范围、测试与训练/恢复命令 |

不要用“新增模块已经写完”替代真实训练入口和序列数据层交付。本批交接应给出：

- 实际新增/修改文件，模型模块表和参数量，哪部分冻结/训练。
- 来源checkpoint及公共权重加载核对，新模块初始化办法。
- 真实执行的测试及GPU检查；未执行项明确列出，不以skip冒充通过。
- 固定后的三组完整配置、预期窗口覆盖、每epoch步数和可用设备。
- 可复制执行的 B/G/GM 全量训练、完整视频评估及 resume 命令，全部新目录。
- 如检查中发现资源限制，直接给出统一后的可运行配置，不扩展成新的方法研究阶段。

**授权边界：这份任务要求开发与必要验收，不等于无条件启动长时GPU任务。若已有明确训练授权，验收后按授权安排三组；否则一次性回报就绪状态与启动命令。不要擅自占用正在运行的GPU、删除旧产物或自动开始未授权的全量任务。**

## 11. 本轮结论应如何解释

本轮是完整特征模型的候选实验，不保证比Frame集成或原G2更好。B验证新预算/流程的影响，G验证独立Frame+门控这一组合，GM验证在相同新结构上加入历史特征的增量。

如果GM优于G，仍需区分性能和代价，并在后续必要消融中核验历史作用；若不优于G，不要继续无依据地扩容。最终保留哪个结构由本轮完整结果决定，不预设Memory必须保留。

第二数据集预留 `num_classes`、传感器/归一化约定、自然序列边界及协议适配，不写死UCF标签或路径。但本次不同时重做THU等数据流水线，也不把其他数据集准备作为UCF完整模型训练的前置条件。

---

## 源码与结果依据（供Codex定位，不是要求重做的任务）

核对基线：`CelestineYang/event-vision-research`，提交 `5a1b0cf0a11775d7249b52174214dd76cbbb0493`。执行前检查当前工作区及后续提交，不覆盖用户的新修改。

- `gcstr_secnet_fusion.py`：`PrototypeStage`、池化前交互、图像adapter与冻结范围。
- `gcstr_eventmamba_fusion.py`、`ucf101_pretrained_fusion.py`：Point/Frame编码、分类头和完整Fusion接口。
- `train_ucf101_pretrained_fusion.py`：已验证的原优化目标、Point辅助权重、原checkpoint与随机状态管理。
- `ucf101_fusion_window_data.py`：同窗事件/Point/投影、2048采样及官方归一化。
- `gcstr_augmentation.py`：当前空间增强的参数范围及随机数消耗顺序。
- `research_exports/ucf101_routing_stage2_2026-10-23/`、`research_exports/ucf101_routing_stage3_2026-10-24/`：已有简单集成、动态Gate及Memory诊断结果。

**一句话目标：这次直接交付并准备训练“特征级 Gate + 融合特征 Memory”的完整模型，不再用新的冻结输出小实验推迟主模型开发。**
