# 第三批修改任务：解除固定外层权重约束，并对齐视频级训练目标

## 0. 核对基线与本批结论

本任务基于 GitHub 提交 `e19ff10f183d5e0e7e3394292bb4711d795a3b1c`（`Add stage-two routing diagnostics and completed multiseed results`）。执行前查看实际工作区：保留此后已有修改，不回滚、不重复开发已实现内容。

本次决策依据：

- `FUSION_ROUTING_STAGE2.md`
- `research_exports/ucf101_routing_stage2_2026-10-23/README.md`
- 同目录 `summary.json`
- 同目录 `suite/analysis.md`
- 同目录 `static_frame_comparison/analysis.md`
- `fusion_router.py`、`routing_stage2.py` 及现有训练/评估接口。

第二阶段已经完成22项同缓存评估、Static Fusion+Frame对照和三种动态模型的三种子训练。不要再次当作待办。可选前缀评估未做，但它不是本批修改的阻塞前提。

已观测结果（原Test用于选模；非独立测试）：

| 方法 | 视频Top-1 | 视频NLL | 说明 |
|---|---:|---:|---|
| Fusion A，同缓存 | 96.04697% | 0.35925712 | 既有专家epoch40 |
| Frame：三个图像头概率平均 | 97.103718% | 0.28017991 | 无学习 |
| Fusion+Frame，固定等权 | 97.025440% | 0.21757199 | 无学习 |
| Static Fusion+Frame | 96.2035% | 0.19364310 | seed0，选中epoch8 |
| Current，三种子 | 96.3209%（三次相同） | 0.196445 ± 0.002217 | 样本标准差 |
| GRU，三种子 | 96.3209%（三次相同） | 0.196322 ± 0.001016 | 样本标准差 |
| Conditional，三种子 | 96.3209%（三次相同） | 0.197241 ± 0.000892 | 样本标准差 |

九个动态模型相对Fusion的纠错集合相同，均纠正7个、无新增错误；这不意味着所有错误视频上的错误类别也完全相同。seed0重置GRU/Conditional历史不改变Top-1，NLL略改善。Current+EMA亦未优于Current。

当前动态模型只学习alpha，eta固定为0.09410769492387772，因此约90.5892%的显式概率混合权重始终留给原Fusion A。17维输入包含当前预测状态、输入统计和真实时间间隔。现有训练目标为有效窗口平均NLL，视频输出则是窗口概率平均后计算视频指标。

由此提出两个待检验假设（不是已经证明的原因）：

1. 只调整alpha、固定eta，可能限制了独立Frame证据改变最终决策的能力。
2. 窗口级监督与视频级聚合存在目标差异，可能影响门控训练。

Frame准确率更高但NLL更差；Static Frame旁路NLL更低但准确率更低，均不是所有指标上的全面优势。不能把释放eta或更换损失写成必然提升。

**本批开始改变小模块结构和训练目标，不再重复第二阶段的诊断矩阵，也不直接解冻专家或加大Memory。**

## 1. 执行范围及必须保持的内容

完成新增代码、配置、CPU单元测试和合成缓存smoke；提供真实运行的独立命令。未经本批额外明确授权，不自动启动全量训练、重新导出五头缓存或GPU专家训练。

沿用：

- 原Fusion A epoch40及其已有五头证据缓存；不重新编码、不重新FPS、不重新选择专家checkpoint。
- 原Train全部有效视频拟合；原Test作为gate_val，按视频NLL选checkpoint；不拆Train、不重新随机划分，不引入OOF前置要求。
- `selection_protocol=original_test_as_val`、`independent_test=false`；标准化器和任何可训练参数只由原Train拟合。
- 第一阶段unrestricted Static的选中checkpoint（epoch10）作为共同初始化，不改用第二阶段alpha固定1的Static Frame对照。
- 已有17维观测及同一Train标准化器；不增加标签、真实类别概率、错误状态、视频/类别身份、未来统计作为输入。
- 原始窗口、Point归一化、三路Frame概率平均、FP32、视频内有效窗口概率平均的评估方式。
- 原专家、现有缓存、checkpoint、研究快照和正在运行的其他数据集任务均不改写。

原源码哈希受到checkpoint校验保护，优先新增：

```
fusion_router_v3.py
train_fusion_routing_v3.py
evaluate_fusion_routing_v3.py
test_fusion_routing_v3.py
configs/routing_stage3/*.json
FUSION_ROUTING_STAGE3.md
```

已有功能可以导入复用。不要为新功能修改`fusion_router.py`、`routing_data.py`、`train_fusion_routing.py`的历史行为或删除哈希校验。数据缓存schema仍是2；新模型采用独立architecture/model_schema，不将新权重伪装成旧Router checkpoint。

## 2. 新增可配置的外层Gate eta

继续使用原来的概率结构：

```
pF_t = (pXY_t + pXT_t + pTY_t) / 3
pB_t = alpha_t * pF_t + (1 - alpha_t) * pP_t
pFinal_t = (1 - eta_t) * pA_t + eta_t * pB_t
```

`alpha_t`仍由当前窗口MLP生成：

```
alpha_t = sigmoid(b_alpha + delta_alpha(o_t))
```

首轮保留原Current的17→64→1结构及激活；不引入新的历史模块。

新增`eta_mode`，明确区分：

| eta_mode | 实现 | 目的 |
|---|---|---|
| fixed | 使用静态checkpoint中的固定eta | 旧Current的回归对照，不默认重训 |
| global | `eta=sigmoid(b_eta)`，b_eta为可训练标量 | 检验重新学习总体混合比例 |
| current | `eta_t=sigmoid(b_eta+delta_eta(o_t))` | 检验逐窗口外层选择是否必要 |

`current`的eta可采用独立17→64→1小MLP，输入与alpha相同；`global/current`中的b_eta均参与训练。不要将全局重标定和动态选择混为一种模式。首版不增加温度、Top-k、硬路由、门控均衡、手工eta下限或偏向Frame的额外损失。

注意：保留原Fusion路径不代表强制其永久占90%以上。本批允许模型在正常sigmoid范围内学习eta，不根据Test表现人工设成0.5或0.9。

### 2.1 共同初始化与受控比较

- 从原unrestricted Static `router_state.alpha_logit`直接恢复b_alpha（冻结参考值），不要经过概率截断再求logit。
- 从其原始`eta_logit`恢复b_eta；fixed使用同dtype sigmoid结果，global/current保留原始logit作为参数。
- 两条动态修正路径的最后一层都零初始化，初始完整概率/log-probability/NLL应等于同一个原Static。
- 所有新实验从这个共同静态起点训练；不让某个方法先加载训练好的Current，再比其他从Static开始的方法。
- 初始化公共alpha路径时使用独立或固定的初始化随机流，避免多构造一个eta网络就改变公共层初始参数。
- 保留有限logit浮点饱和与显式精确端点的区分；eta同样不能由于恢复路径错误被意外冻结。
- 不重新拟合标准化器。严格加载Static的原Train标准化状态及观测定义。

### 2.2 接口与数值

保持输入兼容`logits[B,T,5,C]`、`observations[B,T,17]`、mask、dt。
输出至少包含`log_probs`、逐窗口alpha/eta和三条路径权重。首轮无历史状态，state为None即可，保留后续可扩展接口。

现有`mix_log_probs`已能接收逐窗口eta，应复用其稳定算术概率混合；不能平均logits、混成乘积专家，或另行对概率执行softmax。

路径权重：

```
w_A = 1 - eta
w_F = eta * alpha
w_P = eta * (1 - alpha)
```

检查非负且和为1，说明是显式预测路径权重，不是整个网络的因果模态贡献。

## 3. 新增视频级训练目标，保留窗口级对照

首轮各可训练方法统一采用完整视频批次；Gate无Memory也一样按视频组织。有效窗口仍按真实时间排序，padding不计入任何目标。

新增`loss_mode`：

### window_nll：保持旧目标

所有有效窗口等权：

```
L_window = sum_{v,t valid} -log pFinal_vt[y_v] / sum_v T_v
```

兼容原有batch目标缩放与日志，作为结构变化的受控对照。

### video_nll：先聚合概率，再监督视频

```
log pVideo_v[c] = logsumexp_{t valid}(log pFinal_vt[c]) - log(T_v)
L_video = mean_v[-log pVideo_v[y_v]]
```

- 对每个视频先取窗口概率平均，不是平均logits，不是平均窗口NLL。
- 聚合必须处于梯度图内；不能调用现有`@torch.no_grad`评估器、detach、转NumPy或用已缓存的视频预测训练。
- 使用mask后的稳定logsumexp；每条视频至少一个有效窗口，标签在视频内保持一致。
- 默认视频等权；批次尾部的加权与整个epoch目标明确记录。若沿用现有epoch尺度法，video目标使用`sum_video_loss * steps_per_epoch / videos_per_epoch`，不能套用窗口总数系数。
- Train/Test仍使用同一概率投票定义；Test仅评估和选模，不反向传播。

### video_balanced_window_nll：条件性使用的归因对照

```
L_balanced_window = mean_v[mean_{t valid}(-log pFinal_vt[y_v])]
```

它与video_nll同为视频等权，但聚合顺序不同。实现并准备配置，但不默认启动。

若video_nll优于旧window_nll，要用此对照区分视频重加权与监督聚合的作用；未做这个对照时，不得声称收益全部来自“先聚合再监督”。

首版不添加联合损失权重、标签平滑、蒸馏、对比损失或直接优化不可微Top-1。

## 4. 明确且有限的首轮训练矩阵

旧Current（fixed eta+window_nll）直接复用已有结果，仅用新接口做回归；不要重新跑一遍。

本批提供四个主要运行配置：

| ID | alpha | eta_mode | loss_mode | 比较问题 |
|---|---|---|---|---|
| G1 | current | global | window_nll | 固定外层权重是否限制表现？ |
| G2 | current | current | window_nll | 动态eta是否优于只学全局eta？ |
| G3 | current | global | video_nll | 相同总体外层Gate下，视频监督是否有帮助？ |
| G4 | current | current | video_nll | 动态eta与视频监督是否有互补作用？ |

这四项构成2×2受控设计，不要只做G4并把全部增益归于双门控。

首轮与第一阶段保持80epochs、batch32视频、Adam lr1e-3、wd0、clip1、seed0、FP32，无早停。每个epoch完整覆盖Train，不改变数据增强（继续复用无增强缓存）。所有主配置具有相同优化步数，并记录可训练参数量。

新增模型的epoch0输出单独记录，作为初始化和退化检查；默认不参与epoch1至80的最优checkpoint选择。相同epoch0规则用于全部配置。

选模指标继续固定为原Test视频NLL；同时保存Top-1、Top-5、Brier、ECE、最后一轮指标及完整曲线。不能看到某项Top-1更高就为该方法临时改用Top-1选模。

`video_balanced_window_nll`配置仅用于video目标出现增益后的归因对照；Memory扩展、多种子也不在本次自动组合成大矩阵。

## 5. 定向日志：确认“有自由度”是否真的转化为有效选择

不要重跑22项旧suite作为新实验。复用已有基线和配对分析工具，增加以下针对新Gate的数据：

- 逐epoch的Train与选模集alpha/eta及w_A/w_F/w_P均值、分位数、饱和比例。
- eta MLP输出的标准差：识别真正的样本自适应与近似常量缩放。
- alpha/eta参数梯度范数、训练和选模损失、是否迅速退回eta接近0。
- 每个视频的概率、预测和NLL；相对Fusion、Frame、固定Fusion+Frame、原Current分别报告纠错/新增错误。
- 两模型同时正确、同时错误和各自单独正确组的NLL贡献，复用第二阶段计算逻辑。
- 保留真实权重轨迹开关；只在显式开启时保存大型逐窗口轨迹。

当前专家在原Train上训练过。扩大eta自由度并不能自动消除在样本内专家输出上训练门控的分布问题。请从已有Train缓存计算一次各专家和固定组合的Train性能，与已有选模集结果并列呈现；若Train上Fusion明显占优而gate不断回到Fusion，要如实记录，不能手工将它推向Frame或修改数据来源。

这些统计仅用于训练后的分析；真实类别概率、正确性、视频ID不得成为推理输入。

## 6. 评价标准与Memory后续位置

准确率方面，主表必须包括97.1037%的Frame和97.0254%的固定Fusion+Frame，不得只与96.0470%的Fusion或96.3209%的旧Current比较。
概率质量方面，保留Static Fusion+Frame的NLL 0.193643作为强对照。旧值均需保持同缓存、同覆盖、同选模规则的身份说明。

如果新方法只改善NLL但准确率仍低于Frame，报告为概率质量/准确率权衡；不表述为全面超越。当前数据曾用于方法开发和选模，不将新比较称为独立测试。

本批不新增更复杂Memory、不自动联合微调、不扩大状态维度。为后续保留`alpha/eta`两路输出接口：当无历史双门控确有稳定收益后，再在相同训练目标、输入、初始化和预算下比较无历史、GRU、Conditional；若那时Memory同时影响alpha与eta，必须明确这是新架构，不直接拿旧的只调alpha模型当唯一对照。

本轮现象只说明当前Memory设置没有额外收益证据，不证明事件时序记忆普遍无效。不要将“阶段三”解释成必须启动Backbone大规模训练。

## 7. 版本、保存恢复与资源

- 从真实服务器sources解析路径，核对完整缓存和checkpoint哈希；快照不是模型权重或logits数据。
- 新文件采用独立code_identity/architecture/model_schema，记录旧专家、Static、观测、数据角色及新代码身份。
- 严格读取旧Static时保留原有loader和校验。新模型保存恢复使用新loader；不改写或去除旧源码哈希。
- 新模型恢复不能只靠config中alpha0/eta占位值：raw logit、buffer、MLP、normalizer及优化器状态均从真实state恢复。
- 禁止隐式恢复训练或覆盖旧输出；首轮提供明确fit/eval命令即可，不必为省事引入不完整resume。
- CPU或显式单GPU训练均可，先做小批性能检查；无需DDP、数据搬迁或专家前向。
- 缓存schema、17维观测和归一化不变，故本批无需重新导出全量专家证据。
- 后续其他数据集读取自身dataset_id/classes/roles；不能强制套用UCF的Test-as-Val或把不同语义的同名类别直接混合。

## 8. 必须通过的测试

1. **静态起点等价**：fixed/global/current eta的初始化与共同Static完整预测和NLL一致；含接近边界、有限logit饱和及序列padding。
2. **旧Current回归**：新fixed模式可以显式复制旧Current参数并复现其前向；这个兼容测试不等于让新实验从旧Current预训练。
3. **eta梯度与冻结**：global仅标量eta有梯度，current标量及eta网络有梯度，fixed eta无梯度；专家和原Static均不变。dynamic零输出初始化时，第一步eta第一层梯度可能为零，测试应允许预期的初始化现象，确认输出层可更新且随后前层可学习。
4. **数学端点**：eta=0回到Fusion，eta=1回到Frame/Point组合；alpha=1回到Frame旁路，alpha=0回到Point旁路；权重非负且和为1。
5. **视频目标**：手工概率例子验证video_nll、window_nll、video_balanced_window_nll数值；video_nll对窗口logits/门控保留梯度。单窗口时三者一致；同一batch有变长视频时处理正确。
6. **聚合次序**：同一数据、同一视频权重下，video_nll不大于video_balanced_window_nll（容许数值误差），防止误实现成平均log或平均logits。
7. **padding/空序列/标签**：添加padding不改变loss和梯度；空有效视频显式失败；同视频标签不一致报错；不丢尾批。
8. **训练权重**：不等长视频与尾批的epoch目标系数按各loss定义验证，不混用视频数和窗口数。
9. **身份与来源**：协议、缓存、专家、观测及类别不匹配明确拒绝；Test不能进入梯度更新和normalizer.fit。
10. **保存恢复**：预测、实际alpha/eta、normalizer和优化器状态恢复一致；新模型不误加载成旧Router。
11. **合成smoke**：四个主要配置各用微型合成缓存训练若干步并评估，验证梯度、配置和产物，不启动真实80轮实验。

## 9. 本次交付与停止边界

请交付修改文件、实际测试命令/结果、各配置参数量与初始化等价报告，以及：

1. 旧Current回归与输入身份只读检查命令。
2. G1至G4真实fit/eval命令（各自新目录，不自动执行）。
3. 视频等权窗口NLL条件性对照的命令。
4. 统一结果表/配对分析命令。
5. 清楚标出已经执行的是CPU/合成测试，尚未执行的是哪些真实训练。

本次不再实现已完成的第二阶段suite，不重训专家，不改变原划分，不为使Memory“有收益”而增加大量新模块。先用这一轮受控的小模块训练判断：外层权重自由度和视频监督是否真的解决当前决策瓶颈。
