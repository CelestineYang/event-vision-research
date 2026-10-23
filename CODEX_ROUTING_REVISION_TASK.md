# Codex 第二批修订任务：Gate–Memory 框架修复与原 Test 选模协议

请在现有第一批代码上完成下面的修订。项目主线保持为 **UCF101-DVS：预训练 gCSTR 三投影 + EventMamba V2 + Prototype Fusion A**。

**用户已经确认：沿用已有 Train/Test 划分，不再从 Train 中划分验证集；原 Train 全部用于 Gate/Memory 训练，原 Test 继续用于验证和 checkpoint 选择。** 本任务替代此前“必须从 Train 另划 gate_val”和“原 Test 一律禁止用于选模”的要求。参数拟合、标准化拟合仍只使用原 Train。

本文以仓库提交 `1ae665eed073328400ccdd255ddd71a72e39d27d` 为核对基线。若当前工作区已有后续修复，请检查后复用，不重复实现或回退用户修改。

## 一、本轮执行范围

本轮完成：代码修复、配置与角色映射工具、文档、CPU 单元测试和合成缓存上的小型训练 smoke。若服务器已有真实审计清单，可以读取清单并生成角色映射及数量报告。

本轮不自动执行全量真实数据导出、正式 Gate/Memory 训练、真实权重 CUDA 回归、基础模型重训或联合微调。真实权重回归和后续实验提供单独可执行命令，交接时标明是否实际执行。

保留原 Backbone、Point 点数、投影、归一化和 Cross-Attention 结构；保留旧训练入口、已训练权重、原划分、原缓存和实验目录，不干扰正在运行的任务。新增输出写入新目录。保留源码、配置、权重和数据身份校验，不通过 `strict=False`、伪造来源或删除哈希检查解决兼容问题。

## 二、对截图中四项待确认事项的正式答复

| 截图中的事项 | 本次已经确定的处理 |
|---|---|
| 没有自动划分或填写角色模板 | 现在按已有清单生成角色映射：所有原 Train 的有效视频为 `gate_train`，所有原 Test 的有效视频为 `gate_val`。这是用途映射，不是重新随机划分。 |
| 默认拒绝专家已见或来源未知的 Gate 验证数据 | 为本阶段新增显式的原 Test 选模协议，允许该协议下的已见/未知验证来源；保留实际接触信息，不改成 `unseen`。默认严格模式可以保留。 |
| 原 Test 始终禁止用于 Gate 拟合、标准化或选模 | 将“参数拟合”和“选模”分开：原 Test 允许评估及 checkpoint 选择；仍不进入静态权重、Gate、Memory 的梯度训练，也不拟合标准化参数。 |
| 官方 Point 初始化权重缺少可核验的逐视频训练来源 | 继续记录为 `unknown`，无需因此重训。分别记录官方初始化来源和已知的 Fusion A 微调/选模来源；缓存 Point 输出来自完整恢复后的同一个 Fusion 模型。 |

当前没有额外独立 Test。本阶段统一记录 `independent_test=false`，不要通过角色名称声称新增了独立测试集合，也不要再要求用户先拆 Train 才能继续完成本轮工作。

## 三、原 Test 选模协议需要贯穿整个流程

### 3.1 配置和身份

建议新增统一配置，例如：

```json
{
  "selection_protocol": "original_test_as_val",
  "train_source_split": "train",
  "train_role": "gate_train",
  "selection_split": "test",
  "validation_role": "gate_val",
  "selection_metric": "video_nll",
  "independent_test": false
}
```

上述 `selection_protocol` 是本次待实现的配置示例，不是现有 CLI 已支持的参数。实际 CLI 命名可与现有项目统一，但语义必须明确。

- 原 Train 的 `source_split` 保持 `train`；原 Test 的 `source_split` 保持 `test`。
- `role`、原始来源、增强模式和采样状态继续分开管理。
- 角色文件、导出参数、缓存记录、训练配置、checkpoint 和结果报告均记录采用的协议。
- 显式选择本协议后，应完成对应的来源校验，不应还需要用户再确认同一件事。保留已见/未知来源的记录。
- 默认严格行为继续存在；现有 `--allow-seen-expert-validation` 的含义不能被悄悄扩大后仍沿用旧帮助文本。

### 3.2 两个阻断点必须一起修改

1. **`export_fusion_evidence.py::role_mapping()`** 当前拒绝原 Test 映射为 `gate_train` 或 `gate_val`：
   - 任何协议下都拒绝原 Test 映射为 `gate_train`；
   - 仅在显式启用本协议时，允许原 Test 映射为 `gate_val`；
   - 保留重复记录、同一视频跨角色、错误原始来源等检查。

2. **`routing_data.py::validate_training_roles()`** 当前同时禁止训练和验证来源中出现 Test：
   - 训练路径始终拒绝原 Test；
   - 验证路径在本协议下允许原 Test；
   - 保留专家身份、类别映射、训练/验证相同视频不重叠等检查；
   - 验证已见/未知来源应依据明确协议允许，不通过伪造 `unseen` 通过检查。

3. **`ObservationNormalizer.fit()`** 继续只允许原 Train 的 `gate_train` 数据。不要因允许 Test 选模而放开标准化拟合。

同步修改 `train_fusion_routing.py` 的 CLI、配置保存、checkpoint 恢复及评估日志，避免“导出允许但训练拒绝”或“命令能跑但结果没有协议记录”。

### 3.3 生成完整角色映射，不重建划分

- 以现有经过审计的有效视频/窗口清单为准，并与 `train_ucf.txt`、`test_ucf(1).txt` 的原始归属一致。
- 不重新随机 8:2，不按窗口切分角色，也不使用 DailyDVS-200 的 `gcstr_splits_seed0.csv` 代替 UCF101-DVS 清单。
- 不直接把原始文本中所有视频都送入导出器：无有效窗口、已被既有审计筛除的视频应在报告中说明。
- 全部原 Train 有效视频进入 `gate_train`，全部原 Test 有效视频进入 `gate_val`；同一视频全部窗口继承同一角色。
- 保留原始清单和哈希；角色文件写到新文件，例如 `configs/fusion_routing_roles.original_test_as_val.json`。
- 给出原始/有效视频数、有效窗口数、类别覆盖、缺失/重复记录、相同视频交集和已有 class/group 重叠报告。不得新增“group 必须互斥”的运行门槛去改变用户已选定的划分。

仓库当前审计摘要可作为核对参考：

| 来源 | 原始视频 | 有效视频 | 有效窗口 |
|---|---:|---:|---:|
| Train | 10,819 | 10,296 | 108,066 |
| Test | 2,704 | 2,555 | 27,384 |

这些是已上传摘要中的历史数量，不是新代码应硬编码的常量。真实清单不同时先报告版本与原因，不修改样本去凑数。若工作区缺少真实有效清单，仍完成生成工具、模板和合成测试，并明确哪些真实统计尚未生成。

### 3.4 专家来源和首轮共同 checkpoint

- 首轮沿用现有按视频 NLL 选择的 Fusion A `best_loss.pt` 作为共同专家基准。
- 当前索引对应目录为：
  `/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints`。
  服务器执行时确认文件及身份；本任务不假设 GitHub 已上传权重。
- 不同 Gate/Memory 模式共用同一完整 Fusion checkpoint、同一五头缓存、同一角色映射。
- 官方 Point 初始化的逐视频来源继续为未知；已知的 Fusion A 在原 Train 微调、在原 Test 选模等历史分别记录。
- 原 Test 的专家选模接触状态继续为 `seen`；其专家拟合接触状态不能仅凭 `source_split=test` 就推定为 `unseen`。
- 不能只调用初始化权重加载器来代替完整 Fusion A 恢复，不能把独立官方 Point checkpoint 的输出混入这组五头缓存。

## 四、修复静态组合到动态 Gate 的初始化

涉及 `fusion_router.py` 和 `train_fusion_routing.py`。

目前静态权重先转为概率，再经 `inverse_sigmoid()` 截断到 `[1e-6, 1-1e-6]` 后重建动态起点。静态学到约 `1e-8` 的 alpha 时，动态初始化会被改成约 `1e-6`，不再与静态组合等价。

请修改为：

1. 从静态 checkpoint 的 `router_state["alpha_logit"]` 直接复制原始有限 logit 到动态 `base_logit`，不走概率截断后的逆变换。
2. 动态 `base_logit` 和 eta 继续固定。eta 从静态 `eta_logit` 在一致 dtype 下计算后写入 `fixed_eta`；不能把 logit 本身当概率。
3. 同步 `fixed_alpha`、EMA 初始门控和 padding 回退值，使它们与共享静态起点一致。
4. 保留修正输出零初始化；`current/gru/conditional` 初始完整预测应在相同 dtype 的数值容差内等于同一静态组合。
5. 区分“用户显式指定的精确 0/1”与“有限原始 logit 的 sigmoid 在浮点数下饱和到 0/1”。后者不能被 `alpha0 in (0,1)` 误判为永久关闭动态修正。
6. 保存及恢复以真实 `router_state` 为准，保留静态 checkpoint 哈希、缓存/角色/专家身份和初始化来源；加载后仍应保持等价。
7. eta 接近 0 等修正影响极小的退化情形继续记录，不改写成有效的 Memory 改善证据。

测试覆盖普通权重、约 `1e-8`、接近 1、显式精确端点和有限 logit 饱和；比较完整 log-probability 和 NLL，不只比较 Top-1。alpha 端点测试必须包含 eta 大于 0 的情况，避免所有输出退化为 Fusion A 后掩盖错误。另测保存/恢复、EMA 初始输出及固定 buffer 不进入优化器。

## 五、统一时间间隔信息，保证动态模式对照公平

首轮采用一个明确方案：**`current/gru/conditional` 都使用相同的低维观测，其中统一包含 `log1p(dt_seconds)`。** 不再只让 conditional 获得时间间隔，也不新增多组结构搜索或正式消融任务。

### 5.1 先按视频计算真实 gap

- 按视频有效窗口的真实 `start_us` 排序，计算与前一个有效窗口的起点间隔，转换为秒。
- 每个视频的第一个有效窗口 gap 为 0；不是每个 batch 或分块开头都重置为 0。
- 当前窗口模式把 dt 固定为 0，需修复：窗口模式与序列模式读取同一预计算 gap 映射，再执行窗口打乱或视频组批。
- 改变 shuffle、batch 大小或组批顺序，不得改变同一窗口的 gap。
- 不假定间隔始终是 0.5 秒；跳过无效窗口后使用真实间隔。

### 5.2 共用观测构造和标准化

- 可将 gap 特征加入当前 `OBSERVATIONS` 末尾，使当前观测从 16 维变为 17 维；配置记录特征顺序、秒单位和变换。
- 先对原始非负 gap 执行一次 `log1p`，再按 Train 的均值/尺度标准化。
- 原始 dt 仍可单独保留为元数据。不要对已经标准化、可能为负的 gap 再执行 `log1p`。
- Dataset 和 `ObservationNormalizer.fit()` 必须使用同一个完整观测构造函数和 gap 映射；不能一边拟合 16 维、一边输入 17 维。
- 三种动态模型的输入维度同步更新。conditional 已从共同观测获得 gap 后，删除原来的额外 gap 拼接，不能重复输入两次。
- `current` 仍无历史隐藏状态；它只增加当前窗口可获得的时间间隔，不接收过去分类结果、标签或未来统计。

### 5.3 保留因果性并更新身份

- conditional 仍先读取旧 Memory 生成当前门控，再更新供后续窗口使用的状态。
- GRU 保留既定的当前输入更新与读出方式，文档准确说明与 conditional 的差异。
- 视频之间重置状态；padding 不更新状态、不参与损失或投票。
- 三种模式使用同一观测定义、静态起点、标准化器、数据、训练预算和选模规则，记录实际参数量。
- 更新观测 schema、配置和 checkpoint 身份检查。旧 16 维 Gate checkpoint 不可静默当成新版本加载。
- 五头原始 logits/事件统计的语义不必因此改变；若考虑复用旧缓存，必须明确检查其元数据与身份兼容性，不能绕过现有哈希校验。

新增测试：同一窗口的窗口/序列模式观测一致；真实不等间隔保留；改变 shuffle/batch 不改变 gap；非零读出权重时三种动态模式都具有使用 gap 的通路；零初始化仍等于静态组合；修改未来时间或观测不影响先前输出。零初始化时输出不随 gap 改变是预期行为，不能据此判断时间输入无效。

## 六、修复 spawn 多进程导出

涉及 `export_fusion_evidence.py::make_export_dataset()`。当前 `ExportDataset` 是函数内局部类，`spawn` 时不能按当前方式序列化。

- 将 Dataset 改为可导入模块的顶层类，构造参数显式接收依赖，不再捕获工厂函数闭包。
- 推荐新增一个小型 Dataset 模块，再由导出器延迟导入；这样可以保留导入时不初始化模型/CUDA的行为。
- 不修改原 `FusionWindowDataset` 或模型输入协议来处理这个问题。
- 新增模块纳入新框架代码身份记录；保留原模型源码签名检查。
- 保持真实来源、显式关闭增强、固定采样 epoch/seed、原 Point 归一化和五头对齐。
- 若采用组合包装，确保外层 `rows` 过滤和 `--max-videos` 裁剪真实作用于内层采样索引，避免元数据与输入错位。

至少用真实导出 Dataset 类和小型可序列化 fixture，执行一次 **CPU `spawn + num_workers=1` 的 DataLoader 取 batch**，并与 workers=0 对照样本身份、窗口、元数据和事件统计。仅对一个无关玩具类做 pickle 测试不构成本项验收。

首轮真实权重回归命令仍默认 workers=0；本轮只提供命令，不自动运行真实权重/CUDA检查。

## 七、移除重复标准化拟合

涉及 `train_fusion_routing.py`。

目前动态训练会先 `ObservationNormalizer().fit(train_data)`，随后加载静态标准化器覆盖，造成无效扫描。

- static 阶段在原 Train 的 `gate_train` 上拟合一次。
- current/gru/conditional 创建标准化器后，直接严格加载对应静态 checkpoint 的 `normalizer_state`，不再次调用 fit。
- 标准化器与第五节的新共同观测定义保持一致。
- 保留静态起点、缓存、角色、专家身份及观测 schema 的一致性检查。
- 添加测试确认动态训练未调用 fit，恢复参数与静态完全一致，Test/gate_val 仍不能用于拟合标准化。

## 八、修复分析脚本和过时文案

涉及 `analyze_ucf101_complementarity.py`。

- 固定写入的“NLL 更差但准确率更好”等结论，改为根据当次差值生成改善/持平/变差描述，并给出数值。
- Point 的 unique rescue 应报告实际数量和比例，不再无条件输出“很少”等判断。
- 错误集合为空时，不能访问 `cases[0]` 获取 CSV 字段；显式定义字段名，仍正常输出表头和空结果。
- epoch 6、epoch 40、latest 等 checkpoint 描述从本次输入元数据读取；没有元数据时说明未知，不复制历史事实。
- 删除或更新固定要求“用 Train 的 out-of-fold 预测训练 Gate”的文案，明确本阶段已经选择“原 Train 拟合、原 Test 选模”，不暗中新增 OOF 导出或专家重训要求。
- 保留 oracle 使用真实标签、不能作为可部署 Gate 成绩，以及视频级 CSV 不能替代逐窗口 Gate/Memory 缓存的说明。

测试至少覆盖零错误、指标改善、指标变差、持平和 checkpoint 元数据缺失。

## 九、保留已实现的核心行为作为回归要求

以下内容不是要求重写，而是本轮修改不能破坏：

1. 完整严格恢复 Fusion A；五头来自同一窗口及同一已恢复模型，Point 与 Fusion 共用一次 Point/FPS 计算；图像 logits 使用原 ResNet 平均池化和 fc。
2. 专家全部冻结并处于 eval，参数与 BN 不更新；导出不伪造 source_split 来关闭增强。
3. 保存五头完整 logits、事件统计、时间戳、类别映射和身份记录；原子完成标记、重复/缺失/非有限值校验保持有效。
4. 概率按算术方式混合：
   `p_F=(p_XY+p_XT+p_TY)/3`；
   `p_B=alpha*p_F+(1-alpha)*p_P`；
   `p_final=(1-eta)*p_A+eta*p_B`。
   三路径等权仍为 alpha=1/2、eta=2/3；温度保持 1；NLL 在 log-space 稳定计算。
5. 训练损失为有效窗口平均 NLL；评估先窗口内融合，再对同一视频有效窗口概率平均。
6. 模式保留 fixed、static、current、ema、gru、conditional，以及已有 Fusion A+Frame 静态对照。动态方法共用一个不受该限制的静态起点。
7. 固定预算下所有方法统一按原 Test 的视频 NLL 选择 checkpoint，同时报告窗口/视频 Top-1、Top-5、视频 NLL、Brier、固定定义 ECE、相对 Fusion A 的纠错/新增错误/净变化，以及门控分布、样本数、步数和参数量。
8. 不为个别方法临时切换选模指标；EMA 系数首轮预先固定并记录，温度不新增搜索。
9. 任意类别数、小类别数 Top-k、不等长序列、单窗口视频、padding 和跨视频状态隔离继续支持。

## 十、最小验收清单

| 类别 | 必须验证的行为 |
|---|---|
| 协议默认行为 | 未启用新协议时仍按原严格约束处理 Test-as-Val |
| 显式 Test-as-Val | 原 Test 可映射为 gate_val 并完成导出路径及选模路径；配置/结果记录 independent_test=false |
| 训练边界 | 任何协议下原 Test 不能变为 gate_train，也不能拟合标准化器 |
| 角色完整性 | 原 Train/Test 的有效视频完整映射，无二次随机切分，无相同视频跨角色，同视频全部窗口同角色 |
| 来源记录 | Test 的选模接触保持 seen，官方 Point 无法核验的拟合来源保持 unknown |
| 静态初始化 | 普通/极小/接近 1/精确端点/浮点饱和，完整预测与 NLL、恢复和 EMA 初始状态正确 |
| 公平时间输入 | 三种模式共享同一 gap；窗口与序列一致；shuffle/batch 不改变 gap；不重复变换或拼接 |
| 标准化 | 只在 Train 拟合；动态直接恢复且不重复扫描拟合 |
| 多进程 | 实际 ExportDataset 通过 CPU spawn worker 取 batch，与单进程身份/元数据一致 |
| 因果与状态 | 未来不影响过去，padding 不更新状态，视频间隔离 |
| 数学与通用性 | 概率归一化、权重端点、等权、NLL/梯度及非 101 类支持 |
| 分析脚本 | 空错误集合和不同指标方向均正常输出，不带历史固定结论 |
| 回归边界 | 原模型接口/真实权重 CUDA 回归有独立入口，明确是否实际执行，不把 skip 算作通过 |

默认实际运行 CPU 单元测试，以及新协议下 static/current/gru/conditional 的合成训练 smoke。fixed 和 EMA 运行对应合成评估检查即可。合成数据应包含不同长度视频、不等时间间隔和 padding；不能只用所有视频完全等长的例子验收。

如果环境缺少某项依赖，明确报告具体未执行项，不能把整项标为通过。真实权重、真实全量数据和 CUDA 回归本轮不自动执行。

修改将改变新框架源码哈希。若已有旧版本 Gate checkpoint，采用明确拒绝或可记录的受控迁移；不删除哈希校验以兼容。必要时重新生成小模块实验不等于重训现有专家，本轮工程测试使用新版本合成 checkpoint 即可。

## 十一、文件和交付要求

预计涉及：

- `export_fusion_evidence.py`
- `routing_data.py`
- `fusion_router.py`
- `train_fusion_routing.py`
- `test_fusion_routing.py`
- `analyze_ucf101_complementarity.py`
- `configs/fusion_routing.json`
- `configs/fusion_routing_roles.template.json` 及新的协议/角色示例
- `FUSION_ROUTING.md`
- 必要时新增一个顶层导出 Dataset 模块和一个轻量角色映射生成入口；已有等价组件则复用。

完成后汇报：

1. 实际新增/修改文件及行为变化。
2. 截图四项问题分别如何落实。
3. 真实角色映射是否生成，数量、类别覆盖、视频交集及历史摘要差异。
4. CPU 测试和合成 smoke 的实际命令与结果，未执行/跳过项及原因。
5. 观测 schema、时间间隔处理、静态初始化方式、参数量和 checkpoint 兼容策略。
6. 未执行的 GPU、全量导出、正式训练和基础模型重训操作。
7. 提供按实际 CLI 编写的后续命令，顺序为：
   - 生成/检查角色映射；
   - 单批真实权重回归，workers=0；
   - 少量完整 Train/Test 视频导出五头证据；
   - fixed 基线评估；
   - static 小规模拟合；
   - current/gru/conditional 的公平对照及预先固定 EMA 评估。
8. 命令中包含新协议、完整 Fusion checkpoint、正确缓存/角色及全新输出目录；路径参数说明清楚，不提供当前代码尚不支持却写成可直接执行的命令。

**本轮交付是“按已确认协议修好公共框架并完成小规模工程验收”。不要再次要求拆分 Train，不自动启动大规模实验，也不要改动现有专家训练主线。**

## 核对来源

- [现有数据协议摘要](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/research_exports/ucf101_filename_v2_fusion_A_2026-10-20/data_protocol_summary.json)
- [现有 Fusion 配置](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/research_exports/ucf101_filename_v2_fusion_A_2026-10-20/fusion_config.json)
- [checkpoint 索引](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/research_exports/ucf101_filename_v2_fusion_A_2026-10-20/checkpoint_index.json)
- [当前导出器](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/export_fusion_evidence.py)
- [当前数据与角色逻辑](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/routing_data.py)
- [当前 Router](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/fusion_router.py)
- [当前训练入口](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/train_fusion_routing.py)
- [当前分析脚本](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/analyze_ucf101_complementarity.py)
- [当前运行说明](https://github.com/CelestineYang/event-vision-research/blob/1ae665eed073328400ccdd255ddd71a72e39d27d/FUSION_ROUTING.md)

截图内容已在第二节逐项转写和回答，无需 Codex 再读取原图片。
