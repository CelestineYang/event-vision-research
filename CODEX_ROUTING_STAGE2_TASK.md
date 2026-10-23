# 第二批任务：冻结专家路由的对照补齐与 Memory 机制诊断

## 0. 本批范围与最新依据

基于已推送提交 c505ab387b95adb851cc6eb758b66e0b5b870063 继续工作。若服务器工作区存在更新，先核对，不覆盖未提交修改。

第一批已完成的不只是开发：Fusion A epoch40 的五头证据已导出，Static/Current/GRU/Conditional 均已完成 80 轮、batch32 视频、Adam 1e-3 训练。结果以 research_exports/ucf101_routing_epoch040_2026-10-23/README.md、summary.json、sources.json 和各模式的 config/metrics/status 为准；旧迁移文档中的未运行、epoch6、20轮、batch8等说明不作为本次状态依据。

已选中模型：Static epoch10、Current epoch58、GRU epoch11、Conditional epoch21。视频 Top-1 分别为 96.24266%、96.32094%、96.32094%、96.32094%；NLL 分别为 0.19723610、0.19598950、0.19685699、0.19640851。同缓存 Fusion A Top-1 为 96.04697%。这些是单次原 Test 选模结果，不是独立测试或多种子结论。

本批不重新开发这四个模式，也不默认重跑它们。目标是补齐固定/EMA/图像旁路对照，解释动态门控在做什么，以及已有 Memory 是否真正利用了历史。暂不增加动态 eta、特征层 Gate、新损失、大型 memory bank 或专家联合微调。

本次交付代码、配置、文档和 CPU/合成测试；真实全量评估、新增对照拟合和多种子训练提供独立命令，不自动启动。

## 1. 固定并保护第一批产物

继续使用已确认协议 original_test_as_val：原 Train 全部有效视频用于参数训练和标准化拟合，原 Test 全部有效视频用于 gate_val，并按视频 NLL 选模；independent_test=false。不再拆 Train，不重新划分，不更改官方 Point 初始化来源 unknown 的记录。

使用现有 epoch40 的 train/test 五头证据缓存，不重新运行编码器/FPS，不更换专家 checkpoint，不用历史 DDP 的 96.125% 替代同缓存基线。保持 17 维观测、现有采样和标准化定义。

从 sources.json 解析第一批产物位置，在服务器核验真实路径、checkpoint SHA256、缓存 manifest 和角色/类别/观测身份。GitHub 仅含小型快照，不把报告目录当成 logits 缓存或权重目录。动态模型的实际 alpha/eta/b0 读取 state_dict，不能取 config 中保留的初始化占位值 alpha0=0.5、eta=0.5。

特别注意 load_checkpoint() 会核验 routing_data.py、fusion_router.py、train_fusion_routing.py 的源码哈希。优先不改这三个文件，新增外围工具，复用原实现；不改 checkpoint 中的哈希，不删除校验。若确需变更核心实现，建立明确的版本兼容路径，并保留能复现旧结果的原版本，不静默加载。

建议新增文件（若已有等价实现则复用）：
- evaluate_fusion_routing_suite.py：固定/已训练模式/EMA 的统一评估、历史干预和可选前缀评估。
- analyze_fusion_routing.py：逐视频配对、路径权重、门控/状态轨迹与结果汇总。
- test_fusion_routing_stage2.py：本批回归和合成测试。
- configs/fusion_routing_stage2.json：明确输入产物、评估矩阵、干预及输出位置。
- FUSION_ROUTING_STAGE2.md：任务范围、命令、结果解释边界。
辅助函数可另置小模块，不为凑文件名重复实现模型。

## 2. 补齐同缓存对照

新增 suite 入口，从同一缓存运行下列配置，结果写新目录：
1. 原五个头的独立指标，以及 Frame=(pXY+pXT+pTY)/3。
2. 现有 fixed：fusion、frame、frame_point、fusion_frame、three_way；与第1项重复的结果只计算/展示一次。
3. 已训练 Static/Current/GRU/Conditional 的 best_val_nll.pt，确认复现快照指标。
4. Current 最优 checkpoint + EMA，使用第一批配置中已固定的 decay=0.9，不扫描系数或重新选择 Current epoch。

以上均仅前向读取 logits 缓存，不训练专家或路由模型。固定混合使用概率算术平均，沿用原窗口混合后视频平均的顺序。先补齐同缓存 Fusion A 的 NLL/Top5/Brier/ECE，不能拿历史不同采样输出的 NLL填表。

额外准备一个“小模块拟合”对照的配置和独立命令：Static Fusion+Frame，即 fusion_frame_only=true、alpha=1，仅在原 Train 学习 eta。沿用第一批预算、视频 NLL 选模、同缓存与输出规范；不把它作为 unrestricted 动态模型的新初始化，不覆盖已完成 Static，也不自动执行。原因是现有增益可能主要来自增加独立 Frame 旁路，需要和完整 Frame–Point 路由区分。

## 3. 输出逐视频配对与真实路径权重

保留原指标口径，追加逐视频/逐窗口的可追溯输出，避免只看相同 Top-1 或纠错数量。

对已选模型两两比较：Static/Current、Current/GRU、Current/Conditional、GRU/Conditional。按视频身份精确对齐，核验类别、窗口集合和数量。统计预测不同的视频、两模型谁对谁错、相对同缓存 Fusion 的纠错/新增错误集合、逐视频 NLL差。7/0对7/0不能直接写成纠正了相同视频。

保存视频完整概率或 log-probability（数值数组+索引元数据），供差异复核；逐窗口轨迹通过显式选项导出，不把巨大数组写进普通训练日志。

逐窗口记录 alpha_t、eta 及：
    w_A = 1 - eta
    w_F = eta * alpha_t
    w_P = eta * (1 - alpha_t)
并验证三者非负、和为1。当前共同 eta≈0.0941077，因此约90.5892%的显式混合权重留给原 Fusion A；alpha 变化只重新分配其余部分。不能把 alpha=0.9描述成最终90%依赖独立Frame，更不能把这些路径权重当作整个网络中Point/Frame的因果贡献。

追加当前修正 delta_current、历史读出 delta_memory（适用时）、隐藏状态范数、Conditional 写门均值/分位数、alpha接近0/1的比例，并同时比较门控变化和最终概率变化。可使用仅观察的hook或外围适配器，但必须验证开启轨迹导出不改变普通前向输出，临时hook应在退出/异常时清理。

轨迹记录包含 video_id/window_id/start_us/end_us/dt；真实标签和真实类别概率只用于输出后的离线分析，不能返回模型输入或状态。

## 4. 在现有 checkpoint 上加入可解释的历史干预

第一轮只实现并验证下列三种，不做大规模干预网格：

- normal：完全原始的因果序列前向。
- reset_each_window：每个有效窗口前重置隐藏状态；保留原始当前观测、预计算dt和标准化器，不把所有gap改成0。GRU仍可处理当前观测，Conditional仍保留r(o,0)，所以该测试是去掉跨窗口历史，不等于删除全部新增层。
- zero_memory_readout：将添加到门控logit的记忆读出项置零，保留当前MLP，并保持合法的状态更新。它检验读取分支的作用，不与reset混同；特别是GRU读出包含当前更新后的信息，不能将其全部解释为纯历史贡献。

Current/Static在仅改隐藏历史的控制下应不变。所有模式保持视频间重置、padding不写状态、不计入损失/投票、无未来窗口输入。单窗口和不等长视频必须正常。

输出干预前后的Top1、NLL、Brier、ECE、预测变化、alpha变化和概率变化，明确标注为“固定checkpoint的推理期机制诊断”，不把它当作重新训练后的结构消融。性能不变不自动证明Memory绝无价值，性能下降也不自动证明专门读写设计优于GRU。

可同时准备25%/50%/75%/100%有效窗口前缀评估，但列为可选，不阻塞前述核心任务。前缀定义固定为按start_us排序后的有效窗口数比例，保留max(1,ceil(r*T))个窗口；这不是原始时间长度比例。所有方法看相同前缀，从零状态开始，只对保留窗口投票，并使用已选的同一个checkpoint，不按前缀重新选模。

本批不实现整段乱序后冒充在线因果推理的评估，也不通过其他视频的标签或样本身份构造状态。

## 5. 实验资源与后续接口

本批先只做上述评估/诊断工具和缺失静态对照配置，避免重复开发完整训练框架。资源开销来自已有缓存上的小网络计算，不需要重训ResNet/EventMamba/Fusion A或重导出五头。

提供后续多种子实验的显式配置/命令，建议seed0复用已完成结果，seed1/2另开目录；第一轮可固定同一个已完成Static和专家缓存，只变动态模块初始化/训练顺序，但必须明确这是“固定专家和Static条件下的Gate/Memory多种子”，不是整个系统重训的方差。不得自动执行或将未执行项写进结果表。

不根据当前eta小就直接调大eta，不因为alpha饱和就强制均衡，不在诊断前增加新的融合头/损失。若后续证据支持，再单独设计dynamic eta等候选，并为Current和Memory提供对称的机制对照。

新评估器按缓存class_to_idx和dataset_id工作，不写死101类和UCF路径。后续数据集读取各自明确协议，不全局强制original_test_as_val。仓库已有THU独立便携流水线，复用其已有工作，不在本批重写或声称其端到端已经验收。

## 6. 最低测试与交付

CPU/合成测试至少覆盖：
- suite普通模式与旧evaluate口径一致，fixed/EMA计算正确。
- state_dict中实际权重与诊断权重一致；轨迹开关不改变预测。
- 逐步前向normal与原整段前向数值一致，视频间独立；未来扰动不改变过去输出。
- reset只重置状态，不改变当前观测/dt/标准化；zero_read只屏蔽指定读出；Current/Static负对照不变。
- padding、单窗口、不等间隔、少于5类，以及概率归一化正确。
- 相同准确率但不同错误集合的fixture能被配对分析区分；错误为空时不崩溃。
- 缓存、模型、类别或观测身份不匹配明确报错，不能删除校验；旧目录不覆盖。
- 可选前缀100%复现全序列，其他前缀不包含未来窗口。

交付：修改清单、测试日志、新配置、运行说明和独立命令。说明哪些只实现/合成验证、哪些读取过真实缓存、哪些尚未执行。只提交代码/小型配置/报告，不提交原事件、全量logits或模型权重。

执行优先级：保护并复核第一批产物 → suite固定/EMA与已训练结果复现 → 配对/路径轨迹 → reset/zero_read → 可选前缀和多种子命令。不要用旧交接文档误判第一批仍未训练。

最终希望本批能够回答：实际收益是否超过简单Frame旁路或固定集成？三个动态模型在相同Top1下究竟改变了哪些视频和概率？Memory是否读到了有用历史，还是主要重复当前MLP的选择？这些问题没有答案前，不把新增复杂模块当作默认下一步。
