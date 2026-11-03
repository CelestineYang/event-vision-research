# EventMamba 在 DailyDVS-200 上的论文方法重训

本次实现日期：2026-11-01。继续使用 **DailyDVS-200，200 类**，不把结果称为论文
12 类 Daily DVS 基准的复现准确率。旧的 1 秒、batch128、AdamW、150 轮结果保留供审计。

入口：`eventmamba_paper_data.py`（离线预处理）、`train_eventmamba_paper.py`（单卡训练）。
完整配置：[`configs/eventmamba_paper/dailydvs200.json`](configs/eventmamba_paper/dailydvs200.json)。
本次未修改正在运行的图像分支、其数据、共享 EventMamba 模型或旧实验训练源码。

## 来源与已确认选择

- 论文：[arXiv 2405.06116v4，IV-B](https://arxiv.org/html/2405.06116v4)。
- 官方仓库：[rhwxmx/EventMamba](https://github.com/rhwxmx/EventMamba)，固定提交
  `7f61fd397cf69f10f0f92a182109d36cd7b6fdbb`。本地 MIT 源码快照：`eventmamba_v2_upstream/`。
- 用户确认继续用 DailyDVS-200，保留原有参与者隔离 Train/Test 清单，不重新随机划分。
- 用户确认窗口按官方脚本：**1.5 秒窗口、0.5 秒步长**，实际重叠 1 秒。
  论文表 I/IV-B 把重叠写为 0.5 秒，两者不一致；本配置没有混淆步长与重叠。
- 用户确认**严格过滤**无合格窗口的视频，报告所有被排除片段及实际评测数量，
  不用整段/尾窗/最密窗口兜底，不默认使用 4093 作为最终评测分母。

## 训练设置

| 设置 | 本次取值 | 依据 |
|---|---|---|
| 初始化 | 随机初始化全部 EventMamba 参数 | 独立预训练；不加载任何 FusionA、GM 或历史权重 |
| 模型 | V2，embedding32，三阶段64/128/256，FPS1024/512/256，K24 | 官方 V2 类定义、README 的 DailyDVS 配置及 feature-KNN 指示 |
| 输入 | 2048 个时序排列的 t,x,y 点，无极性通道 | 官方 DailyDVS 预处理 |
| Batch | 单卡32个窗口 | 论文 IV-B |
| 训练轮数 | 350 | 论文 IV-B，覆盖官方脚本150轮默认值 |
| 优化器 | Adam，LR0.001 | 论文 IV-B，覆盖官方脚本AdamW默认值 |
| Weight decay | 1e-4 | 论文未说明，补用官方训练脚本数值；Adam中为耦合L2，并非AdamW |
| Betas / eps | (0.9,0.999) / 1e-8 | Adam默认值，论文未说明 |
| LR调度 | CosineAnnealingLR，T_max350，eta_min0，每个训练epoch后step | 论文cosine + 官方实现细节 |
| 损失 | 真类0.8，其余每类0.2/199的交叉熵 | 官方cal_loss；不同于PyTorch内建label_smoothing=0.2 |
| 其他 | seed0、FP32、BN正常训练、无额外梯度裁剪/早停/增强 | seed是项目选择，其他沿用参考训练路径 |
| 末尾训练batch | drop_last=True | 官方训练脚本 |
| 评测 | 每轮Test，保留全部合格窗口；视频采用窗口argmax多数投票 | 官方脚本的评测意图，修复其尾视频漏计 |
| 选模 | 视频多数投票Top-1严格提高时保存best；相同分数保留较早epoch | 官方脚本；另存final，不引入NLL平分决策 |

Test参与选模，没有独立Val。日志另外报告窗口Top-1/Top-5、普通CE、平滑CE及视频
平均概率Top-1/NLL，便于诊断；不会把平均概率Top-1混作主多数投票指标。
多数投票平票沿用Counter的首个出现类别规则。Top-5只计实际获得投票的类别。

## 数据处理

1. 从原始事件NPZ重新构造点云，读取整数微秒并先减去记录起点，避免绝对时间转float32
   丢精度。复用旧审计清单中的源NPZ SHA256验证输入，**不读取旧1秒窗口的采样点**。
2. 每条记录从首事件开始按0.5秒移动1.5秒窗口。窗口末端必须严格早于最后事件，
   与官方候选窗口条件一致。取事件采用精确半开区间 `[start,end)`，修复官方结束索引
   作为Python切片上界时额外漏掉一条事件的问题。
3. 原始版本仅保留原始事件数 **>10000** 的窗口。官方训练增强在这个判断之外，
   因此训练增强版本即使原始版本未过阈值也会生成；Test只使用合格原始版本。
4. 在**完整原始窗口**上构造original、随机空间/时间裁剪、时间反转、Y翻转、X翻转。
   裁剪直接调用固定官方 `uti.random_crop`，空间坐标保持320×240像素几何；不增加旋转。
5. 每个版本固定离线采样2048点，足够时无放回，不足时有放回；排序后保持时间顺序。
   原始版本按官方脚本先将采样时间量化到毫秒，再按采样时间min/max归一化；增强版本
   保留微秒后按采样时间min/max归一化。空间坐标x/320、y/240。
6. 固定seed0，各视频/窗口独立派生采样种子，避免worker调度改变缓存。每个epoch读取
   相同离线点样本，不重新旋转/裁剪/采样。
7. 每个缓存记录视频、类别、窗口索引、原始区间、增强版本、增强前采样总体的事件数。
   每条记录提交后可恢复；不兼容的旧缓存、源文件变化、不同协议会明确拒绝。
8. `coverage.json` 和 `COMPLETE.json` 列出原始视频数、保留视频数、点云样本数、
   被过滤视频的ID/标签/时长/事件数/原因。视频指标使用保留视频数作为分母。
   过滤后索引重新映射，保证最后一个视频和跨batch的视频均计入一次。

2026-11-01范围核对结果：

| 集合 | 原始视频 | 排除视频 | 保留视频 | 合格原始窗口 |
|---|---:|---:|---:|---:|
| Train | 14773 | 20 | 14753 | 全量离线处理时统计，包括增强样本数量 |
| Test | 4093 | 2 | **4091** | **22935** |

Train数量根据官方清单时长核对：每个足够长的记录首个完整窗口必有首事件，训练增强版本
不受原始版本10000事件阈值限制。Test已读取全部4093个NPZ的时间戳、核验源SHA256并按
窗口实际事件数扫描。被过滤的Test是`test_02342`（1.089890秒，51079事件）和
`test_03343`（0.099905秒，3783事件）。完整ID及依据在
`reports/eventmamba_paper_20261101/full_coverage_scan.json`。
正式缓存仍会独立重新生成覆盖率报告，供训练入口核对，不用扫描结果替代数据检查。

## 无法声称完全一致的细节

- 官方 `generate_daily.py` 调用了 `reverse_T/flip_H/flip_W`，但固定提交的`uti.py`
  没有这些实现。本实现明确补为：反转t并重排至时间递增、y→239−y、x→319−x。
  它们不是已验证与作者私有实现一致的代码。空裁剪跳过并计数；零时间跨度归一化有保护。
- 官方时间归一化增强分支先转float32；这里先在整数/float64中计算时间差，最后输出
  float32，以避免精度损失。原始毫秒归一化与作者公式保持一致。
- 官方窗口结束切片和最后视频统计有边界问题，本实现修复并测试这些问题，不复刻漏计。
- 原始论文没有固定所有训练细节；weight decay、label smoothing来自公开脚本，seed是
  项目选择。框架中的Adam默认值不是论文报告的值。
- 现有环境是Torch2.0.1+cu118及已验证的BiMamba-v2后端。README写Torch2.1.0/cu121，
  且未固定可用BiMamba fork版本。本次不修改共享环境，保存实际源码/扩展二进制hash；
  在**同一后端**上验证官方与本地模型前向、梯度一致性，不声称重建了作者原始环境。
- FPS评测也含随机起点；固定evaluation seed并恢复训练RNG，避免评测消耗训练随机序列。
  CUDA底层kernel仍可能存在微小非确定性，不保证跨硬件逐bit相同。

## 执行方式

以下使用公开版本的显式路径参数。先配置原始 NPZ 和已核验源文件 SHA256 的 prepared_v1 清单，再离线预处理；完成覆盖率验收后才启动训练。不要与已有任务重复运行。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python eventmamba_paper_data.py --root LOCAL_PATH/DailyDvs-200 --audit-root LOCAL_PATH/prepared_v1 --output LOCAL_PATH/eventmamba_paper_v1 --workers 2
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 python train_eventmamba_paper.py --data LOCAL_PATH/eventmamba_paper_v1 --output experiments/dailydvs200_eventmamba_paper_seed0_v1 --workers 2
```

公开数据入口也接受 `DAILYDVS200_ROOT` / `DAILYDVS200_AUDIT_ROOT` 环境变量作为默认值。既有任务维持启动时源码，不能将新的路径默认值热更新到身份校验中的运行目录。

续训在同一命令后加`--resume`，只加载这个输出目录的`last.pt`；源代码、协议、缓存或
依赖hash改变时拒绝混跑。保存optimizer、scheduler、Python/NumPy/Torch/CUDA RNG及
DataLoader shuffle RNG，从最后完整epoch恢复，不声称支持中途某个batch精确恢复。

`last.pt`、`best_video_top1.pt`、`final.pt`各自有`net`、协议、完整优化状态及指标。
最佳/最终模型的逐视频投票、首票位置、平均概率、真实标签保存在对应NPZ中。

## 验收

CPU测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-eventmamba/bin/python -m unittest test_eventmamba_paper -v
```

真实Train小样本：`experiments/eventmamba_paper_data_smoke_filtered_20261101`，
8个Train视频，245个增强训练样本、49个未增强评测窗口；smoke的评测仍来自Train。
正式入口不接受smoke缓存，旧1秒缓存也会被拒绝。

GPU测试在空闲GPU4执行，batch保持32、Adam参数保持正式值、cosine仍按350轮，
仅限制每轮两个batch、总共两轮。测试覆盖第一轮保存后重新启动续训，以及连续两轮运行。
报告目录：`reports/eventmamba_paper_20261101/`。这些测试仅验证实现可执行和续训行为，
不是正式训练结果，也不能保证DailyDVS-200上的泛化准确率。

已执行的验收结果：

- 10项CPU协议测试通过，包括官方采样/归一化/裁剪对照、边界条件、缓存断点恢复、
  排除视频后的索引映射、末尾视频投票及Adam续训。
- 官方V2与本地V2在同一CUDA后端的前向/梯度对照通过。
- 真实GPU两轮smoke、跨进程续训通过；峰值allocated4001.93MiB，reserved4294MiB。
- 从磁盘检查点恢复模型、Adam、scheduler、Python/NumPy/Torch/CUDA RNG后，所有恢复
  状态与保存状态完全相同。CPU训练续跑与连续执行数值完全相同。
- 两次**独立CUDA启动**的训练轨迹未通过紧数值公差对照：第一轮loss已有约1.08e-4差异，
  第二轮最大参数差约0.00504；最终RNG、scheduler相同。本检查不证明GPU逐步轨迹一致，
  未通过项保存在`resume_comparison.json`，不能笼统声称所有数值对照都通过。
- Train-only固定32窗、4类的小样本100步拟合，训练及eval模式均达到100%；平滑loss
  从5.494降至1.704。这是学习链路检查，不能外推全量测试集准确率。

本实现记录的是 DailyDVS-200 上的方法适配与训练设置，不能据工程测试宣称已成功复现论文准确率。该点分支预训练不计入本轮主实验/消融结果表。
