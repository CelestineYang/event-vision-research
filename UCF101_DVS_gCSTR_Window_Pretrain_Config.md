# UCF101-DVS：滑动窗口 gCSTR 独立预训练配置与实施说明

> **划分更新（2026-10-15）**：用户确认改用 `train_ucf.txt` / `test_ucf(1).txt`，新目录 `experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/`。该清单为约80/20视频划分，允许同类别同group跨集合；不称作 RGB 官方 split 1。原训练保持暂停，新划分从 ImageNet 权重重新开始。最新划分说明见 `UCF101_TXTSPLIT_V2_RUNBOOK.md`。

> **后续用户确认的覆盖设置（2026-10-14）**：训练集使用事件级 `spatial` 增强（每窗口每轮一个随机版本），Test 不增强；双 GPU DDP，全局 Batch Size=64；无早停；仅官方 Train/Test，Test 参与选模。下方原始提案中的 `none`、Validation 和 Early Stopping 不再作为当前执行设置。最新说明见 `UCF101_WINDOW_GCSTR_RUNBOOK.md`。

> **三分支执行方式更新**：三个独立 ResNet 同时驻留双卡，每批共用一次事件读取及空间增强，分别生成投影并独立更新各自 Adam。每分支全局 Batch Size 仍为 64（通常每卡 32），三个分支同步推进 epoch。入口为 `train_ucf101_window_joint.py`；这取代下文按分支依次完成训练的执行顺序，不引入融合或联合微调。

> 交接对象：负责整体项目训练与监控的 Codex
> 任务范围：**只实现并运行 gCSTR 三个图像分支的窗口级独立训练**。暂不训练 EventMamba / SECNet，不增加 Cross-Attention、Memory 或 Gate，不进行联合微调。
> 方案名称：`ucf101dvs_gcstr_sliding_pretrain_v1`。

## 0. 核心结论与不可混淆的约束

- 数据集：**UCF101-DVS，101 类**；不得误用现有 DailyDVS-200 的 `gcstr_splits_seed0.csv`（该清单为 200 类）。
- 时间滑窗：**1.0 秒窗口、0.5 秒步长、50% 重叠**；窗口数由原始视频时长决定，而非固定 10 个。
- 后续 EventMamba/Point 分支：**每个有效窗口随机采样 8192 个事件点**（由用户指定，替代官方示例的 2048）。
- **本阶段 gCSTR-only 不采样 8192 点**：每个有效窗口内的**全部原始事件**分别转换为三幅 gCSTR 图像，以保留密集投影的信息；之后融合时再给同一窗口的 Point 分支采样 8192 个点。
- 三个独立模型：`cstr3`（XY）、`xt_p`（XT）、`ty_p`（YT），各使用 **ImageNet 预训练 ResNet-18**，分别训练、分别保存 checkpoint。三分支的这些 variant 是本项目现有实现沿用的选择；**不声称是 gCSTR 作者针对 UCF101-DVS 验证选出的最优 variant**。
- **必须先审计原始数据有无 polarity（`p`）**。EventMamba 官方 UCF101-DVS 滑窗代码只使用 `ts,x,y`，没有证明上传/服务器上的数据一定有 `p`。如果确无极性，本阶段的 `cstr3/xt_p/ty_p` 极性表征**不能按原定义训练**；暂停并报告数据格式阻断，不要伪造全 0/全 1 极性，不要默默改表示。

## 1. 先执行数据与代码审计（阻断条件）

1. 确认服务器上 UCF101-DVS 原始文件路径、实际格式（`.mat` 或其他）、视频 ID、类别名称、事件字段、时间戳单位、坐标范围/传感器宽高、是否保留 polarity。
2. 对至少每类若干样本检查：时间戳单调性、`x/y` 越界、空视频、时间戳单位（官方脚本以微秒处理），以及 `p` 的取值和有效性。若缺少 `p`，**在训练前停止**，汇报可获取的带极性原始数据源或单独立项的“非极性 gCSTR 变体”；未经批准不改变研究对象。
3. 核实服务端是否有 `gcstr_representations.py`、`gcstr_augmentation.py`、`create_gcstr_splits.py`。当前提供的 `gcstr_dataset.py` 引用了它们，但它们不在本次上传文件清单中；不能直接假定缺失代码可运行。
4. 确认类别到 `label_id` 的映射为确定性的 `[0,100]`，并将映射保存到 JSON；按视频或官方 group/split 进行 train/val/test 划分，固定 `seed=0`。**必须先分原始视频，再生成窗口**，禁止同一原始视频的重叠窗口跨集合。若有与官方 benchmark 对应的既定划分，优先沿用；否则记录新划分并明确声明不可直接当作官方可比成绩。
5. 先产出 `dataset_audit.json` 与 `video_splits.csv`，记录每个 split 的原始视频数量、各类计数、窗口事件数分布及被过滤样本数量。任何原始视频丢失全部有效窗口均需列出 ID 与原因。

## 2. 滑动窗口与预处理：统一协议

| 字段 | 配置 |
|---|---|
| `dataset` | `UCF101-DVS` |
| `num_classes` | `101` |
| `window_size_s` | **`1.0`** |
| `step_size_s` | **`0.5`** |
| `overlap_s` | `0.5` |
| `window_alignment` | 从原始视频首个事件时间开始，固定时长平移 |
| `keep_full_windows_only` | `true`；不足 1.0 秒的尾部不单独创建短窗口 |
| `min_raw_events_per_window` | **`10001`**（即原始窗口事件数 `>10000`） |
| `official_max_window_count_exclusive` | **`40`**（官方示例对原视频 `NUM_WINDOWS < 40`）；先审计过滤影响，并明确记录该条件是否启用；本初版按官方示例启用 |
| `event_sampling_for_gcstr` | **`all_events_in_window`** |
| `point_count_for_future_fusion` | **`8192`**（本阶段不调用 Point Encoder） |
| `window_label` | 继承视频动作标签 |
| `sensor_size` | 读取并验证真实数据元信息；不得擅自硬编码 |
| `gCSTR_image_size` | `224 × 224`（沿用现有 ResNet 输入预处理） |
| `time_normalization` | 使用窗口开始时间与固定 1 秒跨度作为局部时间基准；Point 与 gCSTR 后续一致 |

- 第 `i` 个窗口为 `[t0 + i × 0.5s, t0 + i × 0.5s + 1.0s)`，边界采用左闭右开以防重复计算同一事件在单个窗口内的位置；**跨相邻重叠窗口的事件重复出现属于预期**。
- 原始事件数量过滤在 **8192 采样之前**完成。通过 `>10000` 的窗口天然可以在后续无放回采样 8192 点，无需重复填充。
- 对每个有效窗口，在该窗口的全量事件上独立生成 `cstr3/xt_p/ty_p`，不要先抽取 Point 的 8192 点再生成 gCSTR。
- `XT/YT` 的时间 bin、极性通道和缩放/插值等操作以已审计通过的项目 `gcstr_representations.py` 为准；检查其是否依赖完整文件的 `t[0]/t[-1]`，必要时增加窗口局部时间基准并配套单元测试。既有 `GCSTRConfig` 的具体变体设置应记录进配置日志，不得无记录改动。
- 图像缓存单独建目录，缓存键至少包含：数据版本、video_id、window_id、起止时间、表征 variant、极性约定、resize/time-bin 配置及生成代码版本；**不复用 DailyDVS-200 的整段 gCSTR 缓存**。
- 最终 `window_manifest.csv` 至少包含 `video_id,window_id,start_us,end_us,event_count,label_id,split,source_path`，以及可复原三投影缓存文件的路径或键。

## 3. gCSTR-only 的训练配置（Stage 1）

**严格训练 3 个独立模型**，避免提前融合：

| 项目 | 第一版配置 | 说明 |
|---|---|---|
| `views` | `cstr3`, `xt_p`, `ty_p` | 三次独立训练 |
| `backbone` | `ResNet-18` | 沿用 `GCSTRBranchClassifier` |
| `pretrained` | `ImageNet1K_V1` | 现有代码 `pretrained=True` |
| `num_classes` | `101` | 必须改掉目前代码默认的 `200` |
| `input_shape` | `[B,3,224,224]` | 三通道图像、ImageNet normalization |
| `train_unit` | `window` | 每个有效窗口作为一条训练样本 |
| `train_coverage` | **每个 epoch 遍历全部有效训练窗口** | Shuffle 窗口，视频 split 不变 |
| `loss` | `CrossEntropyLoss` | 窗口级 101 类监督 |
| `optimizer` | `Adam` | 对齐当前 `train_gcstr.py` |
| `lr` | `1e-4` | 对齐当前默认值，属于本项目初始建议 |
| `weight_decay` | `0` | 沿用当前 Adam 代码；如更改需记录 |
| `epochs` | `50`（上限） | 验证集 Early Stopping |
| `batch_size` | `64` | OOM 时降至 32/16，并在日志中记录；每次只训练一个 ResNet |
| `num_workers` | `4`（可调） | 受服务器 IO 约束 |
| `seed` | `0` | 固定 numpy、torch、loader 随机源 |
| `augmentation` | `none`（第一版） | 避免引入另一实验变量；后续另作消融 |
| `grad_amp` | 可启用 CUDA AMP | 若启用记录精度和版本，不作为精度变化解释 |
| `checkpoint` | 每个 view 保存 `best_loss.pt`、`best_top1.pt`、`last.pt` | 需保留兼容 `model_state`、`view`、`epoch`、`config` 的字段 |

**训练步骤：**

1. 每个 view 使用相同的 `window_manifest.csv`、相同 train/val/test 划分与一致的滑窗过滤规则。
2. 每次只训练该 view 的 ResNet-18。现有 `gcstr_models.py` 中 `GCSTRBranchClassifier(view, num_classes=200, pretrained=True)` 必须显式改为 `num_classes=101`。
3. 在每个 epoch：训练阶段随机打乱**窗口训练样本**、计算窗口级 CE；验证阶段遍历**全部验证窗口**，每个窗口输出 101 维 logits 并关联其 `video_id`。
4. 验证时先计算单窗口 Top-1 / Top-5，再对同一 `video_id` 的每个窗口 `softmax(logits)` 求平均，得到视频级概率：`p_video = mean_i softmax(logits_i)`；统计 **Video Top-1 / Top-5 和 Video NLL**。分母是视频数，不是窗口数。记录 hard-majority voting 作为辅助对照（平票规则固定）。
5. **Early Stopping 采用视频级 Validation NLL，patience=10**。`best_loss.pt` 按最小视频级 NLL 保存，`best_top1.pt` 按最高视频级 Top-1 保存；同分时以更低 NLL、然后更早 epoch 决定。测试集不得参与训练、选模或 variant 调整。
6. 每个 epoch 输出窗口级与视频级指标、视频数/窗口数、有效窗口覆盖率、训练时间、GPU 显存峰值、optimizer/lr、对应的 checkpoint 路径。

> 说明：此版本保持“所有窗口参与窗口级训练”的思路。长视频产生更多窗口，训练时贡献也更大；先记录每个视频的窗口数分布，是否增加 video-balanced sampler 留作单独实验，不在本轮暗中切换。

## 4. 现有代码的改动边界

- `train_gcstr.py`：当前只支持 DailyDVS-200 数据集，默认 200 类，只算窗口/样本级指标，默认选用 Validation Loss 保存最好权重。**需要新增 UCF101-DVS 专用入口或复制为新脚本**，不要直接覆盖现有 DailyDVS-200 结果。
- `gcstr_dataset.py`：当前直接加载 `.npz` 的 `t,x,y,p`，通过 `build_gcstr_views` 生成图像。应新增**以视频级划分与 window_manifest 为索引的 UCF101 窗口数据集**；不能仅把旧 `gcstr_splits_seed0.csv` 的路径替换掉。
- `gcstr_models.py`：现有 ResNet-18 接口可复用，调用时显式 `num_classes=101`；不要改已有 DailyDVS-200 的默认行为，优先在新实验中传参。
- `gcstr_representations.py`：先审计实际存在及 UCF101-DVS 事件格式可用性，再决定是否需要窗口局部时间轴与传感器参数适配。严禁凭空补缺的极性。
- `gcstr_secnet_fusion.py`：本阶段**不训练、不修改**；但 checkpoint 需与其 `load_image_checkpoints` 的 `view`/`model_state` 加载约定兼容。**后续融合模型的 `num_classes` 也必须设为 101**，这是下一阶段任务。
- **第一版不要加入** Memory、Reliability Gate、Cross-Attention 改进、Point 训练、端到端联合微调。

## 5. 独立的输出目录与交付物

建议：

```text
experiments/ucf101dvs_gcstr_sliding_v1/
  config.yaml
  dataset_audit.json
  class_to_idx.json
  video_splits.csv
  window_manifest.csv
  checkpoints/
    cstr3/{best_loss.pt,best_top1.pt,last.pt}
    xt_p/{best_loss.pt,best_top1.pt,last.pt}
    ty_p/{best_loss.pt,best_top1.pt,last.pt}
  logs/{cstr3,xt_p,ty_p}_metrics.jsonl
  reports/{cstr3,xt_p,ty_p}_val_video_predictions.csv
  reports/summary.json
```

在 `summary.json` 中保留数据版本、split 规则、样本过滤数量、representation config、各分支最佳 epoch、window/video 指标、随机种子、代码 commit 和 checkpoint 哈希；不要覆盖任何原来的 DailyDVS-200 Checkpoint。

## 6. 启动前必须通过的验收测试

- `p` 字段真实存在且有效，否则以 **BLOCKED: UCF101-DVS polarity unavailable** 结束，不自动继续。
- 同一 `video_id` 的所有窗口都属于同一 split；`train ∩ val ∩ test = ∅`（按原始视频 ID）。
- 时间窗口长度为 1 s，步长 0.5 s，合法区间不越界；事件数过滤条件确实是 `>10000`，没有误写为 `>=10000`；必要时核对与官方脚本的边界约定。
- 所有三投影以同一窗口**全量事件**生成，尺寸 `[3,224,224]`，无 NaN/Inf，时间坐标与极性含义正确。
- 101 类标签连续且每种表征训练/验证窗口 ID 完全一致。
- 用极小子集进行一轮 forward/backward/optimizer-step；确认梯度、loss、checkpoint 保存与恢复均正常。
- 验证集的 video-level 汇总按 `video_id`，不会因为 loader shuffle、窗口顺序或重叠造成重复视频计数。
- 测试集严格封存，不以测试结果调整任何参数。

## 7. Codex 执行优先级

**第一优先级：只做数据字段审计、划分和窗口 manifest，并报告是否有 polarity。** 若该阻断项未解决，停止本阶段的正式训练。

审计通过后按顺序：实现窗口 Dataset 与缓存 → 完成必要的单元测试和 smoke test → 独立训练 `cstr3` → 独立训练 `xt_p` → 独立训练 `ty_p` → 汇总三路验证指标与 checkpoint 清单。

如果程序接口和这里的配置存在冲突，**保留现有实验文件，建立 UCF101 专用配置/脚本，记录冲突及具体决策**，不要擅自替换原有项目路径或清理 Checkpoint。

## 8. 来源与设置边界

- EventMamba 官方滑窗脚本：<https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/generate_by_sliding_window.py>。其中 `WINDOW_SIZE=1`, `STEP_SIZE=0.5`, `NUM_POINTS=2048`, `NUM_WINDOWS<40` 与窗口 `>10000` events 来自官方代码；**8192 是本项目用户明确指定的修改**。
- 本项目既有代码：`train_gcstr.py`、`gcstr_dataset.py`、`gcstr_models.py`、`gcstr_secnet_fusion.py`，其参数与接口据本次提供的文件核对。
- `video-level NLL` 选模、`none` augmentation、固定视频 split 和 checkpoint 组织是**本次推荐的实现约定**，不应称为 EventMamba 官方训练细节。
