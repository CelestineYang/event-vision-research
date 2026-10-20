# Event vision research

当前工作是 **UCF101-DVS：预训练 gCSTR + 官方 filename EventMamba V2 的
prototype A 融合**。本仓库同时保留先前 DailyDVS-200、SECNet、EventMamba V1
及整段事件实验；它们的配置不能直接替代当前窗口级实验配置。

## 当前实际代码入口

| 功能 | 文件 |
|---|---|
| 现有 TXT 视频划分 | `train_ucf.txt`、`test_ucf(1).txt`、`prepare_ucf101_txt_split.py` |
| 窗口、原始极性和投影 | `ucf101_window_data.py`、`gcstr_representations.py` |
| 已预处理事件读取 | `ucf101_event_store.py` |
| 三路图像独立预训练 | `train_ucf101_window_joint.py` |
| 融合窗口数据 | `ucf101_fusion_window_data.py` |
| V2 三层特征及官方分类头 | `eventmamba_v2_pretrained.py`，复用 `eventmamba_v2_model.py` |
| 预训练权重加载和融合 | `ucf101_pretrained_fusion.py`，复用 `gcstr_eventmamba_fusion.py`、`gcstr_secnet_fusion.py` |
| 正式融合训练 | `train_ucf101_pretrained_fusion.py` |
| 后台任务与续训管理 | `run_ucf101_pretrained_fusion.py` |
| 曲线和比较报告 | `analyze_ucf101_fusion_curves.py` |
| 精选分享快照导出 | `export_ucf101_research_snapshot.py` |

当前代码按真实训练源文件保存。Git 审阅/提交不修改正在执行的训练源码，也不
热更新进程。源文件 SHA256、数据清单 SHA256 和输入权重 SHA256 保存在
实际实验 `config.json` 中；其经过审阅的副本包含在分享快照内。

## 输入和训练协议

- 先按原始视频 TXT 名单分 Train/Test（约8:2），再生成1s窗口、0.5s步长。
  使用原始事件数>10000的有效窗口，候选窗口数<40的视频保留。
- 三路图像使用同一窗口的全部原始事件及共享空间增强。点分支采样2048点，
  输入归一化的 t,x,y；采用官方 x/320、y/240、采样后时间 min/max。
- 三个 gCSTR ResNet-18 以已选 `best_loss.pt` 初始化并冻结（含BN）。
  filename V2 点模型以官方权重初始化且可训练：32通道嵌入、阶段64/128/256、
  FPS1024/512/256、K24、Feature-space KNN、BiMamba-v2。
- 每层点特征经16个prototype查询，再分别查询三路49个图像token；每层
  输出128维，三层拼接384维后101类分类。保留点分支原始注意力池化分类头。
- 双卡 DDP+SyncBN，150轮，全局batch16，AdamW固定LR1e-4、weight decay1e-4，
  loss = fusion CE + 0.5 × point CE，梯度裁剪1，无早停、scheduler或AMP。
- 每轮在Test上做窗口级评价及视频Soft Voting，以融合视频NLL选模。
  Test参与选模，**不是未参与模型选择的独立测试集**。TXT名单没有同视频重叠，
  但存在同class/group的不同片段跨集合；不声称为group隔离协议。

## 运行说明与分享材料

- [当前 V2 融合运行说明](UCF101_FILENAME_V2_FUSION_RUNBOOK.md)
- [V2 适配与历史 V1 边界](UCF101_PRETRAINED_FUSION.md)
- [TXT 划分说明](UCF101_TXTSPLIT_V2_RUNBOOK.md)
- [事件预处理说明](UCF101_EVENT_PREPROCESS_RUNBOOK.md)
- [2026-10-20 精选结果快照](research_exports/ucf101_filename_v2_fusion_A_2026-10-20/README.md)
- [Git 管理与数据排除规则](GIT_WORKFLOW.md)

快照是截至其清单记录轮次的阶段性结果；其 `fusion_metrics.jsonl` 为训练日志，
`eventmamba_official_training.txt` 为作者的预训练日志，二者来源不同。
模型权重、真实事件数据、缓存和虚拟环境未上传，代码不会在导入时自动训练。

## 验证

只做CPU检查（不需模型权重和GPU）的命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_ucf101_window_gcstr test_ucf101_event_store test_research_snapshot -q
```

`test_pretrained_window_fusion.py` 是显式执行的真实权重/CUDA集成测试，
本次 Git 更新不会运行它。历史适配器 `eventmamba_pretrained.py` 仅针对旧 V1，
不被当前 V2 训练入口引用，不能加载新 filename checkpoint。
