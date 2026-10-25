# DailyDVS-200 全事件窗口预处理

协议：官方 Train/Test，不额外划分；1 秒窗口、0.5 秒步长；每条原始样本独立分窗。
短于1秒样本使用一个名义1秒的半开窗口，不伪造事件。长样本末尾不额外补残窗。
保留全部候选窗口，不使用 >10000 过滤规则。

目标覆盖：Train 14,773 样本 / 96,068 窗口，Test 4,093 样本 / 27,028 窗口。
共123,096窗口、369,288张三投影图。唯一短Test样本test_03343（约0.10秒、3783事件）保留。
完整事件流全部写入mmap，包括窗口尾部未覆盖事件；窗口索引不改变此前约定。

输出根：`/remote-home/datasets/event_project_cache/dailydvs200_full_windows_1s_stride05_v1`

- `experiment/window_manifest.csv`：全部窗口身份、时间边界、事件数、源哈希。
- `experiment/video_splits.csv` / `class_to_idx.json`：官方样本划分和200类映射。
- `experiment/cache/{cstr3,xt_p,ty_p}/`：Train/Test全部未增强、CHW float32三投影，224×224。
- `experiment/dataset_audit.json`：NPZ字段/坐标/极性/时序以及存储、投影检查结果。
- `event_store/videos/<sample_id>/events.npy`：完整事件流，保持原始t/x/y/p类型和值。
- 每条样本的 `metadata.json`：原NPZ、mmap及每张图的SHA256和窗口切片索引；作为可恢复提交。
- `event_store/videos.json` / `summary.json`：兼容现有EventStore读取。
- 根目录 `status.json`：实时进度；`COMPLETE.json`：全量完成标记。

预计事件数组约189 GiB，三投影约207 GiB，总计约400 GiB（含文件头、清单、元数据和文件系统开销）。
不复制原始AEDAT4，不修改已有NPZ，不在实例工作目录存大缓存。不重新解码AEDAT4；本轮验证对象是已有NPZ。

Train缓存是**未增强参考图像**，不能直接拿它与增强Point配对。空间增强训练仍从mmap取全事件，施加同窗/同视频共享几何变换，再构造两路输入。Test可直接读取缓存。
不永久固化2048点采样；Point仍在线从对应窗口采样，以支持共享增强和epoch随机流。

入口：`preprocess_dailydvs200_full.py`，复用既有 `window_views` 数学实现，但不复用UCF的样本过滤规则。
每样本：验证NPZ→完整mmap逐字段精确比较→渲染全部三投影→重载比较并保存哈希→原子提交metadata。
中断后 `--resume` 严格检查协议/源码/清单身份并校验已提交样本，仅重做未提交样本；异常不会被静默跳过。
已完成目录拒绝覆盖。运行锁阻止同目录并发。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-eventmamba/bin/python -u preprocess_dailydvs200_full.py --workers 2 --output /remote-home/datasets/event_project_cache/dailydvs200_full_windows_1s_stride05_v1
# 中断恢复：同一命令末尾加 --resume
tail -f logs/dailydvs200_windows/full_1s_stride05_v1.log
```

全量开始前执行普通Train样本和短Test样本的真实预处理smoke，以及恢复校验；这是数据检查，不是Test调参或模型训练。
此交付仅准备数据。DailyDVS的200类训练入口/模型初始化仍需后续单独适配，不能直接加载UCF101的101类分类头训练。
