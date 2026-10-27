# Git 管理约定

本仓库管理项目源码、测试、运行文档和上游许可证。数据集、模型权重、训练输出、虚拟环境、缓存、凭据和本地交接记录由 `.gitignore` 排除；文件保留在磁盘上，训练路径不变。

当前 `experiments/` 全部视为运行产物，其中生成的配置也不提交。需要复现实验时，将人工维护且不含凭据的配置放入独立的 `configs/` 目录。CSV 清单和报告 JSON 留在本地；根目录 `train_ucf.txt` 与 `test_ucf(1).txt` 作为训练/测试划分清单纳入版本管理，`experiments/` 中的生成副本仍被排除。划分生成脚本也被版本管理。保留依赖清单（例如 requirements.txt、pyproject.toml、environment.yml）和许可证。

用户在 2026-10-20 明确要求上传用于讨论下一步设计的精选结果材料。因此，
`research_exports/ucf101_filename_v2_fusion_A_2026-10-20/` 是一项有限例外：
允许版本管理经过审阅的配置、完整指标快照、曲线、已选模型的预测 CSV、
官方训练日志副本和来源校验清单。原始 `experiments/`、`logs/`、`reports/`
及根目录 `eventmamba.txt` 仍排除；权重、缓存、数据集、凭据不进入快照。
该目录是某一已完成轮次的静态报告，不随后台训练实时更新。更新快照时必须
重新核对轮次、来源 SHA256 与模型选择规则；不要使用 `git add -f`。

用户在 2026-10-23 授权上传本轮路由结果。新增有限例外
`research_exports/ucf101_routing_epoch040_2026-10-23/`：允许经审阅的配置、
四种模式完整 metrics.jsonl、NLL 选中模型的预测 JSON、汇总及来源 SHA256。
该目录不含证据缓存或模型权重，不改变原始实验产物的排除规则。

日常提交前检查：

第二阶段收尾经用户授权上传精选结果至
`research_exports/ucf101_routing_stage2_2026-10-23/`：保留评估指标、三种子完整训练指标、
实际配置、选中预测、小型配对汇总与来源 SHA；不包含权重或完整概率数组。
`reports/routing_stage2/` 仅允许最终完整性审计与 CPU 测试日志，其他 reports 仍排除。

2026-10-24经用户授权，第三阶段精选快照位于
`research_exports/ucf101_routing_stage3_2026-10-24/`：允许完整训练指标、配置、
选中模型的精简视频预测、曲线、三种子配对统计及来源SHA；不含权重、缓存或完整概率数组。
`reports/routing_stage3/`仅允许明确列出的CPU测试、输入审计和一次性Train基线小型报告。

```bash
git status --short
git diff
git add <明确的代码或文档路径>
git diff --cached --stat
git diff --cached
git commit -m "说明本次改动"
```

不要用 `git add -f` 绕过排除规则。`.gitignore` 按路径和文件类型生效，无法识别源码中的硬编码密钥；新增配置或代码仍需检查。`.env.example` 等模板只应包含占位值。不要在此工作目录运行 `git clean`、`git reset --hard` 或切换到会覆盖正在使用的代码的版本。

## 已关联的 GitHub 仓库

- 分支：`main`；远程名称：`origin`。
- HTTPS地址：`https://github.com/CelestineYang/event-vision-research.git`。
- 目标为私有仓库，推送前通过认证后的API重新核验可见性和写权限。
- 仓库已经初始化、作者已经配置，不要重新初始化Git或更换为SSH。

使用官方GitHub CLI的浏览器设备登录，并将GH_CONFIG_DIR放在独立临时目录。
用户在 `https://github.com/login/device` 输入一次性设备码；不在聊天、URL、
命令参数、项目文件或日志中传递访问令牌。推送使用临时进程配置的credential
helper，不把认证写入项目或全局Git配置。结束后清除本次临时认证。

暂存内容经过审阅并提交后，再同步远端：

```bash
git fetch origin main
git merge-base --is-ancestor origin/main main
# 只有上面的祖先检查成功，才普通推送；否则先分析分歧。
git push origin main
```

推送完成后核对远端main的SHA与本地HEAD一致。认证失败时分别报告本地
提交和远端上传状态，不把“本地已提交”说成“已上传”。禁止强制推送。

## 2026-10-27 发布范围更新

用户授权同步消融源码、可复现配置及有用的正式研究资料。API核验该仓库当前为 public；旧文档的 private 描述是历史状态。公开发布使用 `configs/examples/` 中的路径占位模板、`PROJECT_CONTINUITY.md` 和 `research_exports/thu_chl_ablation_2026-10-27/` 中已完成的 GM 指标。原始数据、权重、服务器路径、实时运行状态和故障日志仍保留本地。用户明确表示无需同步 OOM 等运行插曲。

有已连接且具有写权限的 GitHub App 时，可使用其 Git data API 创建 tree/commit，再以预期旧 SHA 和非强制更新 main；无需重复设备登录。推送前后均核对仓库与分支 SHA。API 发布必须同步本地 Git objects、索引和 main，不能只更新远端却声称本地已提交。
