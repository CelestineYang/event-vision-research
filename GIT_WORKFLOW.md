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

日常提交前检查：

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
