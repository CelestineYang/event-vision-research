# Git 管理约定

本仓库管理项目源码、测试、运行文档和上游许可证。数据集、模型权重、训练输出、虚拟环境、缓存、凭据和本地交接记录由 `.gitignore` 排除；文件保留在磁盘上，训练路径不变。

当前 `experiments/` 全部视为运行产物，其中生成的配置也不提交。需要复现实验时，将人工维护且不含凭据的配置放入独立的 `configs/` 目录。CSV 清单、`train_ucf.txt`、`test_ucf*.txt` 和报告 JSON 先留在本地，划分生成脚本仍被版本管理。保留依赖清单（例如 requirements.txt、pyproject.toml、environment.yml）和许可证。

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

## 关联 GitHub 私有仓库

在 GitHub 创建空仓库，选择 **Private**，不要初始化 README、.gitignore 或 License。创建后，在项目目录替换下面的 OWNER 和 REPOSITORY：

```bash
git remote add origin git@github.com:OWNER/REPOSITORY.git
git remote -v
```

关联本身不会上传文件。确认仓库可见性、GitHub 认证和提交内容后，未来需要同步时再执行：

```bash
git push -u origin main
```

参考：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
