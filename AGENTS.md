# MoonVision 项目长期约定

## 更新完成后的同步与发布

用户在 2026-10-28 明确要求并持续授权：本项目每次完成代码更新后，都同步到 GitHub 的 `default` 分支，并更新 Mooncakes；无需为同一流程重复询问是否提交、推送或发布。用户在当前会话给出的暂停或例外要求优先。

- 仓库：`https://github.com/buildliming/moonvision.git`，同步目标为 `origin/default`。
- Mooncakes 模块：`shunge/moonvision`，官方服务为 `https://mooncakes.io`。不要将模块用户名改成 GitHub 用户名。
- 在测试和审查通过后提交本轮相关改动；先获取远程状态并处理分支差异，正常推送，禁止覆盖他人改动或强推 `default`。
- 每次发布递增适当的语义版本，同步 `moon.mod`、CHANGELOG、README 安装命令及固定版本链接；创建并推送对应 `vX.Y.Z` 标签。
- 发布前执行适用的验证与 `python tools/check_package.py`。完整源码验收入口是 `python tools/verify.py --references --real-images`，加上 `moon fmt`、`moon info`、`git diff --check`；已有针对当前源码的通过结果可复用，无须只因文档或版本号修改机械重跑所有数值测试。
- Mooncakes 包保持在 1 MiB 项目预算内；`docs/images/` 展示截图留在 GitHub，`web/samples/` 真实样本必须保留。曾有 2.55 MB 包上传被 HTTP 413 拒绝。不要重写已推送的版本标签，发布修正时递增版本。
- 用现有认证执行 `moon publish`，验证远程分支/标签及注册表版本，并在独立消费项目中验证能获取新版本。存在 CI 时跟进本次提交的检查结果。
- 不把“推送成功”“发布成功”“CI 通过”混为一谈；认证、网络或检查失败时继续完成独立工作并准确说明未完成部分，不能声称已经发布。

## 工作区与包内容

- 保留无关的用户文件和改动。`tools/real_bridge_runner.mjs`、`项目申报书.md` 是既有本地文件，不自动提交或发布。
- 编译产物、`artifacts/`、凭据与本文件不进入 Mooncakes 包；按 `.moonignore` 和 `tools/check_package.py` 审核。
- 本文件是仓库级持久偏好，提交到 GitHub 以便后续工作会话读取；它不创建定时任务，也不要求在没有代码更新时重复发布。
