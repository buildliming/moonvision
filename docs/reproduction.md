# 从干净检出复现

## 环境

本次验证使用 Windows、MoonBit `0.10.14+7d59c7ec9`（`moon 0.1.20260920`）、Node.js `24.15.0`、Python `3.13.9`。native 后端还需要本机 C 编译器；本次为 MinGW GCC 8.1.0。

CI 使用同版本的官方 Linux 工具链归档，安装入口为 MoonBit 官方脚本：

```sh
curl -fsSL https://cli.moonbitlang.com/install/unix.sh -o install-moonbit.sh
bash install-moonbit.sh '0.10.14+7d59c7ec9'
export PATH="$HOME/.moon/bin:$PATH"
```

Windows 可按 [官方安装页](https://www.moonbitlang.com/download/) 配置工具链。先执行 `moon version --all` 记录实际版本。使用不同编译器版本时，应重新运行完整验证。

## 核心构建与测试

```text
git clone --branch default https://github.com/buildliming/moonvision.git
cd moonvision
moon check --target all --deny-warn
moon test --target wasm-gc --deny-warn
moon test --target wasm --deny-warn
moon test --target js --deny-warn
moon test --target native --deny-warn
moon run examples/main
moon run examples/pipeline
moon run examples/annotated
```

这部分不需要 Python 数值包。原生 C 编译器必须能正常启动，JS 测试需要 Node.js。

## 编译后的外部接口

```text
python tools/build_web.py
node tools/test_bridge.mjs
node tools/test_cli.mjs
node tools/moonvision.mjs augment examples/requests/augment.json
node tools/moonvision.mjs warp examples/requests/warp.json
```

`web/moonvision.js` 来自本地 MoonBit 构建，不手写也不提交；从干净检出必须先生成。

可复用的 Node 调用层位于 `tools/moonvision_client.mjs`；`invoke(mode, request)` 返回解析后的响应对象，`processJson(mode, text)` 返回 JSON 字符串，均为异步函数。调用方必须检查 `ok`。纯模块导入不会读取标准输入或启动 CLI。

## 真实照片预处理案例

```text
python -m pip install Pillow==12.3.0
python tools/build_web.py
python examples/real_pipeline.py
```

案例只依赖 Pillow、Node 和编译后的 MoonBit，不依赖 PyTorch/NumPy。默认三张仓库照片，输出到 `artifacts/real-pipeline/`；可用 `--samples` 选择清单中的 RGB 照片、`--output` 指定输出目录。运行前核验文件和解码像素哈希，运行中核对逆序 ID 重放、独立归一化公式及像素往返。具体产物和使用边界见 [案例与改进分析](improvement-review.md)。

## 重新生成独立参考

建议新建虚拟环境，使用 Python 3.13.9。requirements 固定 NumPy 和 PyTorch CPU 版本，参考生成不需要 GPU。

```text
python -m venv .venv
```

Windows 激活：`.venv\Scripts\Activate.ps1`。Linux 激活：`source .venv/bin/activate`。

```text
python -m pip install -r verification/requirements.txt
python verification/generate_references.py --check
python verification/generate_references.py --check-numerical
python verification/generate_extended_references.py --check-numerical
python verification/generate_real_image_references.py --check-numerical
python verification/check_private_tensor_fields.py
python verification/test_real_images.py
```

`--check` 适合同一参考环境的字节级重建。跨操作系统或 CPU 的 BLAS/FMA 末位差异可能造成字节不同；`--check-numerical` 保持结构和整数严格一致，按已声明容差比较数值，并重新执行独立有限差分审计。CI 使用后者。

需要有意更新参考时执行不带 check 参数的生成器，再检查 diff、运行 `moon fmt` 和完整测试。不要在测试失败时直接用新期望覆盖旧期望。

## 演示与浏览器检查

```text
python tools/serve_demo.py --build
```

浏览器打开 `http://127.0.0.1:8765`。上传图片最长边被缩至 256 像素；增强本身在 MoonBit 中执行。配准采用内置 48×48 平滑非对称样本，默认 300 次迭代、学习率 0.5。

可选的自动浏览器验证：

```text
npm install --no-save --package-lock=false playwright@1.60.0
npx playwright install chromium
node tools/browser_check.mjs
```

服务器需在另一个终端运行。也可以通过 `DEMO_URL` 指向不同本地端口；`BROWSER_EXECUTABLE` 指定已有 Chrome，`PLAYWRIGHT_MODULE` 指定本地模块路径。脚本验证重放、确定性、配准下降和移动布局，并将截图写入忽略目录 `artifacts/`。

## 一条命令完成验证

```text
python tools/verify.py --references --real-images
```

此命令对核心检查、四个后端、可运行示例、JS 编译产物、CLI 流传输、三套参考重建、负向编译、真实照片预处理案例和真实图片依次验证，任何失败立即退出。可用重复的 `--target` 选择部分后端；完整交付应运行四个。真实照片已随仓库附带，测试不联网下载，来源与许可见 [真实图说明](real-images.md)。
