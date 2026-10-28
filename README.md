# MoonVision

[![CI](https://github.com/buildliming/moonvision/actions/workflows/ci.yml/badge.svg?branch=default)](https://github.com/buildliming/moonvision/actions/workflows/ci.yml)

纯 MoonBit 计算机视觉工具库：**可复现的标注同步增强 + 可微图像几何变换**。

MoonVision 把视觉数据处理与模型中的空间采样放在一个有明确数值约定的项目中。核心算法不调用 Python、C/C++ 或外部推理引擎；Python/PyTorch 只用于生成独立测试参考。两个包可以分别导入。

| 包 | 用途 | 已实现 |
| --- | --- | --- |
| [`moonaug`](https://github.com/buildliming/moonvision/blob/v0.2.2/moonaug/README.md) | 检测、分割与关键点增强 | 图像/mask/bbox/关键点同步，Pad、Letterbox、RandomResizedCrop、基础变换，按 ID 确定采样与批处理，Replay v1/v2 |
| [`moonwarp`](https://github.com/buildliming/moonvision/blob/v0.2.2/moonwarp/README.md) | 可微采样与图像配准 | 双线性/最近邻 × 零/边缘/反射填充，完整一阶梯度，矩阵组合/求逆/坐标转换，单尺度与抗混叠多尺度配准 |
| [`interop`](https://github.com/buildliming/moonvision/blob/v0.2.2/interop/README.md) | 连接完整视觉流程 | 图像/张量/类别 mask 转换，批次与逐通道归一化，仿射变换同步图像、mask、框和关键点 |

## 快速运行

安装 [MoonBit 工具链](https://www.moonbitlang.com/download/)。本项目验证环境为 `moon 0.1.20260920`、`moonc v0.10.14+7d59c7ec9`，使用新版 `moon.mod` / `moon.pkg` 配置。

```text
git clone --branch default https://github.com/buildliming/moonvision.git
cd moonvision
moon check --target all --deny-warn
moon test
moon run examples/main
moon run examples/pipeline
moon run examples/annotated
```

正常构建和 MoonBit 测试无需 Python、NumPy 或 PyTorch，也没有第三方 Mooncakes 依赖。

### 浏览器演示

需要 Python 3 和 Node.js；无须 npm 安装依赖。

```text
python tools/build_web.py
python tools/serve_demo.py
```

打开 `http://127.0.0.1:8765`。图片只在本机浏览器处理，计算在 Web Worker 中运行。

- **标注同步增强**：切换十组真实照片样本、编辑检测框、上传本地图片、选择 Letterbox/随机缩放裁剪、改变随机种子、导入/导出重放记录。
- **可微图像配准**：切换照片、边界模式和单/多尺度优化，观察六个仿射参数、损失曲线和配准结果。

![标注同步增强演示](https://raw.githubusercontent.com/buildliming/moonvision/v0.2.2/docs/images/moonaug.png)

![可微图像配准演示](https://raw.githubusercontent.com/buildliming/moonvision/v0.2.2/docs/images/moonwarp.png)

### 作为 MoonBit 库使用

Mooncakes 包名为 [`shunge/moonvision`](https://mooncakes.io/docs/shunge/moonvision)，与 GitHub 仓库名分别使用各自平台的账号。安装固定版本：

```text
moon add shunge/moonvision@0.2.2
```

在消费项目的 `moon.pkg` 中导入所需包：

```moonbit
import {
  "shunge/moonvision/moonaug",
  "shunge/moonvision/moonwarp",
  "shunge/moonvision/interop",
}
```

具体 typed API、错误处理和示例见 [MoonAug 文档](https://github.com/buildliming/moonvision/blob/v0.2.2/moonaug/README.md)、[MoonWarp 文档](https://github.com/buildliming/moonvision/blob/v0.2.2/moonwarp/README.md)，完整公共接口由各包的 `pkg.generated.mbti` 记录。

### JSON / Node.js 接口

```text
node tools/moonvision.mjs augment examples/requests/augment.json
node tools/moonvision.mjs warp examples/requests/warp.json
node tools/moonvision.mjs interop examples/requests/annotated.json
node tools/test_bridge.mjs
```

不带文件名时，CLI 逐行读取标准输入的 JSON 请求并逐行返回结果。请求错误返回 `{"ok":false,"error":"..."}`。浏览器与 Node 使用相同的 MoonBit 编译产物。

Node 项目也可以直接导入可复用调用层，不启动 CLI 进程：

```javascript
import { invoke } from './tools/moonvision_client.mjs';
const result = await invoke('augment', request);
if (!result.ok) throw new Error(result.error);
```

先运行 `python tools/build_web.py` 生成编译产物。调用层支持 `augment`、`warp`、`interop`，并将非法模式、序列化或加载失败转换为结构化错误。CLI 按核心的 UTF-16 字符预算增量读取输入，超限行返回错误后可继续处理下一行，并等待输出写入以限制慢消费者造成的积压。单文件失败退出码为 1；JSONL 请求失败通过逐行 `ok:false` 表达，流正常读完仍退出 0。

### 可复用真实照片案例

```text
python -m pip install Pillow==12.3.0
python tools/build_web.py
python examples/real_pipeline.py
```

离线运行人物、猫、咖啡三张真实照片的训练数据预处理：核验来源哈希 → 标注同步 Letterbox/翻转 → 按 ID 重放 → 归一化 NCHW 批次 → 反归一化像素核验。生成模型输入、带来源信息的重放记录、报告和对照拼图到 `artifacts/real-pipeline/`。框和关键点是几何探针，不是人工语义标签；此案例验证预处理流程，不运行识别模型。详见 [案例与多角度改进分析](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/improvement-review.md)。

## 正确性与复现

- 当前源码的 **193 个 MoonBit 测试**在 `wasm-gc`、`wasm`、`js`、`native` 后端验证；每个后端单独执行，包含真实照片像素回归。
- 独立 CPU PyTorch float64 参考覆盖采样前向、输入/grid/theta 梯度、多 batch/channel、非方形与单像素维度、边界和两种角点约定。
- 中央有限差分校验平滑位置的梯度；不可微边界单独对照 PyTorch 的取值约定。
- NumPy/PyTorch 图像参考覆盖增强结果、框几何、整数 mask、alpha 保留、组合变换和重放。
- Node.js 桥接的 **29 项端到端检查**验证 JSON 传输、参考数值、错误输入和重放。当前参考下采样与梯度最大绝对误差为 `8.881784197001252e-16`；不代表所有输入的误差上限。
- Node 调用层与 CLI 的 **19 组检查**覆盖 UTF-16 精确预算、跨块 UTF-8、超限恢复、文件失败、缺构建产物和慢消费者。真实照片预处理案例额外核验 ID 重放、NCHW 索引、归一化与字节往返。
- **6 张原始真实照片、10 组离线样本、173 次真实图接口检查**：人物、动物、物体、室外、纹理、模糊，覆盖 L/RGB/RGBA、竖幅/宽幅/缩略图。像素、掩码、标注、梯度和重放均有独立对照。[真实图说明与结果](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/real-images.md)
- [复杂图像回归](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/real-images.md#退化图像与复杂边界回归)：100 组高斯模糊、失焦、运动模糊、低对比、噪声、遮挡、常量与单像素轴案例，独立核对全部采样组合、梯度、重放、最大 int32 类别值和批次归一化，并检查局部配准的返回值与损失一致性。
- 跨包负向编译验证 Tensor/Grid 内部数组无法被访问或修改；资源预算、非有限数、JSON 严格整数与退化几何均有回归。

```text
python tools/verify.py
```

重新生成参考才需要 Python 的数值依赖：

```text
python -m pip install -r verification/requirements.txt
python verification/generate_references.py --check
python tools/verify.py --references --real-images
```

详见 [复现说明](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/reproduction.md)、[独立参考说明](https://github.com/buildliming/moonvision/blob/v0.2.2/verification/README.md) 和 [验收记录](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/validation.md)。

## 支持边界

MoonAug 处理已解码的灰度/RGB/RGBA 像素和非负整数类别 mask，框使用连续半开区间 xyxy。颜色变换保留 alpha；缩放不执行抗锯齿、线性光转换或预乘 alpha。重放记录保存已解析参数和输入尺寸，不保存源图像身份。

MoonWarp 支持二维、最近邻/双线性、三种填充和一阶梯度。配准是局部优化，依赖纹理及初始位置；不保证任意图像都能对齐。张量、请求/响应文本和操作数量都有明确上限，详见包文档。

本版本不包含通用训练框架、GPU、任意透视变换、三维采样、高阶梯度或图像编解码器。浏览器上传图片使用浏览器解码能力；MoonBit 核心接收像素数组。

## 核心规模与兼容性

`python tools/count_core.py` 可复核核心规模：约 4500 行库源码，其中约 3800 行非空非注释内容。只统计 `moonaug/`、`moonwarp/`、`interop/` 的实现，排除测试、生成接口、图片数据、Python、CLI 与浏览器代码。

0.2 读取旧 Replay v1，含 Pad 的记录使用 v2；旧 `Sample` 结构及默认采样行为保留。Tensor/Grid 的内部数组已封装，原先直接读写 `.dims`/`.values` 的调用需改用 `shape()`/`data()` 与校验构造函数。JSON 新增明确资源预算。详见 [迁移说明](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/migration-0.2.md)。

## 项目结构

```text
moonaug/       标注同步增强、确定性采样、replay、JSON 边界
moonwarp/      张量、采样、解析梯度、配准、JSON 边界
bridge/        两个包的 JavaScript 导出
examples/      MoonBit 可运行示例和 JSON 请求
verification/  独立参考数据、生成器、固定依赖
web/           本地浏览器实验室
tools/         构建、验证、命令行、演示服务器
docs/          数值设计、复现、验收、参赛说明
```

## License / References

沿用仓库原有 [MIT 许可证](https://github.com/buildliming/moonvision/blob/v0.2.2/LICENSE)。实现参考公开语义与论文，测试调用第三方工具作为独立数值基准；第三方工具不随核心库打包。

- [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
- [PyTorch affine_grid](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html)
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- [Albumentations additional targets](https://albumentations.ai/docs/4-advanced-guides/additional-targets/)

开发使用 AI 辅助代码、测试和文档，功能与验收结果以本仓库可运行代码和复现命令为准。参赛摘要见 [docs/hackathon.md](https://github.com/buildliming/moonvision/blob/v0.2.2/docs/hackathon.md)。
