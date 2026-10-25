# MoonVision 参赛说明

## 项目名称

MoonVision：纯 MoonBit 可复现视觉增强与可微几何变换工具库。

## 项目摘要

纯 MoonBit 计算机视觉工具库，包含标注同步、可重放的数据增强，以及支持反向传播的可微图像几何变换。提供图像、检测框和分割掩码处理、仿射网格与双线性采样，配套 PyTorch 对照验证、可复现实例和浏览器演示。

## 真实问题与实现

目标检测和分割数据处理需要图像、检测框、mask 保持几何一致；随机增强还需要可定位、可重现。MoonAug 将空间变换作用于联合样本，记录实际参数，提供按样本确定的随机采样与精确重放。

模型中的空间采样需要对坐标与图像反传，普通 resize/rotate 不能提供这些梯度。MoonWarp 实现仿射网格、双线性采样及解析梯度，通过小型图像配准任务展示可学习变换。interop 把两个模块连接成实际可运行的像素处理流程。

0.2 补齐标注同步 Letterbox/随机缩放裁剪、关键点、批处理与 Replay v2，增加最近邻/边界模式、矩阵工具和多尺度配准。真实照片与派生样本共十组，来源许可与像素哈希随仓库保存；四个后端均有真实像素回归。

## 与已有生态的关系

公开生态中的 millow/pixelforge 已覆盖较多图像处理与基础增强，MoonXi-net/MbTorch 已有训练或张量框架。本项目的范围集中在联合标注的变换语义、版本化重放以及可微坐标采样；不宣称“首个 MoonBit 图像库”或“首个深度学习框架”，也不把公开检索未发现同题当作原创性证明。

## 验收入口

1. `moon run examples/main`：JSON 外部接口示例。
2. `moon run examples/pipeline`：typed API 联合流程。
3. `python tools/verify.py --references --real-images`：完整构建、测试、独立参考和真实图校验。
4. `python tools/serve_demo.py --build`：交互演示。
5. `moon run examples/annotated`：图像、mask、框、关键点与归一化联合流程。
6. [数值设计](architecture.md)、[复现说明](reproduction.md)、[验收记录](validation.md)、[真实图结果](real-images.md)。

## 许可证、来源与 AI 辅助

沿用仓库 MIT 许可证。核心算法以 MoonBit 独立实现；PyTorch、NumPy 用于生成独立对照数据，浏览器负责文件解码与显示。实现语义的论文和官方文档链接列在 README 与设计说明中。

开发过程中使用 Codex/AI 辅助编写代码、生成测试、审查边界和整理文档。公开仓库保留按功能拆分的真实提交；不伪造开发日期、测试结果、基准性能或第三方参赛记录。
