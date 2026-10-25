# 真实图片测试

仓库包含 **6 张原照片、10 组离线样本**。每个 PNG、原始来源文件和解码像素都记录 SHA-256，测试不联网下载。

![真实照片与派生样本](images/real-fixtures.png)

## 图片种类与来源

| 样本 | 内容 | 数据形态 |
| --- | --- | --- |
| astronaut | NASA 人物照片 | 192×192 RGB |
| chelsea | 猫与毛发纹理 | 224×149 RGB |
| coffee | 咖啡杯、瓷器、木纹 | 240×160 RGB |
| rocket | 室外发射场与暗天空 | 256×171 RGB |
| grass | 自然草地纹理 | 192×192 灰度 |
| clock | 运动模糊时钟 | 224×168 灰度 |
| chelsea_alpha | 猫照片的透明通道派生样本 | 160×106 RGBA |
| astronaut_portrait | 人物照片竖幅裁剪 | 96×192 RGB |
| rocket_wide | 室外照片宽幅裁剪 | 256×64 RGB |
| coffee_thumbnail | 极小真实照片缩略图 | 17×11 RGB |

照片由 scikit-image 数据集维护者明确标为 public domain / CC0。作者、原始链接、许可证据和派生步骤见 [逐图归属](../web/samples/ATTRIBUTION.md) 与 [机器可读清单](../web/samples/manifest.json)。透明 alpha 是明确生成的测试渐变，并非原照片的真实抠图。

框、mask 和关键点是叠加在真实像素上的**几何验证探针**，用于检查坐标与像素同步；它们不是人工语义标注，不用于宣称目标检测准确率或分割精度。

## 大图编译产物对照

```text
python -m pip install -r verification/requirements.txt
python tools/build_web.py
python verification/test_real_images.py
```

此脚本以持久 Node 子进程调用真正的 MoonBit 编译产物，完成 **173 次 JSON 接口检查**：

- 每图三条增强流水线，包括基础几何与颜色、Letterbox、RandomResizedCrop。
- NumPy/PyTorch 独立计算像素、mask、bbox 和关键点期望值。mask 必须精确相等，resize/颜色允许一个 uint8 灰阶差。
- 每条流水线检查序列化重放，随机裁剪再次执行检查确定性。
- bilinear/zeros、bilinear/border、bilinear/reflection、nearest/zeros 的真实像素对照。
- 对输入和 theta 的解析梯度对照，以及有明确距离像素节点余量的中央有限差分。
- 整张图片与框、mask、关键点联合仿射变换。
- 双图 batch 的归一化和反归一化，要求字节往返一致。
- 人物、动物和模糊图片的多尺度局部配准，核对返回参数、输出与损失。

实际结果写入 `artifacts/real-images-report.json`，含逐图误差、有限差分余量、配准损失与依赖版本；同时生成 `artifacts/real-images-results.png`。本次验收快照见 [结果 JSON](real-images-results.json)。

[查看真实图变换对照图](images/real-results.png)。

## 最近邻与梯度的非光滑边界

独立 affine_grid 的 1ulp 误差可能跨过半像素最近邻边界，或双线性导数的整数像素节点。因此：

- 采样器在**相同输入 grid**上严格对拍；nearest 像素误差要求为零。
- 两个独立 grid 本身按 `1e-14` 对照，theta 路径远离最近邻节点的像素仍要求精确一致。
- 梯度在相同 grid 上通过 PyTorch autograd 和 affine-grid Jacobian 独立对照。
- 仿射链有限差分使用确定性的小扰动，先确认所有坐标离节点至少 `1e-5` 像素，再以 `1e-8` 差分。报告实际余量。

这保持了不连续位置的真实语义，测试不会把离散像素差直接解释为普通连续误差。

## 四后端真实像素回归

```text
python verification/generate_real_image_references.py --check-numerical
moon test moonwarp --target wasm-gc
moon test moonwarp --target wasm
moon test moonwarp --target js
moon test moonwarp --target native
```

每张图另外生成一组 10×8 的真实像素参考，嵌入 MoonBit 黑盒测试，覆盖反射双线性前向、输入/theta 梯度、nearest/border 与精确类别选择。这些测试直接随四个后端运行，不依赖 MoonBit 图像解码器、Python 或网络。

本次完整真实图验收：增强字节误差最多 1；连续采样最大绝对误差约 `5.56e-14`；梯度最大绝对误差约 `4.1e-16`。这些是所列图片与参数的实测，不能外推为全部输入的统一界限。
