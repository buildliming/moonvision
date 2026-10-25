# 从 0.1 迁移到 0.2

## 继续有效的用法

- `Sample` 仍只有 `image`、`mask`、`boxes`。关键点通过新增的 `ExtendedSample` 和可选顶层 JSON `keypoints` 使用。
- 旧 `augment` / `replay` 请求继续支持，旧参数默认值与原有变换的随机流保留。
- MoonWarp 默认仍是 bilinear + zeros + `align_corners=false`。旧 `register_affine` 默认行为保留，多尺度使用独立 API。
- 原先单图 `image_to_tensor` / `tensor_to_image` 继续有效。

## 需要留意的变化

### Tensor/Grid 封装

0.1 的 `pub struct` 会暴露可变数组引用，外部可绕过校验。本轮将类型表示封装。读取改用 `tensor.shape()` / `tensor.data()`；修改后重新调用 `Tensor::new(shape,data)`。Grid 同理。返回的数组是副本。

### Replay 版本和操作枚举

不含 Pad 的记录继续使用 v1；含 Pad 使用 v2。读取器支持两版。Letterbox 和 RandomResizedCrop 被解析成确定的 Resize/Pad 或 Crop/Resize。只有支持 v2 的客户端才能读取带 Pad 的新记录。

`Transform` 新增三个变体，外部穷尽匹配需要增加分支。展开后的解析操作总数仍受 256 上限约束，完整计划在像素输出分配前校验。

### 边界、填充和标注

Pad 的 `fill` 必须与通道数一致，`mask_value` / `mask_fill` 是非负类别 ID，可以向 mask 引入显式填充类别。RGBA 需要明确四个填充值。

关键点采用闭区间边缘坐标，像素中心位于 `(x+0.5,y+0.5)`。隐藏点保留索引并持续隐藏。仿射标注变换只在极窄的数值误差区间把原可见点贴回边界，详见 interop 文档。

极小框的面积比较改用轴向比例；浮点运算后退化为零宽/高的框被过滤，输出保持可再次验证的合法样本。

### JSON 资源预算

typed API 的图像/张量上限保留；JSON 还单独限制解码输入、中间图像、输出数组和文本响应。过大请求应拆批或使用 typed API。MoonWarp JSON 单数组最多 262144 个值，MoonAug JSON 通道值预算为 1048576；完整限制见包文档。

### 数值对照

nearest 在半像素位置不连续，bilinear 梯度在整数像素点也可能跳变。独立计算的 affine grid 发生 1ulp 误差时，不能直接要求这些位置的离散像素或导数一致。测试使用同一输入 grid 严格验证采样器，并对远离不光滑位置的仿射链做有限差分。未改变 nearest 的 ties-to-even 语义。
