# MoonVision image/tensor and annotation interop

Import `buildliming/moonvision/interop` to connect MoonAug's decoded pixels and annotations to MoonWarp's numeric operations. The original single-image functions remain compatible.

```moonbit
let tensor = @interop.image_to_tensor(augmented.sample.image)
let output = @moonwarp.warp_affine(tensor, theta, height, width)
let image = @interop.tensor_to_image(output, batch_index=0)
```

- `image_to_tensor(Image)` converts packed HWC gray/RGB/RGBA bytes into a batch-one NCHW `Tensor`, dividing every channel by 255.
- `tensor_to_image(Tensor, batch_index=0)` selects a batch item, packs channels into HWC, clamps values to `[0,1]`, and rounds `value * 255` half up.
- Alpha is an ordinary channel. No premultiplication, linear-light, or color-space conversion occurs. Conversions preserve input buffers and return independent data.
- Both functions raise catchable `InteropError::Invalid(message)` for incompatible inputs. They validate the common resource envelope **before** allocating/copying pixel values: channels 1/3/4, dimensions 1–4096, and at most 1,048,576 values in the complete NCHW tensor. Image validation also uses MoonAug's contract.

## Batches, normalization and masks

| Function | Contract |
| --- | --- |
| `images_to_tensor(images)` | Pack 1–64 images with identical width, height and channels into NCHW; divide every byte by 255. The complete batch has at most 1,048,576 values. |
| `tensor_to_images(tensor)` | Unpack all batch items into separate HWC byte images; at most 64 items. Clamp and quantize exactly as `tensor_to_image`. |
| `normalize(tensor, mean, std)` | Compute `(value - mean[channel]) / std[channel]` for every batch item, without clamping. |
| `denormalize(tensor, mean, std)` | Compute `value * std[channel] + mean[channel]`. |
| `mask_to_tensor(mask, width, height)` | Pack an HW integer category mask as `[1,1,H,W]`; preserve IDs exactly, including `2147483647`. No division by 255. |
| `tensor_to_mask(tensor, batch_index=0)` | Select one single-channel batch item. Reject fractional, negative and out-of-int32 categories instead of rounding. |

Normalization requires one finite mean and finite positive standard deviation per channel. RGBA needs four explicit parameters. Nonfinite arithmetic results raise `InteropError`. Every returned array is independently owned; inputs are unchanged.

```moonbit
let packed = @interop.images_to_tensor(images)
let model_input = @interop.normalize(packed, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
let restored = @interop.denormalize(model_input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
let decoded = @interop.tensor_to_images(restored)
```

Use nearest sampling for mask tensors. Bilinear sampling can mix category IDs; `tensor_to_mask` rejects nonintegral results but cannot detect every accidental integral mixture.

## Warp all annotations with the image

`warp_sample(sample, theta, out_height, out_width, align_corners=false, min_visibility=0.0, mask_fill=0)` transforms image, mask and boxes together. `warp_extended` accepts an `@moonaug.ExtendedSample::{ sample, keypoints }` and also transforms labeled keypoints. Both return the corresponding sample type.

```moonbit
// Move the image content one pixel right. This pixel matrix maps output to input.
let theta = @moonwarp.pixel_affine_to_theta(
  [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
  sample.image.height, sample.image.width,
  sample.image.height, sample.image.width,
)
let moved = @interop.warp_sample(sample, theta, sample.image.height, sample.image.width, mask_fill=255)
```

- `theta` contains one row-major six-coefficient **normalized output-to-input** affine map, matching MoonWarp. Images use bilinear interpolation and zero padding. Masks use nearest interpolation and zero padding; a separately sampled coverage mask applies `mask_fill` only outside the source, preserving valid category zero.
- Boxes and keypoints use pixel-edge coordinates: the center of pixel `(x,y)` is `(x+0.5,y+0.5)`. Geometry converts edges to pixel-center indices, applies the inverse sampling matrix, then converts back. This handles different input/output sizes and both corner conventions.
- Each transformed box becomes the enclosing axis-aligned box, clipped to the output rectangle. `min_visibility` is the clipped AABB area divided by the un-clipped transformed AABB area, in `[0,1]`. It does not measure the visible area of a rotated polygon. Empty boxes are dropped; surviving labels and order are retained.
- Keypoints keep their labels and indices. Previously invisible points remain invisible; other points become invisible outside the closed output rectangle. For previously visible points, outward roundoff within `1e-12 * max(out_width,out_height)` is snapped to the boundary. Larger excursions retain their coordinates and become invisible.
- Singular or ill-conditioned matrices and nonfinite geometry are rejected. `align_corners=true` rejects any singleton input/output spatial axis because the coordinate conversion is not invertible. The default `false` supports singleton axes.
- Input/output tensors each obey the 1,048,576-value limit. The NHW2 grid separately obeys the same limit, so at most 524,288 output pixels are allowed even for gray images. Shapes, resource budgets and annotation arithmetic are checked before raster output allocation.

The lower-level `image_to_tensor` → `warp_affine` → `tensor_to_image` sequence transforms image values only. Use these annotated warp functions when targets must follow the same geometry.

## JSON and JavaScript bridge

`@interop.process_json(request)` and the compiled JavaScript export `interop_json(request)` accept and return JSON strings. Success responses have `ok:true`; failures return `{ "ok": false, "error": "..." }`. Requests and responses are limited to 16,000,000 UTF-16 code units.

| Action | Request fields | Success fields |
| --- | --- | --- |
| `pack` | `images`, optional paired `mean`/`std` | `tensor: {shape,data}` |
| `unpack` | `tensor: {shape,data}`, optional paired `mean`/`std` | `images` |
| `warp_sample` | `sample`, `theta`, `out_height`, `out_width`; optional `keypoints`, `align_corners`, `min_visibility`, `mask_fill` | `sample`, `keypoints` |

An image object is `{width,height,channels,data}` with packed integer bytes. A sample is `{image,mask,boxes}`; omitted mask/boxes default to `null`/`[]`. A keypoint is `{x,y,label,visible}`. Omitted `keypoints` defaults to `[]`. JSON dimensions and mask IDs must be exact int32 integers. JSON `pack`/`unpack` tensors are limited to 262,144 values, and all pack shapes and their combined budget are checked before image decoding.

```javascript
const packed = JSON.parse(interop_json(JSON.stringify({
  action: 'pack',
  images: [{ width: 2, height: 1, channels: 1, data: [0, 255] }],
  mean: [0.5], std: [0.25],
})));
// packed.tensor is { shape: [1,1,1,2], data: [-2,2] }.
const restored = JSON.parse(interop_json(JSON.stringify({
  action: 'unpack', tensor: packed.tensor, mean: [0.5], std: [0.25],
})));
```

Run `moon test -p buildliming/moonvision/interop` and repeat with `--target js`. Tests cover byte-exact batch round trips, normalization, full int32 category IDs, synchronized translation/rotation, clipped visibility, boundary roundoff, malformed JSON and resource limits. Independent PyTorch/NumPy fixtures additionally check raster sampling and forward annotation geometry.

Run `moon run examples/pipeline` for a typed MoonAug → tensor → MoonWarp → image example. It flips an image with MoonAug, reflects it with MoonWarp, and verifies that the original pixels are restored.
