# MoonVision image/tensor interop

Import `buildliming/moonvision/interop` to connect MoonAug's decoded pixels to MoonWarp's numeric operations.

```moonbit
let tensor = @interop.image_to_tensor(augmented.sample.image)
let output = @moonwarp.warp_affine(tensor, theta, height, width)
let image = @interop.tensor_to_image(output, batch_index=0)
```

- `image_to_tensor(Image)` converts packed HWC gray/RGB/RGBA bytes into a batch-one NCHW `Tensor`, dividing every channel by 255.
- `tensor_to_image(Tensor, batch_index=0)` selects a batch item, packs channels into HWC, clamps values to `[0,1]`, and rounds `value * 255` half up.
- Alpha is an ordinary channel. No premultiplication, linear-light, or color-space conversion occurs. Conversions preserve input buffers and return independent data.
- Both functions raise catchable `InteropError::Invalid(message)` for incompatible inputs. They validate the common resource envelope **before** allocating/copying pixel values: channels 1/3/4, dimensions 1–4096, and at most 1,048,576 values in the complete NCHW tensor. Image validation also uses MoonAug's contract.

These functions convert image values only. They do not move masks or boxes through a subsequent MoonWarp operation. Keep annotation geometry consistent with any additional spatial transform at the application layer.

Run `moon run examples/pipeline` for a typed MoonAug → tensor → MoonWarp → image example. It flips an image with MoonAug, reflects it with MoonWarp, and verifies that the original pixels are restored.
