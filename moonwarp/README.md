# MoonWarp

Pure MoonBit differentiable image sampling and affine registration. The numeric
core uses Double values, runs without Python, and supports native, JavaScript,
and Wasm GC. Python/PyTorch is used only to regenerate independent test fixtures.

## Numerical contract

- Images use **NCHW**: `[batch, channels, height, width]`.
- Grids use **NHW2**: each point is normalized **x, y**, in that order.
- `theta` contains a row-major 2×3 matrix for each batch item. It maps normalized
  output coordinates into normalized input coordinates. This is backward
  sampling, so the visual movement is the inverse of a forward point transform.
- Interpolation is bilinear; padding is zero. `align_corners=false` is the default.
  Both corner modes match PyTorch's `affine_grid`/`grid_sample`. Under the default,
  `-1` and `1` denote image edges; under `true`, they denote corner pixel centers.
  A unit output dimension has base coordinate zero.
- `grid_sample_backward` returns gradients for both image values and normalized
  grid coordinates. `affine_grid_backward` returns six gradients per batch item.
  `warp_affine_backward` composes these two operations by the chain rule.
- At integer pixel knots, bilinear derivatives are not unique. MoonWarp follows
  the floor-selected right/bottom convention used by PyTorch's CPU reference.
  Central finite differences are checked away from these nondifferentiable knots.

## Use from MoonBit

Import `buildliming/moonvision/moonwarp` as `@moonwarp` in your package.

```moonbit
let image = @moonwarp.Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
let theta = @moonwarp.identity_theta(1)
let output = @moonwarp.warp_affine(image, theta, 2, 2)
// output.data() == [1.0, 2.0, 3.0, 4.0]

let upstream = @moonwarp.Tensor::new([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
let gradients = @moonwarp.warp_affine_backward(image, theta, upstream)
// gradients.input_grad : Tensor
// gradients.theta_grad : Array[Double]
```

Fallible operations raise `WarpError::Invalid(message)`; callers should use a
fallible function or catch the error. Tensor and grid constructors copy their
inputs, and `shape()`/`data()` return copies so callers cannot invalidate them.

## Local affine registration

```moonbit
let fitted = @moonwarp.register_affine(
  image,
  target,
  iterations=300,
  learning_rate=0.5,
)
// fitted.theta, fitted.output, fitted.losses
```

This optimizes all six coefficients per image using analytic gradients and MSE.
It starts from identity, or optional `initial_theta`. Backtracking accepts only
steps that decrease MSE. `losses[0]` is the initial error; its last element
corresponds to the returned theta and output. Early termination can occur when
the gradient is tiny or a decreasing step cannot be found. This is a local
optimizer: distant poses, unrelated images, and symmetric patterns can require
better initialization. Use normalized image values, usually `[0,1]`.

## JSON bridge

`process_json(request : String) -> String` always returns an object with `ok`.
Invalid JSON, shape mismatches, nonfinite values, unsupported actions, or invalid
optimizer parameters return `{"ok":false,"error":"..."}`.

Sampling with an affine transform:

```json
{
  "action": "sample",
  "input": {"shape": [1,1,2,2], "data": [1,2,3,4]},
  "theta": [1,0,0,0,1,0],
  "out_height": 2,
  "out_width": 2,
  "align_corners": false
}
```

The result includes `output:{shape,data}` and `grid:{shape,data}`. Alternatively
provide a `grid` object instead of `theta`/output dimensions for direct sampling.

Registration uses `action:"register"`, `input`, `target`, and optional
`theta`, `iterations` (default 120), `learning_rate` (default 0.5), and
`align_corners` (default false). Results include `theta`, `output`, `losses`,
`initial_loss`, and `final_loss`.

An affine-chain backward request uses `action:"backward"`, `input`, `theta`,
`grad_output`, and optional `align_corners`. Results include `input_grad` and
`theta_grad`.

## Bounds and validation

Dimensions are positive and at most 4096; each tensor/grid has at most 1,048,576
values. Shape products are checked before allocation, and all numeric inputs and
computed results must be finite. Arbitrarily distant finite grid coordinates
produce zeros without an unsafe float-to-int conversion. Registration is limited
to 16,384 input/output values, 0–500 iterations, and 16,000,000 output
value-iterations. Its learning rate must be in `(0,10]`. JSON requests are limited
to 16,000,000 characters.

This release has no GPU backend, arbitrary computational graph, training framework,
3D sampler, border/reflection padding, or general image codec.

## Verification

```text
moon test moonwarp
moon test moonwarp --target native
moon test moonwarp --target js
```

Independent PyTorch float64 fixtures cover forward samples, input/grid/theta
gradients, multiple batches/channels, non-square shapes, unit dimensions, both
corner modes, and out-of-bounds behavior. Additional tests check central finite
differences, invalid inputs, defensive copies, JSON handling, monotonic
registration, and recovery from a 12-degree rotation with translation on a
48×48 image. See `../verification/README.md` for fixture regeneration.
