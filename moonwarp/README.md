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
- Interpolation defaults to bilinear and padding defaults to zero.
  `mode=Nearest` uses ties-to-even rounding and is useful for segmentation class
  labels. `padding=Border` replicates edge pixels; `padding=Reflection` repeatedly
  reflects around image boundaries. `align_corners=false` is the default.
  Both corner modes match PyTorch's `affine_grid`/`grid_sample`. Under the default,
  `-1` and `1` denote image edges; under `true`, they denote corner pixel centers.
  A unit output dimension has base coordinate zero.
- `grid_sample_backward` returns gradients for both image values and normalized
  grid coordinates. `affine_grid_backward` returns six gradients per batch item.
  `warp_affine_backward` composes these two operations by the chain rule.
- Nearest sampling scatters input gradients into selected pixels and has zero
  grid/theta gradients. Bilinear border/reflection gradients include the
  coordinate mapping derivative; the derivative is zero at clipping endpoints.
  Clipped axes and zero upstream gradients skip inactive differences, avoiding
  spurious overflow from finite samples with extreme dynamic range.
- At integer pixel knots, bilinear derivatives are not unique. MoonWarp follows
  the floor-selected right/bottom convention used by PyTorch's CPU reference.
  Central finite differences are checked away from these nondifferentiable knots.
- Reflection reduces normalized coordinates modulo four before conversion to
  pixels. This handles huge finite coordinates without integer overflow. Exact
  parity with PyTorch is tested at ordinary coordinates and reflection periods;
  floating-point phase information is inherently limited at extreme magnitudes.

### Comparing nonsmooth operations

Independent affine-grid generators can differ by a few floating-point ULPs
because their arithmetic order differs. That is harmless for continuous
bilinear output values, but nearest sampling can select a different pixel when
the mapped coordinate lies on a half-pixel boundary. Bilinear derivatives can
also differ at integer-pixel knots. These differences should not be hidden by
loosening pixel or gradient tolerances or snapping coordinates.

For strict nearest or sampler-gradient comparisons, supply the **same grid** to
both implementations and separately compare the generated affine grids with a
floating-point tolerance. For end-to-end theta gradient/finite-difference tests,
choose coordinates away from integer knots and verify that the finite-difference
step cannot cross one. The discrete nearest result is specified by the actual
supplied floating-point coordinates, including ties-to-even at exact halves.

## Use from MoonBit

Import `shunge/moonvision/moonwarp` as `@moonwarp` in your package.

```moonbit
let image = @moonwarp.Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
let theta = @moonwarp.identity_theta(1)
let output = @moonwarp.warp_affine(image, theta, 2, 2)
// output.data() == [1.0, 2.0, 3.0, 4.0]

let upstream = @moonwarp.Tensor::new([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
let gradients = @moonwarp.warp_affine_backward(image, theta, upstream)
// gradients.input_grad : Tensor
// gradients.theta_grad : Array[Double]

let mask_output = @moonwarp.warp_affine(
  image, theta, 2, 2, mode=Nearest, padding=Border,
)
```

Fallible operations raise `WarpError::Invalid(message)`; callers should use a
fallible function or catch the error. Tensor and grid constructors copy their
inputs, and `shape()`/`data()` return copies so callers cannot invalidate them.
Their representations are opaque across packages, so internal mutable arrays
are not exposed as fields. `tensor_from_json` and `tensor_to_json` provide
reusable bridges without exposing those arrays.

## Affine coordinate tools

`compose_theta(outer, inner)` returns `outer(inner(point))` for each batch item.
`invert_theta(theta, epsilon=1e-12)` rejects singular or relatively ill-conditioned
linear parts. All matrices contain six row-major coefficients per image, and
composition requires matching batch sizes without broadcasting. Composing
matrices samples once; applying two raster warps interpolates twice and can
therefore produce different pixel values.

`pixel_affine_to_theta(pixel_theta, input_height, input_width, output_height,
output_width, align_corners=false)` converts **output-pixel → input-pixel** maps
into normalized theta. `theta_to_pixel_affine` performs the inverse conversion.
Pixel coordinates refer to center indices `(0,0)..(width-1,height-1)`. Annotation
box edges need their own half-pixel adjustment; do not pass a forward point
transform directly as a backward sampling map.

`normalize_pixel_point` / `denormalize_point` convert individual points using
the same convention. `transform_points(theta, points, batch_index=0)` applies a
selected matrix to `(x,y)` tuples. Conversions reject unit axes when
`align_corners=true`, because that coordinate normalization is not invertible.

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
The optional `padding` defaults to `Zeros`; registration always uses bilinear
sampling because nearest coordinates have zero gradients.
`mse_loss` scales residuals before compensated square accumulation, preserving
representable subnormal means and finite means whose unscaled sum would overflow.
An unrepresentable final loss still raises `WarpError::Invalid`.

## Image pyramids and multiscale registration

`resize_tensor` uses bilinear sampling and border padding by default.
`pyramid_downsample` first applies separable `[1,4,6,4,1]/16` low-pass filtering
with replicated boundaries, then resizes to `ceil(height/2) × ceil(width/2)`.
`build_image_pyramid(input, levels=3)` returns fine-to-coarse tensors and stops
before either next dimension becomes smaller than four. At most six levels can
be requested. Both corner conventions are supported consistently across levels.

```moonbit
let fit = @moonwarp.register_affine_multiscale(
  image, target, levels=3, iterations_per_level=100, learning_rate=0.5,
)
// fit.theta, fit.output, fit.initial_loss, fit.final_loss
// fit.levels : coarse-to-fine RegistrationLevel entries
```

The normalized output-to-input theta transfers across pyramid levels. Each
stage has its own monotonic `losses`, target `height`/`width`, and final `theta`.
Losses from different image scales are different objectives and should not be
plotted as one continuous decreasing loss curve. The final transform is selected
by evaluating the initialization and all completed stages at original resolution,
so `final_loss <= initial_loss`. The selected transform can therefore differ
from the last stage's candidate. This is still local registration rather than a
global search. The multiscale API defaults to `padding=Border`; other padding
modes and `initial_theta` are accepted.

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
Sampling and backward requests accept `mode:"bilinear"|"nearest"` and
`padding:"zeros"|"border"|"reflection"`, with bilinear/zeros defaults.

Registration uses `action:"register"`, `input`, `target`, and optional
`theta`, `iterations` (default 120), `learning_rate` (default 0.5), and
`align_corners` (default false). Results include `theta`, `output`, `losses`,
`initial_loss`, and `final_loss`.
It also accepts `padding`, defaulting to `"zeros"`.

Multiscale registration uses `action:"register_multiscale"`, `input`, `target`,
and optional `theta`, `levels` (3), `iterations_per_level` (80), `learning_rate`
(0.5), `align_corners` (false), and `padding` (`"border"`). Results contain
`theta`, `output`, `initial_loss`, `final_loss`, and `levels`, where each entry
contains `height`, `width`, `theta`, and `losses`.

An affine-chain backward request uses `action:"backward"`, `input`, `theta`,
`grad_output`, and optional `align_corners`. Results include `input_grad` and
`theta_grad`.
Provide `grid` instead of `theta` to backpropagate direct grid sampling; this
returns `input_grad` and `grid_grad`.

## Bounds and validation

Dimensions are positive and at most 4096; each tensor/grid has at most 1,048,576
values. Shape products are checked before allocation, and all numeric inputs and
computed results must be finite. Arbitrarily distant finite grid coordinates
produce zeros without an unsafe float-to-int conversion. Bilinear backward
sampling skips inactive clipped axes, zero upstream derivatives, and zero-weight
row/column differences, so unused extreme values do not invalidate an otherwise
finite edge gradient. Active
nonfinite gradient results are still rejected. Registration is limited
to 16,384 input/output values, 0–500 iterations, and 16,000,000 output
value-iterations. Its learning rate must be in `(0,10]`. JSON requests are limited
to 16,000,000 characters. JSON tensors and grids have an additional 262,144-value
limit each, and generated sample/backward output arrays are preflighted to the
same limit. Shape, raw data length, and theta lengths are checked before numeric
array copies. Sample responses contain at most 524,288 output/grid values.
Serialized responses also have a 16,000,000-character limit. Multiscale
registration keeps the same input/output size and iteration constraints and
preflights the aggregate value-iteration budget over all pyramid stages.

This release has no GPU backend, arbitrary computational graph, training framework,
3D sampler, bicubic sampling, or general image codec.

## Verification

```text
moon test moonwarp
moon test moonwarp --target native
moon test moonwarp --target js
```

Independent PyTorch float64 fixtures cover forward samples, input/grid/theta
gradients, multiple batches/channels, non-square shapes, unit dimensions, both
corner modes, all six interpolation/padding combinations, and out-of-bounds
behavior. NumPy fixtures verify batched composition, inversion, and coordinate
conversion. Independent grouped-convolution and interpolation fixtures check
nonconstant antialiased pyramids, replicated boundaries, odd sizes, multiple
batches/channels, and unit axes under both corner conventions. Additional tests check central finite
differences, invalid inputs, defensive copies, JSON handling, monotonic
registration, and recovery from a 12-degree rotation with translation on a
48×48 image, antialiased pyramids, multiscale traces, and extreme finite
coordinates. See `../verification/README.md` for fixture regeneration.
