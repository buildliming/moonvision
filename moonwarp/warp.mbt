///|
/// Generate normalized coordinates from row-major [N,2,3] affine transforms.
/// Theta maps output positions into input positions (backward sampling).
/// For a unit output dimension, the base coordinate is zero under both modes.
pub fn affine_grid(
  theta : Array[Double],
  batch : Int,
  out_height : Int,
  out_width : Int,
  align_corners? : Bool = false,
) -> Grid raise WarpError {
  validate_theta(theta, batch)
  let shape = [batch, out_height, out_width, 2]
  let values = Array::make(checked_size(shape, 4), 0.0)
  for n in 0..<batch {
    let t = n * 6
    for y in 0..<out_height {
      let by = normalized_coordinate(y, out_height, align_corners)
      for x in 0..<out_width {
        let bx = normalized_coordinate(x, out_width, align_corners)
        let i = ((n * out_height + y) * out_width + x) * 2
        values[i] = theta[t] * bx + theta[t + 1] * by + theta[t + 2]
        values[i + 1] = theta[t + 3] * bx + theta[t + 4] * by + theta[t + 5]
      }
    }
  }
  validate_values(values)
  { dims: shape, values, }
}

///|
fn check_sample(input : Tensor, grid : Grid) -> Array[Int] raise WarpError {
  if input.dims[0] != grid.dims[0] {
    raise Invalid("input and grid batch dimensions differ")
  }
  let shape = [input.dims[0], input.dims[1], grid.dims[1], grid.dims[2]]
  ignore(checked_size(shape, 4))
  shape
}

// Return a pixel coordinate and its derivative with respect to normalized input.
// Reflection is periodic in normalized coordinates with period four. Reducing
// before unnormalizing avoids overflow and integer conversion for huge inputs.

///|
fn sampling_coordinate(
  value : Double,
  size : Int,
  align_corners : Bool,
  padding : PaddingMode,
) -> (Double, Double) {
  let scale = coordinate_scale(size, align_corners)
  match padding {
    Zeros => (pixel_coordinate(value, size, align_corners), scale)
    Border => {
      if size == 1 || value <= -1.0 {
        return (0.0, 0.0)
      }
      if value >= 1.0 {
        return ((size - 1).to_double(), 0.0)
      }
      let pixel = pixel_coordinate(value, size, align_corners)
      if pixel <= 0.0 {
        (0.0, 0.0)
      } else if pixel >= (size - 1).to_double() {
        ((size - 1).to_double(), 0.0)
      } else {
        (pixel, scale)
      }
    }
    Reflection => {
      if size == 1 {
        return (0.0, 0.0)
      }
      let reduced = value % 4.0
      let pixel = pixel_coordinate(reduced, size, align_corners)
      let low = if align_corners { 0.0 } else { -0.5 }
      let span = if align_corners {
        (size - 1).to_double()
      } else {
        size.to_double()
      }
      let shifted = pixel - low
      let phase = shifted.abs() % (2.0 * span)
      let direction = if shifted < 0.0 { -1.0 } else { 1.0 }
      let (reflected, gradient) = if phase <= span {
        (low + phase, direction * scale)
      } else {
        (low + 2.0 * span - phase, -direction * scale)
      }
      // Exact clipping endpoints are outside for gradients, as in PyTorch.
      if reflected <= 0.0 {
        (0.0, 0.0)
      } else if reflected >= (size - 1).to_double() {
        ((size - 1).to_double(), 0.0)
      } else {
        (reflected, gradient)
      }
    }
  }
}

// Only used after bounding coordinates to the small image support interval.

///|
fn nearest_pixel(value : Double) -> Int {
  let lower = @double.floor(value).to_int()
  let fraction = value - lower.to_double()
  if fraction < 0.5 {
    lower
  } else if fraction > 0.5 {
    lower + 1
  } else if lower % 2 == 0 {
    lower
  } else {
    lower + 1
  }
}

///|
/// Sample NCHW input using normalized NHW2 coordinates. Defaults preserve the
/// original bilinear/zero-padding API. Nearest is suitable for class masks.
/// Both corner modes and all padding modes follow PyTorch grid_sample semantics.
pub fn grid_sample(
  input : Tensor,
  grid : Grid,
  align_corners? : Bool = false,
  mode? : SamplingMode = Bilinear,
  padding? : PaddingMode = Zeros,
) -> Tensor raise WarpError {
  let shape = check_sample(input, grid)
  let channels = shape[1]
  let height = shape[2]
  let width = shape[3]
  let values = Array::make(checked_size(shape, 4), 0.0)
  for n in 0..<shape[0] {
    for y in 0..<height {
      for x in 0..<width {
        let gi = ((n * height + y) * width + x) * 2
        let (px, _) = sampling_coordinate(
          grid.values[gi],
          input.dims[3],
          align_corners,
          padding,
        )
        let (py, _) = sampling_coordinate(
          grid.values[gi + 1],
          input.dims[2],
          align_corners,
          padding,
        )
        // This test precedes every float-to-int conversion, even for huge grids.
        if px >= -1.0 &&
          px <= input.dims[3].to_double() &&
          py >= -1.0 &&
          py <= input.dims[2].to_double() {
          match mode {
            Nearest => {
              let ix = nearest_pixel(px)
              let iy = nearest_pixel(py)
              for c in 0..<channels {
                values[((n * channels + c) * height + y) * width + x] = read_zero(
                  input, n, c, iy, ix,
                )
              }
            }
            Bilinear => {
              let x0 = @double.floor(px).to_int()
              let y0 = @double.floor(py).to_int()
              let dx = px - x0.to_double()
              let dy = py - y0.to_double()
              for c in 0..<channels {
                let a = read_zero(input, n, c, y0, x0)
                let b = read_zero(input, n, c, y0, x0 + 1)
                let d = read_zero(input, n, c, y0 + 1, x0)
                let e = read_zero(input, n, c, y0 + 1, x0 + 1)
                values[((n * channels + c) * height + y) * width + x] = a *
                  (1.0 - dx) *
                  (1.0 - dy) +
                  b * dx * (1.0 - dy) +
                  d * (1.0 - dx) * dy +
                  e * dx * dy
              }
            }
          }
        }
      }
    }
  }
  validate_values(values)
  { dims: shape, values, }
}

///|
fn add_input_grad(
  values : Array[Double],
  input : Tensor,
  n : Int,
  c : Int,
  y : Int,
  x : Int,
  gradient : Double,
) -> Unit {
  let height = input.dims[2]
  let width = input.dims[3]
  if y >= 0 && y < height && x >= 0 && x < width {
    let i = ((n * input.dims[1] + c) * height + y) * width + x
    values[i] = values[i] + gradient
  }
}

///|
/// Backpropagate sampling into image and normalized grid values. Nearest
/// scatters gradients into selected pixels and returns an identically zero
/// grid gradient. Bilinear gradients include clipping/reflection derivatives.
pub fn grid_sample_backward(
  input : Tensor,
  grid : Grid,
  grad_output : Tensor,
  align_corners? : Bool = false,
  mode? : SamplingMode = Bilinear,
  padding? : PaddingMode = Zeros,
) -> SampleGrad raise WarpError {
  let shape = check_sample(input, grid)
  if shape != grad_output.dims {
    raise Invalid("output gradient shape does not match sampling result")
  }
  let channels = shape[1]
  let height = shape[2]
  let width = shape[3]
  let input_values = Array::make(input.values.length(), 0.0)
  let grid_values = Array::make(grid.values.length(), 0.0)
  for n in 0..<shape[0] {
    for y in 0..<height {
      for x in 0..<width {
        let gi = ((n * height + y) * width + x) * 2
        let (px, xscale) = sampling_coordinate(
          grid.values[gi],
          input.dims[3],
          align_corners,
          padding,
        )
        let (py, yscale) = sampling_coordinate(
          grid.values[gi + 1],
          input.dims[2],
          align_corners,
          padding,
        )
        if px >= -1.0 &&
          px <= input.dims[3].to_double() &&
          py >= -1.0 &&
          py <= input.dims[2].to_double() {
          match mode {
            Nearest => {
              let ix = nearest_pixel(px)
              let iy = nearest_pixel(py)
              for c in 0..<channels {
                let g = grad_output.values[((n * channels + c) * height + y) *
                  width +
                  x]
                add_input_grad(input_values, input, n, c, iy, ix, g)
              }
            }
            Bilinear => {
              let x0 = @double.floor(px).to_int()
              let y0 = @double.floor(py).to_int()
              let dx = px - x0.to_double()
              let dy = py - y0.to_double()
              let mut gx = 0.0
              let mut gy = 0.0
              for c in 0..<channels {
                let g = grad_output.values[((n * channels + c) * height + y) *
                  width +
                  x]
                let a = read_zero(input, n, c, y0, x0)
                let b = read_zero(input, n, c, y0, x0 + 1)
                let d = read_zero(input, n, c, y0 + 1, x0)
                let e = read_zero(input, n, c, y0 + 1, x0 + 1)
                add_input_grad(
                  input_values,
                  input,
                  n,
                  c,
                  y0,
                  x0,
                  g * (1.0 - dx) * (1.0 - dy),
                )
                add_input_grad(
                  input_values,
                  input,
                  n,
                  c,
                  y0,
                  x0 + 1,
                  g * dx * (1.0 - dy),
                )
                add_input_grad(
                  input_values,
                  input,
                  n,
                  c,
                  y0 + 1,
                  x0,
                  g * (1.0 - dx) * dy,
                )
                add_input_grad(
                  input_values,
                  input,
                  n,
                  c,
                  y0 + 1,
                  x0 + 1,
                  g * dx * dy,
                )
                // A clipped axis or zero upstream has exactly zero derivative.
                // Avoid overflowing unused differences before multiplying by 0.
                if g != 0.0 && xscale != 0.0 {
                  gx = gx + g * ((b - a) * (1.0 - dy) + (e - d) * dy)
                }
                if g != 0.0 && yscale != 0.0 {
                  gy = gy + g * ((d - a) * (1.0 - dx) + (e - b) * dx)
                }
              }
              grid_values[gi] = gx * xscale
              grid_values[gi + 1] = gy * yscale
            }
          }
        }
      }
    }
  }
  validate_values(input_values)
  validate_values(grid_values)
  {
    input_grad: { dims: input.dims.copy(), values: input_values, },
    grid_grad: { dims: grid.dims.copy(), values: grid_values, },
  }
}

///|
/// Backpropagate an NHW2 grid gradient into row-major [N,2,3] affine coefficients.
pub fn affine_grid_backward(
  grid_grad : Grid,
  align_corners? : Bool = false,
) -> Array[Double] raise WarpError {
  let height = grid_grad.dims[1]
  let width = grid_grad.dims[2]
  let result = Array::make(grid_grad.dims[0] * 6, 0.0)
  for n in 0..<grid_grad.dims[0] {
    let t = n * 6
    for y in 0..<height {
      let by = normalized_coordinate(y, height, align_corners)
      for x in 0..<width {
        let bx = normalized_coordinate(x, width, align_corners)
        let gi = ((n * height + y) * width + x) * 2
        let gx = grid_grad.values[gi]
        let gy = grid_grad.values[gi + 1]
        result[t] = result[t] + gx * bx
        result[t + 1] = result[t + 1] + gx * by
        result[t + 2] = result[t + 2] + gx
        result[t + 3] = result[t + 3] + gy * bx
        result[t + 4] = result[t + 4] + gy * by
        result[t + 5] = result[t + 5] + gy
      }
    }
  }
  validate_values(result)
  result
}

///|
/// Apply affine-grid generation and sampling as one operation. Theta maps
/// output normalized coordinates to input normalized coordinates.
pub fn warp_affine(
  input : Tensor,
  theta : Array[Double],
  out_height : Int,
  out_width : Int,
  align_corners? : Bool = false,
  mode? : SamplingMode = Bilinear,
  padding? : PaddingMode = Zeros,
) -> Tensor raise WarpError {
  let grid = affine_grid(
    theta,
    input.dims[0],
    out_height,
    out_width,
    align_corners~,
  )
  grid_sample(input, grid, align_corners~, mode~, padding~)
}

///|
/// Apply the chain rule through affine-grid generation and sampling.
pub fn warp_affine_backward(
  input : Tensor,
  theta : Array[Double],
  grad_output : Tensor,
  align_corners? : Bool = false,
  mode? : SamplingMode = Bilinear,
  padding? : PaddingMode = Zeros,
) -> WarpGrad raise WarpError {
  let grid = affine_grid(
    theta,
    input.dims[0],
    grad_output.dims[2],
    grad_output.dims[3],
    align_corners~,
  )
  let gradients = grid_sample_backward(
    input,
    grid,
    grad_output,
    align_corners~,
    mode~,
    padding~,
  )
  {
    input_grad: gradients.input_grad,
    theta_grad: affine_grid_backward(gradients.grid_grad, align_corners~),
  }
}
