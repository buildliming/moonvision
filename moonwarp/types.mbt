///|
/// Invalid dimensions, nonfinite values, or incompatible image operations.
pub(all) suberror WarpError {
  Invalid(String)
}

///|
/// A validated dense image batch in NCHW order, with Double samples.
/// Construct with `Tensor::new`; accessors return copies.
struct Tensor {
  dims : Array[Int]
  values : Array[Double]
}

///|
/// Normalized sampling coordinates in NHW2 order, with x before y.
/// Coordinates outside [-1,1] are allowed; sampling chooses their padding rule.
struct Grid {
  dims : Array[Int]
  values : Array[Double]
}

///|
/// Gradients returned by bilinear sampling.
pub(all) struct SampleGrad {
  input_grad : Tensor
  grid_grad : Grid
}

///|
/// Gradients through affine-grid generation followed by bilinear sampling.
pub(all) struct WarpGrad {
  input_grad : Tensor
  theta_grad : Array[Double]
}

///|
/// Result of affine image registration. The last loss belongs to returned theta.
pub(all) struct Registration {
  theta : Array[Double]
  output : Tensor
  losses : Array[Double]
}

///|
/// Interpolation used by sampling. Nearest uses ties-to-even pixel rounding.
pub(all) enum SamplingMode {
  Bilinear
  Nearest
}

///|
/// Behavior outside the image: zeros, clipped edge pixels, or repeated reflection.
pub(all) enum PaddingMode {
  Zeros
  Border
  Reflection
}

///|
/// One coarse-to-fine optimization stage. Losses are comparable within this
/// stage only; height and width describe its target tensor.
pub(all) struct RegistrationLevel {
  height : Int
  width : Int
  theta : Array[Double]
  losses : Array[Double]
}

///|
/// Coarse-to-fine registration result in the original normalized coordinates.
/// initial_loss and final_loss are measured at the original target resolution.
pub(all) struct PyramidRegistration {
  theta : Array[Double]
  output : Tensor
  levels : Array[RegistrationLevel]
  initial_loss : Double
  final_loss : Double
}

///|
/// True only for finite floating-point values.
fn finite(value : Double) -> Bool {
  value == value && value != @double.infinity && value != @double.neg_infinity
}

///|
/// Check shape arithmetic before allocation. At most 1,048,576 values per object.
fn checked_size(shape : Array[Int], rank : Int) -> Int raise WarpError {
  if shape.length() != rank {
    raise Invalid("expected rank \{rank}, got \{shape.length()}")
  }
  let mut size = 1
  for dim in shape {
    if dim <= 0 || dim > 4096 {
      raise Invalid("each dimension must be between 1 and 4096")
    }
    if size > 1048576 / dim {
      raise Invalid("tensor exceeds the 1048576-element limit")
    }
    size = size * dim
  }
  size
}

///|
fn validate_values(values : Array[Double]) -> Unit raise WarpError {
  for value in values {
    if !finite(value) {
      raise Invalid("all samples and coordinates must be finite")
    }
  }
}

///|
/// Create a NCHW image tensor. Dimensions are positive and total size is bounded.
pub fn Tensor::new(
  shape : Array[Int],
  data : Array[Double],
) -> Tensor raise WarpError {
  let size = checked_size(shape, 4)
  if size != data.length() {
    raise Invalid("tensor shape does not match data length")
  }
  validate_values(data)
  { dims: shape.copy(), values: data.copy(), }
}

///|
/// Return an independent copy of NCHW dimensions.
pub fn Tensor::shape(self : Tensor) -> Array[Int] {
  self.dims.copy()
}

///|
/// Return an independent copy of NCHW samples.
pub fn Tensor::data(self : Tensor) -> Array[Double] {
  self.values.copy()
}

///|
/// Create an NHW2 sampling grid, where each point contains normalized x then y.
pub fn Grid::new(
  shape : Array[Int],
  data : Array[Double],
) -> Grid raise WarpError {
  let size = checked_size(shape, 4)
  if shape[3] != 2 {
    raise Invalid("grid last dimension must be 2")
  }
  if size != data.length() {
    raise Invalid("grid shape does not match data length")
  }
  validate_values(data)
  { dims: shape.copy(), values: data.copy(), }
}

///|
/// Return an independent copy of NHW2 dimensions.
pub fn Grid::shape(self : Grid) -> Array[Int] {
  self.dims.copy()
}

///|
/// Return an independent copy of interleaved x/y coordinates.
pub fn Grid::data(self : Grid) -> Array[Double] {
  self.values.copy()
}

///|
/// Return a separate identity 2x3 transform for each batch item.
pub fn identity_theta(batch : Int) -> Array[Double] raise WarpError {
  if batch <= 0 || batch > 4096 {
    raise Invalid("batch must be between 1 and 4096")
  }
  let theta = Array::make(batch * 6, 0.0)
  for n in 0..<batch {
    theta[n * 6] = 1.0
    theta[n * 6 + 4] = 1.0
  }
  theta
}

///|
fn validate_theta(theta : Array[Double], batch : Int) -> Unit raise WarpError {
  if batch <= 0 || batch > 4096 || theta.length() != batch * 6 {
    raise Invalid(
      "theta must contain six row-major coefficients per batch item",
    )
  }
  validate_values(theta)
}

///|
fn pixel_coordinate(value : Double, size : Int, align_corners : Bool) -> Double {
  if align_corners {
    (value + 1.0) * (size - 1).to_double() / 2.0
  } else {
    ((value + 1.0) * size.to_double() - 1.0) / 2.0
  }
}

///|
fn coordinate_scale(size : Int, align_corners : Bool) -> Double {
  if align_corners {
    (size - 1).to_double() / 2.0
  } else {
    size.to_double() / 2.0
  }
}

///|
fn normalized_coordinate(
  index : Int,
  size : Int,
  align_corners : Bool,
) -> Double {
  if align_corners {
    if size == 1 {
      0.0
    } else {
      2.0 * index.to_double() / (size - 1).to_double() - 1.0
    }
  } else {
    (2.0 * index.to_double() + 1.0) / size.to_double() - 1.0
  }
}

///|
fn read_zero(input : Tensor, n : Int, c : Int, y : Int, x : Int) -> Double {
  let height = input.dims[2]
  let width = input.dims[3]
  if y < 0 || y >= height || x < 0 || x >= width {
    0.0
  } else {
    input.values[((n * input.dims[1] + c) * height + y) * width + x]
  }
}
