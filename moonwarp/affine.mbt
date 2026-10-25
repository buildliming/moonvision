///|
fn matrix_batch(theta : Array[Double]) -> Int raise WarpError {
  if theta.length() == 0 || theta.length() % 6 != 0 {
    raise Invalid("affine arrays require six coefficients per batch item")
  }
  let batch = theta.length() / 6
  validate_theta(theta, batch)
  batch
}

///|
fn validate_spatial(height : Int, width : Int) -> Unit raise WarpError {
  if height < 1 || height > 4096 || width < 1 || width > 4096 {
    raise Invalid("spatial dimensions must be between 1 and 4096")
  }
}

///|
fn validate_convert_spatial(
  height : Int,
  width : Int,
  align_corners : Bool,
) -> Unit raise WarpError {
  validate_spatial(height, width)
  if align_corners && (height == 1 || width == 1) {
    raise Invalid(
      "align_corners=true coordinate inversion is undefined on a unit axis",
    )
  }
}

///|
/// Compose batched affine maps as outer(inner(point)). Both arrays have exactly
/// six coefficients per image, in the same coordinate system. No broadcasting.
/// A composed warp samples once; two successive raster warps may interpolate
/// twice, so their pixel values need not equal a single composed warp.
pub fn compose_theta(
  outer : Array[Double],
  inner : Array[Double],
) -> Array[Double] raise WarpError {
  let batch = matrix_batch(outer)
  validate_theta(inner, batch)
  let result = Array::make(outer.length(), 0.0)
  for n in 0..<batch {
    let t = n * 6
    result[t] = outer[t] * inner[t] + outer[t + 1] * inner[t + 3]
    result[t + 1] = outer[t] * inner[t + 1] + outer[t + 1] * inner[t + 4]
    result[t + 2] = outer[t] * inner[t + 2] +
      outer[t + 1] * inner[t + 5] +
      outer[t + 2]
    result[t + 3] = outer[t + 3] * inner[t] + outer[t + 4] * inner[t + 3]
    result[t + 4] = outer[t + 3] * inner[t + 1] + outer[t + 4] * inner[t + 4]
    result[t + 5] = outer[t + 3] * inner[t + 2] +
      outer[t + 4] * inner[t + 5] +
      outer[t + 5]
  }
  validate_values(result)
  result
}

///|
/// Invert batched 2x3 affine maps. A relative determinant threshold rejects
/// singular and nearly singular linear parts. Scaling before the determinant
/// avoids squaring extremely large or small coefficients.
pub fn invert_theta(
  theta : Array[Double],
  epsilon? : Double = 0.000000000001,
) -> Array[Double] raise WarpError {
  let batch = matrix_batch(theta)
  if !finite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0 {
    raise Invalid("inverse epsilon must be finite and in (0,1)")
  }
  let result = Array::make(theta.length(), 0.0)
  for n in 0..<batch {
    let t = n * 6
    let mut scale = theta[t].abs()
    for k in [1, 3, 4] {
      if theta[t + k].abs() > scale {
        scale = theta[t + k].abs()
      }
    }
    if scale == 0.0 {
      raise Invalid("singular affine matrix")
    }
    let a = theta[t] / scale
    let b = theta[t + 1] / scale
    let c = theta[t + 3] / scale
    let d = theta[t + 4] / scale
    let determinant = a * d - b * c
    if determinant.abs() <= epsilon {
      raise Invalid("singular or ill-conditioned affine matrix")
    }
    let ia = d / determinant / scale
    let ib = -b / determinant / scale
    let ic = -c / determinant / scale
    let id = a / determinant / scale
    result[t] = ia
    result[t + 1] = ib
    result[t + 2] = -(ia * theta[t + 2] + ib * theta[t + 5])
    result[t + 3] = ic
    result[t + 4] = id
    result[t + 5] = -(ic * theta[t + 2] + id * theta[t + 5])
  }
  validate_values(result)
  result
}

///|
/// Apply one selected affine matrix to finite points. Pixel or normalized
/// coordinates are both valid, provided points and matrix use the same system.
pub fn transform_points(
  theta : Array[Double],
  points : Array[(Double, Double)],
  batch_index? : Int = 0,
) -> Array[(Double, Double)] raise WarpError {
  let batch = matrix_batch(theta)
  if batch_index < 0 || batch_index >= batch {
    raise Invalid("affine batch_index is out of range")
  }
  if points.length() > 524288 {
    raise Invalid("too many affine points")
  }
  let t = batch_index * 6
  let result = []
  for point in points {
    let (x, y) = point
    if !finite(x) || !finite(y) {
      raise Invalid("point coordinates must be finite")
    }
    let xx = theta[t] * x + theta[t + 1] * y + theta[t + 2]
    let yy = theta[t + 3] * x + theta[t + 4] * y + theta[t + 5]
    if !finite(xx) || !finite(yy) {
      raise Invalid("affine point transform overflow")
    }
    result.push((xx, yy))
  }
  result
}

///|
/// Convert a pixel-center index point to normalized coordinates. Pixel centers
/// run from (0,0) through (width-1,height-1); box-edge coordinates need a separate
/// half-pixel conversion. Unit axes with align_corners=true are not invertible.
pub fn normalize_pixel_point(
  x : Double,
  y : Double,
  height : Int,
  width : Int,
  align_corners? : Bool = false,
) -> (Double, Double) raise WarpError {
  validate_convert_spatial(height, width, align_corners)
  if !finite(x) || !finite(y) {
    raise Invalid("point coordinates must be finite")
  }
  let sx = coordinate_scale(width, align_corners)
  let sy = coordinate_scale(height, align_corners)
  let nx = (x - (width - 1).to_double() / 2.0) / sx
  let ny = (y - (height - 1).to_double() / 2.0) / sy
  if !finite(nx) || !finite(ny) {
    raise Invalid("coordinate normalization overflow")
  }
  (nx, ny)
}

///|
/// Convert a normalized point to pixel-center indices. The coordinate system is
/// the inverse of normalize_pixel_point; unit align-corners axes are rejected.
pub fn denormalize_point(
  x : Double,
  y : Double,
  height : Int,
  width : Int,
  align_corners? : Bool = false,
) -> (Double, Double) raise WarpError {
  validate_convert_spatial(height, width, align_corners)
  if !finite(x) || !finite(y) {
    raise Invalid("point coordinates must be finite")
  }
  let px = x * coordinate_scale(width, align_corners) +
    (width - 1).to_double() / 2.0
  let py = y * coordinate_scale(height, align_corners) +
    (height - 1).to_double() / 2.0
  if !finite(px) || !finite(py) {
    raise Invalid("coordinate denormalization overflow")
  }
  (px, py)
}

///|
/// Convert output-pixel -> input-pixel affine matrices into normalized theta.
/// Pixel coordinates refer to pixel-center indices, not box edges. The formula
/// is N_input * pixel_theta * D_output, with explicit input and output sizes.
pub fn pixel_affine_to_theta(
  pixel_theta : Array[Double],
  input_height : Int,
  input_width : Int,
  output_height : Int,
  output_width : Int,
  align_corners? : Bool = false,
) -> Array[Double] raise WarpError {
  let batch = matrix_batch(pixel_theta)
  validate_convert_spatial(input_height, input_width, align_corners)
  validate_convert_spatial(output_height, output_width, align_corners)
  let ix = coordinate_scale(input_width, align_corners)
  let iy = coordinate_scale(input_height, align_corners)
  let ox = coordinate_scale(output_width, align_corners)
  let oy = coordinate_scale(output_height, align_corners)
  let itx = (input_width - 1).to_double() / 2.0
  let ity = (input_height - 1).to_double() / 2.0
  let otx = (output_width - 1).to_double() / 2.0
  let oty = (output_height - 1).to_double() / 2.0
  let result = Array::make(pixel_theta.length(), 0.0)
  for n in 0..<batch {
    let t = n * 6
    result[t] = pixel_theta[t] * ox / ix
    result[t + 1] = pixel_theta[t + 1] * oy / ix
    result[t + 2] = (
        pixel_theta[t] * otx +
        pixel_theta[t + 1] * oty +
        pixel_theta[t + 2] -
        itx
      ) /
      ix
    result[t + 3] = pixel_theta[t + 3] * ox / iy
    result[t + 4] = pixel_theta[t + 4] * oy / iy
    result[t + 5] = (
        pixel_theta[t + 3] * otx +
        pixel_theta[t + 4] * oty +
        pixel_theta[t + 5] -
        ity
      ) /
      iy
  }
  validate_values(result)
  result
}

///|
/// Convert normalized output->input theta into pixel-center output->input maps.
/// This is the inverse of pixel_affine_to_theta for nondegenerate dimensions.
pub fn theta_to_pixel_affine(
  theta : Array[Double],
  input_height : Int,
  input_width : Int,
  output_height : Int,
  output_width : Int,
  align_corners? : Bool = false,
) -> Array[Double] raise WarpError {
  let batch = matrix_batch(theta)
  validate_convert_spatial(input_height, input_width, align_corners)
  validate_convert_spatial(output_height, output_width, align_corners)
  let ix = coordinate_scale(input_width, align_corners)
  let iy = coordinate_scale(input_height, align_corners)
  let ox = coordinate_scale(output_width, align_corners)
  let oy = coordinate_scale(output_height, align_corners)
  let itx = (input_width - 1).to_double() / 2.0
  let ity = (input_height - 1).to_double() / 2.0
  let otx = (output_width - 1).to_double() / 2.0
  let oty = (output_height - 1).to_double() / 2.0
  let result = Array::make(theta.length(), 0.0)
  for n in 0..<batch {
    let t = n * 6
    let a = theta[t] * ix / ox
    let b = theta[t + 1] * ix / oy
    let c = theta[t + 3] * iy / ox
    let d = theta[t + 4] * iy / oy
    result[t] = a
    result[t + 1] = b
    result[t + 2] = theta[t + 2] * ix + itx - a * otx - b * oty
    result[t + 3] = c
    result[t + 4] = d
    result[t + 5] = theta[t + 5] * iy + ity - c * otx - d * oty
  }
  validate_values(result)
  result
}
