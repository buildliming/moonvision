///|
fn annotation_forward(
  theta : Array[Double],
  input : @moonaug.Image,
  height : Int,
  width : Int,
  align_corners : Bool,
) -> Array[Double] raise InteropError {
  guard theta.length() == 6 else {
    raise Invalid("sample warp requires one six-coefficient affine matrix")
  }
  let backward = @moonwarp.theta_to_pixel_affine(
    theta,
    input.height,
    input.width,
    height,
    width,
    align_corners~,
  ) catch {
    error => raise convert_error(error)
  }
  @moonwarp.invert_theta(backward) catch {
    error => raise convert_error(error)
  }
}

///|
fn mapped_edge(
  matrix : Array[Double],
  x : Double,
  y : Double,
) -> (Double, Double) raise InteropError {
  let ox = matrix[0] * x +
    matrix[1] * y +
    matrix[2] +
    0.5 * (1.0 - matrix[0] - matrix[1])
  let oy = matrix[3] * x +
    matrix[4] * y +
    matrix[5] +
    0.5 * (1.0 - matrix[3] - matrix[4])
  guard ox == ox &&
    oy == oy &&
    ox >= -1.7976931348623157e308 &&
    ox <= 1.7976931348623157e308 &&
    oy >= -1.7976931348623157e308 &&
    oy <= 1.7976931348623157e308 else {
    raise Invalid(
      "annotation coordinates overflowed during affine transformation",
    )
  }
  (ox, oy)
}

///|
fn mapped_boxes(
  boxes : Array[@moonaug.BBox],
  matrix : Array[Double],
  width : Int,
  height : Int,
  min_visibility : Double,
) -> Array[@moonaug.BBox] raise InteropError {
  let result : Array[@moonaug.BBox] = []
  for box in boxes {
    let corners = [
      mapped_edge(matrix, box.x_min, box.y_min),
      mapped_edge(matrix, box.x_max, box.y_min),
      mapped_edge(matrix, box.x_min, box.y_max),
      mapped_edge(matrix, box.x_max, box.y_max),
    ]
    let mut left = corners[0].0
    let mut top = corners[0].1
    let mut right = left
    let mut bottom = top
    for (x, y) in corners {
      left = left.min(x)
      right = right.max(x)
      top = top.min(y)
      bottom = bottom.max(y)
    }
    let x_min = left.max(0.0).min(width.to_double())
    let y_min = top.max(0.0).min(height.to_double())
    let x_max = right.max(0.0).min(width.to_double())
    let y_max = bottom.max(0.0).min(height.to_double())
    if x_max > x_min && y_max > y_min && right > left && bottom > top {
      // Compute relative widths before multiplying; tiny valid boxes must not
      // disappear only because their floating-point area underflows to zero.
      let visible = (x_max - x_min) /
        (right - left) *
        ((y_max - y_min) / (bottom - top))
      if visible >= min_visibility {
        result.push({ x_min, y_min, x_max, y_max, label: box.label, })
      }
    }
  }
  result
}

///|
fn mapped_keypoints(
  points : Array[@moonaug.Keypoint],
  matrix : Array[Double],
  width : Int,
  height : Int,
) -> Array[@moonaug.Keypoint] raise InteropError {
  let result : Array[@moonaug.Keypoint] = []
  let tolerance = width.max(height).to_double() * 1.0e-12
  for point in points {
    let (x, y) = mapped_edge(matrix, point.x, point.y)
    // Inverting equivalent matrices can place an exact edge a few ulps outside.
    // Snap only previously visible points; genuinely outside coordinates survive.
    let (x, y) = if point.visible {
      (
        snap_boundary(x, width.to_double(), tolerance),
        snap_boundary(y, height.to_double(), tolerance),
      )
    } else {
      (x, y)
    }
    result.push({
      x,
      y,
      label: point.label,
      visible: point.visible &&
      x >= 0.0 &&
      x <= width.to_double() &&
      y >= 0.0 &&
      y <= height.to_double(),
    })
  }
  result
}

///|
fn snap_boundary(value : Double, limit : Double, tolerance : Double) -> Double {
  if value >= -tolerance && value <= 0.0 {
    0.0
  } else if value >= limit && value <= limit + tolerance {
    limit
  } else {
    value
  }
}

///|
fn warped_mask(
  mask : Array[Int],
  image : @moonaug.Image,
  grid : @moonwarp.Grid,
  width : Int,
  height : Int,
  align_corners : Bool,
  fill : Int,
) -> Array[Int] raise InteropError {
  let input = mask_to_tensor(mask, image.width, image.height)
  let sampled = @moonwarp.grid_sample(
    input,
    grid,
    mode=@moonwarp.Nearest,
    padding=@moonwarp.Zeros,
    align_corners~,
  ) catch {
    error => raise convert_error(error)
  }
  let result = tensor_to_mask(sampled)
  if fill != 0 {
    let coverage = @moonwarp.Tensor::new(
      [1, 1, image.height, image.width],
      Array::make(image.width * image.height, 1.0),
    ) catch {
      error => raise convert_error(error)
    }
    let inside = @moonwarp.grid_sample(
      coverage,
      grid,
      mode=@moonwarp.Nearest,
      padding=@moonwarp.Zeros,
      align_corners~,
    ) catch {
      error => raise convert_error(error)
    }
    let values = inside.data()
    for index in 0..<(width * height) {
      if values[index] == 0.0 {
        result[index] = fill
      }
    }
  }
  result
}

///|
fn plan_targets(
  sample : @moonaug.Sample,
  theta : Array[Double],
  height : Int,
  width : Int,
  align_corners : Bool,
  min_visibility : Double,
  mask_fill : Int,
) -> (Array[@moonaug.BBox], Array[Double]) raise InteropError {
  @moonaug.validate(sample) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
  guard min_visibility == min_visibility &&
    min_visibility >= 0.0 &&
    min_visibility <= 1.0 else {
    raise Invalid("min_visibility must be finite and in [0,1]")
  }
  guard mask_fill >= 0 else {
    raise Invalid("mask_fill must be a non-negative category ID")
  }
  ignore(compatible_size([1, sample.image.channels, height, width]))
  ignore(
    compatible_size([
      1,
      sample.image.channels,
      sample.image.height,
      sample.image.width,
    ]),
  )
  // The NHW2 grid can exceed the tensor budget even for a valid gray image.
  guard height <= 524288 / width else {
    raise Invalid("affine grid exceeds the 1048576-value limit")
  }
  let matrix = annotation_forward(
    theta,
    sample.image,
    height,
    width,
    align_corners,
  )
  let boxes = mapped_boxes(sample.boxes, matrix, width, height, min_visibility)
  (boxes, matrix)
}

///|
fn execute_targets(
  sample : @moonaug.Sample,
  boxes : Array[@moonaug.BBox],
  theta : Array[Double],
  height : Int,
  width : Int,
  align_corners : Bool,
  mask_fill : Int,
) -> @moonaug.Sample raise InteropError {
  let grid = @moonwarp.affine_grid(theta, 1, height, width, align_corners~) catch {
    error => raise convert_error(error)
  }
  let input = image_to_tensor(sample.image)
  let output = @moonwarp.grid_sample(
    input,
    grid,
    mode=@moonwarp.Bilinear,
    padding=@moonwarp.Zeros,
    align_corners~,
  ) catch {
    error => raise convert_error(error)
  }
  let image = tensor_to_image(output)
  let mask = match sample.mask {
    None => None
    Some(labels) =>
      Some(
        warped_mask(
          labels,
          sample.image,
          grid,
          width,
          height,
          align_corners,
          mask_fill,
        ),
      )
  }
  let result : @moonaug.Sample = { image, mask, boxes, }
  @moonaug.validate(result) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
  result
}

///|
/// Warp an entire annotated sample using a normalized output-to-input matrix.
/// Images use bilinear/zeros; categorical masks use nearest/zeros plus mask_fill.
/// Input box edge coordinates are mapped with the inverse pixel-center matrix,
/// enclosed in output axis-aligned boxes, clipped and filtered by visible AABB
/// fraction. The matrix must be invertible, unlike image-only grid_sample.
/// No source buffer is modified. Output buffers obey the shared tensor limit.
pub fn warp_sample(
  sample : @moonaug.Sample,
  theta : Array[Double],
  out_height : Int,
  out_width : Int,
  align_corners? : Bool = false,
  min_visibility? : Double = 0.0,
  mask_fill? : Int = 0,
) -> @moonaug.Sample raise InteropError {
  let (boxes, _) = plan_targets(
    sample, theta, out_height, out_width, align_corners, min_visibility, mask_fill,
  )
  execute_targets(
    sample, boxes, theta, out_height, out_width, align_corners, mask_fill,
  )
}

///|
/// Warp image, mask, boxes and edge-coordinate keypoints together. Point order
/// and labels remain stable; an invisible point never becomes visible again.
/// For align_corners=true, singleton axes are rejected by matrix conversion.
pub fn warp_extended(
  sample : @moonaug.ExtendedSample,
  theta : Array[Double],
  out_height : Int,
  out_width : Int,
  align_corners? : Bool = false,
  min_visibility? : Double = 0.0,
  mask_fill? : Int = 0,
) -> @moonaug.ExtendedSample raise InteropError {
  @moonaug.validate_extended(sample) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
  let (boxes, matrix) = plan_targets(
    sample.sample,
    theta,
    out_height,
    out_width,
    align_corners,
    min_visibility,
    mask_fill,
  )
  // Finish keypoint overflow checks before allocating any raster output buffers.
  let keypoints = mapped_keypoints(
    sample.keypoints,
    matrix,
    out_width,
    out_height,
  )
  let output = execute_targets(
    sample.sample,
    boxes,
    theta,
    out_height,
    out_width,
    align_corners,
    mask_fill,
  )
  let result : @moonaug.ExtendedSample = { sample: output, keypoints, }
  @moonaug.validate_extended(result) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
  result
}
