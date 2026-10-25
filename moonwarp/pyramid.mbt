///|
/// Resize a NCHW tensor by identity-coordinate sampling. Border padding is the
/// default, avoiding dark edges under the half-pixel convention. For reduction
/// with antialiasing use pyramid_downsample instead.
pub fn resize_tensor(
  input : Tensor,
  out_height : Int,
  out_width : Int,
  align_corners? : Bool = false,
  mode? : SamplingMode = Bilinear,
  padding? : PaddingMode = Border,
) -> Tensor raise WarpError {
  warp_affine(
    input,
    identity_theta(input.dims[0]),
    out_height,
    out_width,
    align_corners~,
    mode~,
    padding~,
  )
}

// Separable [1,4,6,4,1]/16 filtering, with replicated edge pixels.

///|
fn binomial_blur5(input : Tensor) -> Tensor raise WarpError {
  let height = input.dims[2]
  let width = input.dims[3]
  let channels = input.dims[1]
  let horizontal = Array::make(input.values.length(), 0.0)
  let result = Array::make(input.values.length(), 0.0)
  let weights = [1.0, 4.0, 6.0, 4.0, 1.0]
  for n in 0..<input.dims[0] {
    for c in 0..<channels {
      for y in 0..<height {
        for x in 0..<width {
          let mut value = 0.0
          for k in 0..<5 {
            let mut xx = x + k - 2
            if xx < 0 {
              xx = 0
            }
            if xx >= width {
              xx = width - 1
            }
            value = value +
              input.values[((n * channels + c) * height + y) * width + xx] *
              (weights[k] / 16.0)
          }
          horizontal[((n * channels + c) * height + y) * width + x] = value
        }
      }
      for y in 0..<height {
        for x in 0..<width {
          let mut value = 0.0
          for k in 0..<5 {
            let mut yy = y + k - 2
            if yy < 0 {
              yy = 0
            }
            if yy >= height {
              yy = height - 1
            }
            value = value +
              horizontal[((n * channels + c) * height + yy) * width + x] *
              (weights[k] / 16.0)
          }
          result[((n * channels + c) * height + y) * width + x] = value
        }
      }
    }
  }
  validate_values(result)
  { dims: input.dims.copy(), values: result, }
}

///|
/// Low-pass filter then reduce each spatial dimension to ceil(size/2). The
/// separable binomial filter reduces aliasing; interpolation uses border padding.
/// Unit dimensions remain one. NCHW batch and channels are unchanged.
pub fn pyramid_downsample(
  input : Tensor,
  align_corners? : Bool = false,
) -> Tensor raise WarpError {
  let filtered = binomial_blur5(input)
  resize_tensor(
    filtered,
    (input.dims[2] + 1) / 2,
    (input.dims[3] + 1) / 2,
    align_corners~,
  )
}

///|
/// Build an antialiased image pyramid from fine to coarse. At most six levels
/// are requested, and reduction stops before either next dimension drops below
/// four pixels. The first level is the original immutable tensor.
pub fn build_image_pyramid(
  input : Tensor,
  levels? : Int = 3,
  align_corners? : Bool = false,
) -> Array[Tensor] raise WarpError {
  if levels < 1 || levels > 6 {
    raise Invalid("pyramid levels must be between 1 and 6")
  }
  let pyramid = [input]
  let mut current = input
  for _ in 1..<levels {
    let next_height = (current.dims[2] + 1) / 2
    let next_width = (current.dims[3] + 1) / 2
    if next_height < 4 || next_width < 4 {
      break
    }
    current = pyramid_downsample(current, align_corners~)
    pyramid.push(current)
  }
  pyramid
}

///|
/// Register at coarse resolutions before refining at the original resolution.
/// Theta stays in normalized output->input coordinates across all levels.
/// Each level has its own monotonic loss history; different scales are different
/// objectives. The returned theta is the best original-resolution candidate
/// among the initial transform and all completed stages, preventing a harmful
/// coarse local minimum from making the final result worse than initialization.
pub fn register_affine_multiscale(
  input : Tensor,
  target : Tensor,
  levels? : Int = 3,
  iterations_per_level? : Int = 80,
  learning_rate? : Double = 0.5,
  initial_theta? : Array[Double],
  align_corners? : Bool = false,
  padding? : PaddingMode = Border,
) -> PyramidRegistration raise WarpError {
  if input.dims[0] != target.dims[0] || input.dims[1] != target.dims[1] {
    raise Invalid("registration requires equal batch and channel dimensions")
  }
  if input.values.length() > 16384 || target.values.length() > 16384 {
    raise Invalid(
      "multiscale registration is limited to 16384 input/output values",
    )
  }
  if levels < 1 || levels > 6 {
    raise Invalid("pyramid levels must be between 1 and 6")
  }
  if iterations_per_level < 0 || iterations_per_level > 500 {
    raise Invalid("iterations_per_level must be between 0 and 500")
  }
  if !finite(learning_rate) || learning_rate <= 0.0 || learning_rate > 10.0 {
    raise Invalid("learning_rate must be finite and in (0,10]")
  }
  let mut theta = match initial_theta {
    Some(t) => {
      validate_theta(t, input.dims[0])
      t.copy()
    }
    None => identity_theta(input.dims[0])
  }
  // Preflight the entire optimization budget before allocating pyramid images.
  let mut ih = input.dims[2]
  let mut iw = input.dims[3]
  let mut th = target.dims[2]
  let mut tw = target.dims[3]
  let mut level_count = 1
  let mut total_values = target.values.length()
  for _ in 1..<levels {
    let nih = (ih + 1) / 2
    let niw = (iw + 1) / 2
    let nth = (th + 1) / 2
    let ntw = (tw + 1) / 2
    if nih < 4 || niw < 4 || nth < 4 || ntw < 4 {
      break
    }
    ih = nih
    iw = niw
    th = nth
    tw = ntw
    total_values = total_values + input.dims[0] * input.dims[1] * th * tw
    level_count = level_count + 1
  }
  if iterations_per_level > 0 && total_values > 16000000 / iterations_per_level {
    raise Invalid(
      "multiscale aggregate budget exceeds 16000000 value-iterations",
    )
  }
  let inputs = build_image_pyramid(input, levels=level_count, align_corners~)
  let targets = build_image_pyramid(target, levels=level_count, align_corners~)
  let initial_output = warp_affine(
    input,
    theta,
    target.dims[2],
    target.dims[3],
    align_corners~,
    padding~,
  )
  let initial_loss = mse_loss(initial_output, target)
  let mut best_theta = theta.copy()
  let mut best_output = initial_output
  let mut best_loss = initial_loss
  let traces : Array[RegistrationLevel] = []
  for index = level_count - 1; index >= 0; index = index - 1 {
    let stage_target = targets[index]
    let stage = register_affine(
      inputs[index],
      stage_target,
      iterations=iterations_per_level,
      learning_rate~,
      initial_theta=theta,
      align_corners~,
      padding~,
    )
    theta = stage.theta.copy()
    traces.push({
      height: stage_target.dims[2],
      width: stage_target.dims[3],
      theta: theta.copy(),
      losses: stage.losses.copy(),
    })
    let full_output = if index == 0 {
      stage.output
    } else {
      warp_affine(
        input,
        theta,
        target.dims[2],
        target.dims[3],
        align_corners~,
        padding~,
      )
    }
    let full_loss = mse_loss(full_output, target)
    if full_loss < best_loss {
      best_theta = theta.copy()
      best_output = full_output
      best_loss = full_loss
    }
  }
  {
    theta: best_theta,
    output: best_output,
    levels: traces,
    initial_loss,
    final_loss: best_loss,
  }
}
