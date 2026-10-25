///|
/// Mean squared error over all NCHW samples, rejecting mismatched tensors.
pub fn mse_loss(left : Tensor, right : Tensor) -> Double raise WarpError {
  if left.dims != right.dims {
    raise Invalid("MSE requires equal tensor shapes")
  }
  let mut sum = 0.0
  let mut correction = 0.0
  let count = left.values.length().to_double()
  for i in 0..<left.values.length() {
    let diff = left.values[i] - right.values[i]
    // Divide before multiplying, so a finite mean need not overflow its sum.
    let adjusted = diff / count * diff - correction
    let next = sum + adjusted
    correction = next - sum - adjusted
    sum = next
  }
  let loss = sum
  if !finite(loss) {
    raise Invalid("MSE overflow: rescale input samples")
  }
  loss
}

///|
fn mse_gradient(output : Tensor, target : Tensor) -> Tensor {
  let factor = 2.0 / output.values.length().to_double()
  {
    dims: output.dims.copy(),
    values: Array::makei(output.values.length(), i => {
      factor * (output.values[i] - target.values[i])
    }),
  }
}

///|
/// Fit all six affine coefficients per image to minimize MSE.
/// Uses analytic gradients and backtracking descent. This is local registration:
/// large displacements or unrelated images need an appropriate initial theta.
/// Input samples should normally be normalized to [0,1].
/// Each loss is accepted monotonically; returned output and theta always agree.
pub fn register_affine(
  input : Tensor,
  target : Tensor,
  iterations? : Int = 120,
  learning_rate? : Double = 0.5,
  initial_theta? : Array[Double],
  align_corners? : Bool = false,
  padding? : PaddingMode = Zeros,
) -> Registration raise WarpError {
  if input.dims[0] != target.dims[0] || input.dims[1] != target.dims[1] {
    raise Invalid("registration requires equal batch and channel dimensions")
  }
  if iterations < 0 || iterations > 500 {
    raise Invalid("iterations must be between 0 and 500")
  }
  if !finite(learning_rate) || learning_rate <= 0.0 || learning_rate > 10.0 {
    raise Invalid("learning_rate must be finite and in (0,10]")
  }
  if input.values.length() > 16384 ||
    target.values.length() > 16384 ||
    (iterations > 0 && target.values.length() > 16000000 / iterations) {
    raise Invalid(
      "registration is limited to 16384 output values and 16000000 value-iterations",
    )
  }
  let mut theta = match initial_theta {
    Some(t) => {
      validate_theta(t, input.dims[0])
      t.copy()
    }
    None => identity_theta(input.dims[0])
  }
  let mut output = warp_affine(
    input,
    theta,
    target.dims[2],
    target.dims[3],
    align_corners~,
    padding~,
  )
  let mut loss = mse_loss(output, target)
  let losses = [loss]
  for _ in 0..<iterations {
    let grad_output = mse_gradient(output, target)
    let gradient = warp_affine_backward(
        input,
        theta,
        grad_output,
        align_corners~,
        padding~,
      ).theta_grad
    let mut norm_squared = 0.0
    for value in gradient {
      norm_squared = norm_squared + value * value
    }
    if !finite(norm_squared) {
      raise Invalid("registration gradient overflow: rescale samples")
    }
    if norm_squared < 1.0e-20 {
      break
    }
    let mut step = learning_rate
    let mut accepted = false
    for _ in 0..<16 {
      let candidate = Array::makei(theta.length(), i => {
        theta[i] - step * gradient[i]
      })
      validate_values(candidate)
      let candidate_output = warp_affine(
        input,
        candidate,
        target.dims[2],
        target.dims[3],
        align_corners~,
        padding~,
      )
      let candidate_loss = mse_loss(candidate_output, target)
      if candidate_loss < loss {
        theta = candidate
        output = candidate_output
        loss = candidate_loss
        accepted = true
        break
      }
      step = step * 0.5
    }
    if !accepted {
      break
    }
    losses.push(loss)
  }
  { theta, output, losses, }
}
