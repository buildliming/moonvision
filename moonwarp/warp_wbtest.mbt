///|
fn near(
  actual : Double,
  expected : Double,
  tolerance? : Double = 0.000001,
) -> Unit raise {
  assert_true((actual - expected).abs() <= tolerance)
}

///|
fn dot(left : Array[Double], right : Array[Double]) -> Double {
  let mut result = 0.0
  for i in 0..<left.length() {
    result = result + left[i] * right[i]
  }
  result
}

///|
test "identity handles NCHW batches channels and both corner conventions" {
  let data = Array::makei(48, i => i.to_double() / 7.0)
  let input = Tensor::new([2, 2, 3, 4], data)
  for align_corners in [false, true] {
    let output = warp_affine(input, identity_theta(2), 3, 4, align_corners~)
    assert_eq(output.shape(), input.shape())
    for i in 0..<data.length() {
      near(output.data()[i], data[i])
    }
  }
}

///|
test "half-pixel zero padding and singleton align-corners semantics" {
  let input = Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
  let corners = Grid::new([1, 1, 3, 2], [-1.0, -1.0, 1.0, 1.0, 5.0, 5.0])
  assert_eq(grid_sample(input, corners).data(), [0.25, 1.0, 0.0])
  assert_eq(grid_sample(input, corners, align_corners=true).data(), [
    1.0, 4.0, 0.0,
  ])
  let single = Tensor::new([1, 1, 1, 1], [9.0])
  let grid = Grid::new([1, 1, 2, 2], [-1000.0, 1000.0, 1.0, 1.0])
  assert_eq(grid_sample(single, grid, align_corners=true).data(), [9.0, 9.0])
  assert_eq(grid_sample(single, grid).data(), [0.0, 2.25])
  assert_eq(
    affine_grid(identity_theta(1), 1, 1, 1, align_corners=true).data(),
    [0.0, 0.0],
  )
}

///|
test "analytic sampling gradients match central finite differences" {
  let shape = [1, 2, 2, 3]
  let data = [0.3, 1.7, -0.5, 2.2, 0.8, 0.4, 0.1, -0.7, 2.1, 1.4, -0.2, 0.9]
  let input = Tensor::new(shape, data)
  let grid_data = [-0.37, 0.13, 0.42, -0.29, 1.14, 0.33, -1.26, -0.11]
  let grid_shape = [1, 2, 2, 2]
  let grid = Grid::new(grid_shape, grid_data)
  let upstream = Tensor::new([1, 2, 2, 2], [
    0.3, -0.4, 0.5, 0.7, 0.2, 0.8, -0.9, 0.1,
  ])
  let epsilon = 0.000001
  for align_corners in [false, true] {
    let gradient = grid_sample_backward(input, grid, upstream, align_corners~)
    for i in 0..<data.length() {
      let plus = data.copy()
      let minus = data.copy()
      plus[i] = plus[i] + epsilon
      minus[i] = minus[i] - epsilon
      let a = grid_sample(Tensor::new(shape, plus), grid, align_corners~)
      let b = grid_sample(Tensor::new(shape, minus), grid, align_corners~)
      let numerical = (
          dot(a.data(), upstream.data()) - dot(b.data(), upstream.data())
        ) /
        (2.0 * epsilon)
      near(gradient.input_grad.data()[i], numerical)
    }
    for i in 0..<grid_data.length() {
      let plus = grid_data.copy()
      let minus = grid_data.copy()
      plus[i] = plus[i] + epsilon
      minus[i] = minus[i] - epsilon
      let a = grid_sample(input, Grid::new(grid_shape, plus), align_corners~)
      let b = grid_sample(input, Grid::new(grid_shape, minus), align_corners~)
      let numerical = (
          dot(a.data(), upstream.data()) - dot(b.data(), upstream.data())
        ) /
        (2.0 * epsilon)
      near(gradient.grid_grad.data()[i], numerical)
    }
  }
}

///|
test "affine chain rule matches finite differences for all six parameters" {
  let input = Tensor::new(
    [1, 1, 3, 4],
    Array::makei(12, i => (i * i + 3 * i % 5).to_double() / 20.0),
  )
  let theta = [0.82, -0.11, 0.037, 0.09, 0.84, -0.023]
  let upstream = Tensor::new([1, 1, 2, 3], [0.1, 0.3, -0.4, 0.6, -0.2, 0.5])
  let epsilon = 0.000001
  for align_corners in [false, true] {
    let gradient = warp_affine_backward(input, theta, upstream, align_corners~)
    for i in 0..<6 {
      let plus = theta.copy()
      let minus = theta.copy()
      plus[i] = plus[i] + epsilon
      minus[i] = minus[i] - epsilon
      let a = warp_affine(input, plus, 2, 3, align_corners~)
      let b = warp_affine(input, minus, 2, 3, align_corners~)
      let numerical = (
          dot(a.data(), upstream.data()) - dot(b.data(), upstream.data())
        ) /
        (2.0 * epsilon)
      near(gradient.theta_grad[i], numerical)
    }
  }
}

///|
test "far outside coordinates and overflowed coordinate arithmetic are safe" {
  let image = Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
  let grid = Grid::new([1, 1, 2, 2], [1.0e300, -1.0e300, 1.0e308, 1.0e308])
  assert_eq(grid_sample(image, grid).data(), [0.0, 0.0])
  let gradient = grid_sample_backward(
    image,
    grid,
    Tensor::new([1, 1, 1, 2], [1.0, 1.0]),
  )
  assert_eq(gradient.input_grad.data(), [0.0, 0.0, 0.0, 0.0])
  assert_eq(gradient.grid_grad.data(), [0.0, 0.0, 0.0, 0.0])
}

///|
test "validated tensor construction rejects bad shapes and nonfinite values" {
  let errors = [
    try {
      ignore(Tensor::new([1, 1, 0, 2], []))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(Tensor::new([1, 1, 2, 2], [1.0]))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(Tensor::new([4096, 4096, 4096, 4096], []))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(Tensor::new([1, 1, 1, 1], [@double.not_a_number]))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(Grid::new([1, 1, 1, 3], [0.0, 0.0, 0.0]))
      false
    } catch {
      Invalid(_) => true
    },
  ]
  for rejected in errors {
    assert_true(rejected)
  }
}

///|
test "public accessors preserve shape and sample invariants" {
  let shape = [1, 1, 1, 1]
  let values = [2.0]
  let tensor = Tensor::new(shape, values)
  shape[3] = 9
  values[0] = @double.infinity
  let copy = tensor.data()
  copy[0] = -8.0
  assert_eq(tensor.shape(), [1, 1, 1, 1])
  assert_eq(tensor.data(), [2.0])
}

///|
test "affine registration reduces image MSE with consistent final parameters" {
  let data = Array::makei(320, i => {
    let x = (i % 20).to_double() / 19.0
    let y = (i / 20).to_double() / 15.0
    let dx = x - 0.35
    let dy = y - 0.40
    let ex = x - 0.75
    let ey = y - 0.65
    0.8 / (1.0 + 80.0 * (dx * dx + dy * dy)) +
    0.6 / (1.0 + 100.0 * (ex * ex + ey * ey))
  })
  let input = Tensor::new([1, 1, 16, 20], data)
  let target = warp_affine(
    input,
    [0.98, -0.04, 0.08, 0.04, 0.98, -0.06],
    16,
    20,
  )
  let result = register_affine(input, target, iterations=250, learning_rate=0.5)
  assert_true(result.losses.length() > 2)
  assert_true(
    result.losses[result.losses.length() - 1] < result.losses[0] * 0.1,
  )
  for i in 1..<result.losses.length() {
    assert_true(result.losses[i] < result.losses[i - 1])
  }
  let output = warp_affine(input, result.theta, 16, 20)
  near(mse_loss(output, result.output), 0.0)
  near(
    mse_loss(result.output, target),
    result.losses[result.losses.length() - 1],
  )
}

///|
test "JSON bridge returns errors and exposes real sampling and gradients" {
  let sample_request =
    #|{"action":"sample","input":{"shape":[1,1,2,2],"data":[1,2,3,4]},"theta":[1,0,0,0,1,0],"out_height":2,"out_width":2}
  let sample = process_json(sample_request)
  let response = @json.parse(sample)
  assert_true(
    response
    is Object(
      {
        "ok": True,
        "output": Object(
          {
            "data": Array(
              [
                Number(1.0, ..),
                Number(2.0, ..),
                Number(3.0, ..),
                Number(4.0, ..),
              ]
            ),
            ..
          }
        ),
        ..
      }
    ),
  )
  for invalid in ["{", "[]", "{}", "{\"action\":\"unknown\"}"] {
    assert_true(
      @json.parse(process_json(invalid)) is Object({ "ok": False, .. }),
    )
  }
}

///|
test "48px registration recovers a moderate rotation and translation" {
  let data = Array::makei(48 * 48, i => {
    let x = (i % 48).to_double() / 47.0
    let y = (i / 48).to_double() / 47.0
    let dx = x - 0.32
    let dy = y - 0.38
    let ex = x - 0.68
    let ey = y - 0.61
    let fx = x - 0.63
    let fy = y - 0.28
    0.8 * @math.exp(-90.0 * (dx * dx + dy * dy)) +
    0.6 * @math.exp(-120.0 * (ex * ex + ey * ey)) +
    0.35 * @math.exp(-160.0 * (fx * fx + fy * fy))
  })
  let input = Tensor::new([1, 1, 48, 48], data)
  let expected = [
    0.9781476007338057, -0.20791169081775934, 0.12, 0.20791169081775934, 0.9781476007338057,
    -0.08,
  ]
  let target = warp_affine(input, expected, 48, 48)
  let result = register_affine(input, target, iterations=300, learning_rate=0.5)
  assert_true(
    result.losses[result.losses.length() - 1] < result.losses[0] * 0.01,
  )
}

///|
test "mismatched batches gradients and optimizer settings raise WarpError" {
  let image = Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
  let grid = affine_grid(identity_theta(2), 2, 2, 2)
  let errors = [
    try {
      ignore(grid_sample(image, grid))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(
        warp_affine_backward(
          image,
          identity_theta(1),
          Tensor::new([1, 2, 2, 2], Array::make(8, 1.0)),
        ),
      )
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(register_affine(image, image, iterations=-1))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(register_affine(image, image, learning_rate=0.0))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(register_affine(image, image, learning_rate=@double.infinity))
      false
    } catch {
      Invalid(_) => true
    },
    try {
      ignore(affine_grid([1.0, 0.0], 1, 2, 2))
      false
    } catch {
      Invalid(_) => true
    },
  ]
  for rejected in errors {
    assert_true(rejected)
  }
  let unchanged = register_affine(image, image, iterations=0)
  assert_eq(unchanged.theta, identity_theta(1))
  assert_eq(unchanged.losses, [0.0])
}

///|
test "JSON registration and backward use the public core numerics" {
  let image = Tensor::new([1, 1, 2, 2], [0.0, 0.2, 0.5, 1.0])
  let input = Json::object({
    "shape": image.shape().to_json(),
    "data": image.data().to_json(),
  })
  let registration = Json::object({
    "action": Json::string("register"),
    "input": input,
    "target": input,
    "iterations": Json::number(0.0),
  })
  assert_true(
    @json.parse(process_json(registration.stringify()))
    is Object(
      {
        "ok": True,
        "initial_loss": Number(0.0, ..),
        "final_loss": Number(0.0, ..),
        ..
      }
    ),
  )
  let backward = Json::object({
    "action": Json::string("backward"),
    "input": input,
    "theta": identity_theta(1).to_json(),
    "grad_output": input,
  })
  let response = @json.parse(process_json(backward.stringify()))
  assert_true(
    response
    is Object({ "ok": True, "theta_grad": Array([_, _, _, _, _, _]), .. }),
  )
  let bad_rate = Json::object({
    "action": Json::string("register"),
    "input": input,
    "target": input,
    "learning_rate": Json::string("bad"),
  })
  assert_true(
    @json.parse(process_json(bad_rate.stringify()))
    is Object({ "ok": False, .. }),
  )
}
