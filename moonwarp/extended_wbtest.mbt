///|
test "nearest ties are even on positive and negative half pixels" {
  assert_eq(nearest_pixel(-2.5), -2)
  assert_eq(nearest_pixel(-1.5), -2)
  assert_eq(nearest_pixel(-0.5), 0)
  assert_eq(nearest_pixel(0.5), 0)
  assert_eq(nearest_pixel(1.5), 2)
  assert_eq(nearest_pixel(2.5), 2)
  assert_eq(nearest_pixel(3.5), 4)
}

///|
test "nearest preserves supplied coordinates on either side of a floating-point tie" {
  // Real-image affine-grid comparison found these two values one ULP apart.
  // At width 192 they map to 92.50000000000001 and exactly 92.5 pixels.
  // Snapping a near-half coordinate would change the defined nearest result.
  let input = Tensor::new([1, 1, 1, 192], Array::makei(192, i => i.to_double()))
  let grid = Grid::new([1, 1, 3, 2], [
    -0.03124999999999993, 0.0, -0.03125000000000004, 0.0, -0.03125, 0.0,
  ])
  assert_eq(grid_sample(input, grid, mode=Nearest).data(), [93.0, 92.0, 92.0])
  let upstream = Tensor::new([1, 1, 1, 3], [1.0, 2.0, 3.0])
  let gradient = grid_sample_backward(input, grid, upstream, mode=Nearest)
  near(gradient.input_grad.data()[92], 5.0)
  near(gradient.input_grad.data()[93], 1.0)
  assert_eq(gradient.grid_grad.data(), Array::make(6, 0.0))
}

///|
test "all padding and interpolation modes handle huge finite coordinates" {
  let input = Tensor::new([1, 1, 2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let grid = Grid::new([1, 1, 3, 2], [
    1.0e308, -1.0e308, -1.0e300, 1.0e300, 1.0e308, 1.0e308,
  ])
  let upstream = Tensor::new([1, 1, 1, 3], [1.0, 2.0, 3.0])
  for align_corners in [false, true] {
    for padding in [Zeros, Border, Reflection] {
      for mode in [Bilinear, Nearest] {
        let output = grid_sample(input, grid, align_corners~, mode~, padding~)
        let grad = grid_sample_backward(
          input,
          grid,
          upstream,
          align_corners~,
          mode~,
          padding~,
        )
        for value in output.data() {
          assert_true(finite(value))
        }
        for value in grad.input_grad.data() {
          assert_true(finite(value))
        }
        for value in grad.grid_grad.data() {
          assert_true(finite(value))
        }
      }
    }
  }
}

///|
test "clipped and zero-upstream bilinear gradients avoid inactive overflow" {
  let input = Tensor::new([1, 1, 2, 2], [1.0e308, -1.0e308, 1.0e308, -1.0e308])
  for padding in [Border, Reflection] {
    for align_corners in [false, true] {
      let grid = Grid::new([1, 1, 1, 2], [-1.0, -1.0])
      let upstream = Tensor::new([1, 1, 1, 1], [1.0])
      let gradient = grid_sample_backward(
        input,
        grid,
        upstream,
        padding~,
        align_corners~,
      )
      assert_eq(gradient.input_grad.data(), [1.0, 0.0, 0.0, 0.0])
      assert_eq(gradient.grid_grad.data(), [0.0, 0.0])
    }
  }
  let grid = Grid::new([1, 1, 1, 2], [0.0, 0.0])
  let upstream = Tensor::new([1, 1, 1, 1], [0.0])
  for padding in [Zeros, Border, Reflection] {
    let gradient = grid_sample_backward(input, grid, upstream, padding~)
    assert_eq(gradient.input_grad.data(), [0.0, 0.0, 0.0, 0.0])
    assert_eq(gradient.grid_grad.data(), [0.0, 0.0])
  }
}

///|
test "bilinear edge gradients ignore overflowing differences with zero weight" {
  let upstream = Tensor::new([1, 1, 1, 1], [1.0])
  for align_corners in [false, true] {
    let edge = if align_corners { -1.0 } else { -0.5 }
    let scale = if align_corners { 0.5 } else { 1.0 }
    for padding in [Border, Reflection] {
      for horizontal in [false, true] {
        let input = Tensor::new(
          [1, 1, 2, 2],
          if horizontal {
            [1.0, 2.0, -1.0e308, 1.0e308]
          } else {
            [1.0, -1.0e308, 2.0, 1.0e308]
          },
        )
        let grid = Grid::new(
          [1, 1, 1, 2],
          if horizontal {
            [0.0, edge]
          } else {
            [edge, 0.0]
          },
        )
        assert_eq(grid_sample(input, grid, padding~, align_corners~).data(), [
          1.5,
        ])
        let gradient = grid_sample_backward(
          input,
          grid,
          upstream,
          padding~,
          align_corners~,
        )
        assert_eq(
          gradient.input_grad.data(),
          if horizontal {
            [0.5, 0.5, 0.0, 0.0]
          } else {
            [0.5, 0.0, 0.5, 0.0]
          },
        )
        assert_eq(
          gradient.grid_grad.data(),
          if horizontal {
            [scale, 0.0]
          } else {
            [0.0, scale]
          },
        )
      }
    }
  }
}

///|
test "bilinear gradients still reject overflow in an active difference" {
  let grid = Grid::new([1, 1, 1, 2], [0.0, 0.0])
  let upstream = Tensor::new([1, 1, 1, 1], [1.0])
  for
    data in [
      [-1.0e308, 1.0e308, -1.0e308, 1.0e308],
      [-1.0e308, -1.0e308, 1.0e308, 1.0e308],
    ] {
    let input = Tensor::new([1, 1, 2, 2], data)
    assert_true(
      try {
        ignore(grid_sample_backward(input, grid, upstream, padding=Border))
        false
      } catch {
        Invalid(_) => true
      },
    )
  }
}

///|
test "scaled MSE remains finite when its unnormalized sum would overflow" {
  let left = Tensor::new([1, 1, 1, 1024], Array::make(1024, 1.0e153))
  let right = Tensor::new([1, 1, 1, 1024], Array::make(1024, 0.0))
  near(mse_loss(left, right) / 1.0e306, 1.0)
}

///|
test "MSE retains subnormal means without underflowing each contribution" {
  let left = Tensor::new([1, 1, 256, 256], Array::make(65536, 1.0e-160))
  let right = Tensor::new([1, 1, 256, 256], Array::make(65536, 0.0))
  near(mse_loss(left, right) / 1.0e-320, 1.0)
  near(mse_loss(right, left) / 1.0e-320, 1.0)
  assert_eq(mse_loss(left, left), 0.0)
}

///|
test "MSE scales sparse high-dynamic-range residuals and rejects true overflow" {
  let data = Array::make(1024, 1.0e-160)
  data[0] = 1.0e155
  let left = Tensor::new([1, 1, 1, 1024], data)
  let right = Tensor::new([1, 1, 1, 1024], Array::make(1024, 0.0))
  near(mse_loss(left, right) / 9.765625e306, 1.0)
  assert_true(
    try {
      ignore(
        mse_loss(
          Tensor::new([1, 1, 1, 1], [1.0e155]),
          Tensor::new([1, 1, 1, 1], [0.0]),
        ),
      )
      false
    } catch {
      Invalid(_) => true
    },
  )
}

///|
test "affine inverse is stable under uniform coefficient scaling" {
  for scale in [1.0e-150, 1.0, 1.0e150] {
    let matrix = [
      1.2 * scale,
      -0.3 * scale,
      0.2,
      0.4 * scale,
      0.8 * scale,
      -0.1,
    ]
    let composed = compose_theta(matrix, invert_theta(matrix))
    let identity = identity_theta(1)
    for i in 0..<6 {
      near(composed[i], identity[i])
    }
  }
  let singular = try {
    ignore(invert_theta([1.0, 2.0, 3.0, 2.0, 4.0, 5.0]))
    false
  } catch {
    Invalid(_) => true
  }
  assert_true(singular)
}

///|
test "point and matrix coordinate conversions handle unit axes explicitly" {
  for align_corners in [false, true] {
    let normalized = normalize_pixel_point(2.7, 1.3, 5, 9, align_corners~)
    let pixel = denormalize_point(
      normalized.0,
      normalized.1,
      5,
      9,
      align_corners~,
    )
    near(pixel.0, 2.7)
    near(pixel.1, 1.3)
  }
  let single = normalize_pixel_point(0.0, 2.0, 5, 1)
  near(single.0, 0.0)
  let matrix = [1.0, 0.2, 0.3, -0.1, 0.8, -0.4]
  let converted = pixel_affine_to_theta(matrix, 5, 1, 3, 1)
  let restored = theta_to_pixel_affine(converted, 5, 1, 3, 1)
  for i in 0..<6 {
    near(restored[i], matrix[i])
  }
  assert_true(
    try {
      ignore(pixel_affine_to_theta(matrix, 5, 1, 3, 2, align_corners=true))
      false
    } catch {
      Invalid(_) => true
    },
  )
  assert_true(
    try {
      ignore(normalize_pixel_point(0.0, 0.0, 1, 4, align_corners=true))
      false
    } catch {
      Invalid(_) => true
    },
  )
}

///|
test "affine composition order is outer after inner for arbitrary points" {
  let outer = [1.1, -0.3, 2.0, 0.2, 0.8, -1.0]
  let inner = [0.7, 0.4, -3.0, -0.1, 1.2, 4.0]
  let points = [(-1.0, 2.3), (4.5, -6.7), (0.0, 0.0)]
  let combined = transform_points(compose_theta(outer, inner), points)
  let sequential = transform_points(outer, transform_points(inner, points))
  for i in 0..<points.length() {
    near(combined[i].0, sequential[i].0)
    near(combined[i].1, sequential[i].1)
  }
}

///|
test "antialiased pyramids preserve constants and stop at small dimensions" {
  let constant = Tensor::new([1, 2, 33, 17], Array::make(1122, 0.7))
  let reduced = pyramid_downsample(constant)
  assert_eq(reduced.shape(), [1, 2, 17, 9])
  for value in reduced.data() {
    near(value, 0.7)
  }
  let pyramid = build_image_pyramid(constant, levels=6)
  assert_eq(pyramid.length(), 3)
  assert_eq(pyramid[2].shape(), [1, 2, 9, 5])
  let single = Tensor::new([1, 1, 1, 7], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  assert_eq(pyramid_downsample(single).shape(), [1, 1, 1, 4])
  assert_eq(build_image_pyramid(single).length(), 1)
}

///|
test "coarse-to-fine registration records separate losses and consistent output" {
  let image = Tensor::new(
    [1, 1, 48, 48],
    Array::makei(48 * 48, i => {
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
    }),
  )
  let target = warp_affine(
    image,
    [
      0.9781476007338057, -0.20791169081775934, 0.12, 0.20791169081775934, 0.9781476007338057,
      -0.08,
    ],
    48,
    48,
    padding=Border,
  )
  let fit = register_affine_multiscale(
    image,
    target,
    levels=3,
    iterations_per_level=100,
  )
  assert_eq(fit.levels.length(), 3)
  assert_eq(fit.levels[0].height, 12)
  assert_eq(fit.levels[2].width, 48)
  assert_true(fit.final_loss < fit.initial_loss * 0.02)
  for level in fit.levels {
    for i in 1..<level.losses.length() {
      assert_true(level.losses[i] < level.losses[i - 1])
    }
  }
  near(mse_loss(fit.output, target), fit.final_loss)
  near(
    mse_loss(fit.output, warp_affine(image, fit.theta, 48, 48, padding=Border)),
    0.0,
  )
}

///|
test "pyramid registration validates settings and preserves a perfect initial fit" {
  let image = Tensor::new(
    [1, 1, 8, 8],
    Array::makei(64, i => (i % 9).to_double() / 9.0),
  )
  let result = register_affine_multiscale(
    image,
    image,
    levels=6,
    iterations_per_level=0,
  )
  assert_eq(result.levels.length(), 2)
  assert_eq(result.theta, identity_theta(1))
  near(result.initial_loss, 0.0)
  near(result.final_loss, 0.0)
  assert_true(
    try {
      ignore(register_affine_multiscale(image, image, levels=0))
      false
    } catch {
      Invalid(_) => true
    },
  )
  assert_true(
    try {
      ignore(register_affine_multiscale(image, image, iterations_per_level=501))
      false
    } catch {
      Invalid(_) => true
    },
  )
}

///|
test "JSON exposes reusable tensors modes direct-grid gradients and pyramid traces" {
  let raw = Json::object({
    "shape": [1, 1, 4, 4].to_json(),
    "data": Array::makei(16, i => i.to_double() / 16.0).to_json(),
  })
  let tensor = tensor_from_json(raw)
  assert_eq(tensor_from_json(tensor_to_json(tensor)).data(), tensor.data())
  let base = Json::object({
    "action": Json::string("register_multiscale"),
    "input": raw,
    "target": raw,
    "iterations_per_level": Json::number(0.0),
  })
  assert_true(
    @json.parse(process_json(base.stringify()))
    is Object(
      { "ok": True, "levels": Array([_]), "final_loss": Number(0.0, ..), .. }
    ),
  )
  let grid = Json::object({
    "shape": [1, 1, 1, 2].to_json(),
    "data": [0.0, 0.0].to_json(),
  })
  let upstream = Json::object({
    "shape": [1, 1, 1, 1].to_json(),
    "data": [1.0].to_json(),
  })
  let backward = Json::object({
    "action": Json::string("backward"),
    "input": raw,
    "grid": grid,
    "grad_output": upstream,
    "mode": Json::string("nearest"),
    "padding": Json::string("reflection"),
  })
  assert_true(
    @json.parse(process_json(backward.stringify()))
    is Object(
      {
        "ok": True,
        "grid_grad": Object(
          { "data": Array([Number(0.0, ..), Number(0.0, ..)]), .. }
        ),
        ..
      }
    ),
  )
  let invalid = Json::object({
    "action": Json::string("sample"),
    "input": raw,
    "grid": grid,
    "mode": Json::string("unknown"),
  })
  assert_true(
    @json.parse(process_json(invalid.stringify()))
    is Object({ "ok": False, .. }),
  )
}

///|
test "JSON tensor budget supports 256px RGB and rejects size before copying data" {
  let image = Json::object({
    "shape": [1, 3, 256, 256].to_json(),
    "data": Array::make(3 * 256 * 256, 0.25).to_json(),
  })
  let decoded = tensor_from_json(image)
  assert_eq(decoded.shape(), [1, 3, 256, 256])
  assert_eq(decoded.data().length(), 196608)
  let oversized = Json::object({
    "shape": [1, 1, 513, 513].to_json(),
    "data": Json::array([]),
  })
  let size_rejected_first = try {
    ignore(tensor_from_json(oversized))
    false
  } catch {
    Invalid(message) => message.contains("262144")
  }
  assert_true(size_rejected_first)
  let input = Json::object({
    "shape": [1, 1, 1, 1].to_json(),
    "data": [1.0].to_json(),
  })
  let request = Json::object({
    "action": Json::string("sample"),
    "input": input,
    "theta": identity_theta(1).to_json(),
    "out_height": Json::number(512.0),
    "out_width": Json::number(512.0),
  })
  assert_true(
    @json.parse(process_json(request.stringify()))
    is Object({ "ok": False, .. }),
  )
}
