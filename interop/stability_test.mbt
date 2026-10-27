///|
test "normalization retains finite results across overflowing intermediate differences" {
  let original = [1.0e308, -1.0e308, -1.0e308, 1.0e308]
  let tensor = @moonwarp.Tensor::new([2, 2, 1, 1], original)
  let mean = [-1.0e308, 1.0e308]
  let std = [1.0e308, 1.0e308]
  let result = @interop.normalize(tensor, mean, std)
  assert_eq(result.data(), [2.0, -2.0, 0.0, 0.0])
  assert_eq(tensor.data(), original)
  assert_eq(mean, [-1.0e308, 1.0e308])
  assert_eq(std, [1.0e308, 1.0e308])
}

///|
test "denormalization rescales overflowing products when the final sum is finite" {
  let tensor = @moonwarp.Tensor::new([2, 2, 1, 1], [
    1.0e308, -1.0e308, 5.0e307, -5.0e307,
  ])
  let result = @interop.denormalize(tensor, [-1.0e308, 1.0e308], [2.0, 2.0])
  let values = result.data()
  interop_close(values[0] / 1.0e308, 1.0)
  interop_close(values[1] / 1.0e308, -1.0)
  assert_eq(values[2], 0.0)
  assert_eq(values[3], 0.0)
  assert_eq(tensor.data(), [1.0e308, -1.0e308, 5.0e307, -5.0e307])
  // Cancellation cannot rescue a product larger than the finite range by
  // more than any finite mean. The public error contract still applies.
  expect_interop_error(() => {
    ignore(@interop.denormalize(tensor, [-1.0e308, 1.0e308], [8.0, 8.0]))
  })
  expect_interop_error(() => {
    ignore(@interop.normalize(tensor, [-1.0e308, 1.0e308], [0.5, 0.5]))
  })
}

///|
test "JSON unpack uses the same stable denormalization as the typed API" {
  let response = invoke_interop(
    Json::object({
      "action": "unpack".to_json(),
      "tensor": Json::object({
        "shape": [2, 1, 1, 1].to_json(),
        "data": [1.0e308, 5.0e307].to_json(),
      }),
      "mean": [-1.0e308].to_json(),
      "std": [2.0].to_json(),
    }),
  )
  let expected = [
    Json::object({
      "width": (1).to_json(),
      "height": (1).to_json(),
      "channels": (1).to_json(),
      "data": [255].to_json(),
    }),
    Json::object({
      "width": (1).to_json(),
      "height": (1).to_json(),
      "channels": (1).to_json(),
      "data": [0].to_json(),
    }),
  ]
  assert_eq(response_field(response, "images"), expected.to_json())
}

///|
test "a complete image box survives exact visibility filtering after resize" {
  for (input_size, output_size) in [(5, 23), (11, 3), (13, 33)] {
    let sample : @moonaug.Sample = {
      image: {
        width: input_size,
        height: input_size,
        channels: 1,
        data: Array::make(input_size * input_size, b'\x80'),
      },
      mask: None,
      boxes: [
        {
          x_min: 0.0,
          y_min: 0.0,
          x_max: input_size.to_double(),
          y_max: input_size.to_double(),
          label: "whole image",
        },
      ],
    }
    let result = @interop.warp_sample(
      sample,
      [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      output_size,
      output_size,
      min_visibility=1.0,
    )
    assert_eq(result.boxes.length(), 1)
    assert_eq(result.boxes[0].label, "whole image")
    interop_close(result.boxes[0].x_min, 0.0)
    interop_close(result.boxes[0].y_min, 0.0)
    interop_close(result.boxes[0].x_max, output_size.to_double())
    interop_close(result.boxes[0].y_max, output_size.to_double())
    let shifted = @interop.warp_sample(
      sample,
      [1.0, 0.0, 1.0e-6, 0.0, 1.0, 0.0],
      output_size,
      output_size,
      min_visibility=1.0,
    )
    assert_eq(shifted.boxes, [])
  }
}

///|
test "boundary roundoff tolerance preserves real clipping of tiny boxes" {
  for axis in 0..<2 {
    for upper_edge in [false, true] {
      let lower = if upper_edge { 5.0 - 1.0e-13 } else { 0.0 }
      let upper = if upper_edge { 5.0 } else { 1.0e-13 }
      let sample : @moonaug.Sample = {
        image: {
          width: 5,
          height: 5,
          channels: 1,
          data: Array::make(25, b'\x80'),
        },
        mask: None,
        boxes: [
          {
            x_min: if axis == 0 {
              lower
            } else {
              0.0
            },
            y_min: if axis == 1 {
              lower
            } else {
              0.0
            },
            x_max: if axis == 0 {
              upper
            } else {
              1.0
            },
            y_max: if axis == 1 {
              upper
            } else {
              1.0
            },
            label: "partially clipped tiny box",
          },
        ],
      }
      let theta = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
      theta[if axis == 0 { 2 } else { 5 }] = if upper_edge {
        -1.0e-14
      } else {
        1.0e-14
      }
      // About a quarter of the tiny box is outside. Leave a margin around
      // 0.75 because the edge-coordinate translation also has roundoff.
      for threshold in [0.74, 0.76, 1.0] {
        let output = @interop.warp_sample(
          sample,
          theta,
          5,
          5,
          min_visibility=threshold,
        )
        assert_eq(output.boxes.length(), if threshold < 0.75 { 1 } else { 0 })
      }
    }
  }
}
