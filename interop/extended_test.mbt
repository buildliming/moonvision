///|
fn interop_close(actual : Double, expected : Double) -> Unit raise {
  assert_true((actual - expected).abs() <= 1.0e-10)
}

///|
fn response_field(value : Json, key : String) -> Json {
  match value {
    Object(fields) => fields.get(key).unwrap()
    _ => abort("expected JSON response object")
  }
}

///|
fn invoke_interop(request : Json) -> Json raise {
  let response = @json.parse(@interop.process_json(request.stringify()))
  assert_eq(response_field(response, "ok"), true.to_json())
  response
}

///|
test "batched RGB normalization round trip preserves bytes and ownership" {
  let first : @moonaug.Image = {
    width: 2,
    height: 1,
    channels: 3,
    data: [0, 16, 255, 128, 64, 1],
  }
  let second : @moonaug.Image = { ..first, data: [30, 40, 50, 200, 210, 220], }
  let tensor = @interop.images_to_tensor([first, second])
  assert_eq(tensor.shape(), [2, 3, 1, 2])
  let expected = [0, 128, 16, 64, 255, 1, 30, 200, 40, 210, 50, 220]
  let original = tensor.data()
  for i in 0..<expected.length() {
    interop_close(original[i], expected[i].to_double() / 255.0)
  }
  let mean = [0.485, 0.456, 0.406]
  let std = [0.229, 0.224, 0.225]
  let normalized = @interop.normalize(tensor, mean, std)
  interop_close(normalized.data()[0], -0.485 / 0.229)
  let output = @interop.tensor_to_images(
    @interop.denormalize(normalized, mean, std),
  )
  assert_eq(output, [first, second])
  output[0].data[0] = 255
  assert_eq(first.data[0], b'\x00')
  assert_eq(tensor.data(), original)
}

///|
test "batch shapes aggregate resources and normalization parameters are checked" {
  let image : @moonaug.Image = {
    width: 1,
    height: 1,
    channels: 1,
    data: [127],
  }
  expect_interop_error(() => ignore(@interop.images_to_tensor([])))
  expect_interop_error(() => {
    ignore(@interop.images_to_tensor(Array::make(65, image)))
  })
  expect_interop_error(() => {
    ignore(
      @interop.images_to_tensor([image, { ..image, width: 2, data: [1, 2], }]),
    )
  })
  let large : @moonaug.Image = {
    width: 1024,
    height: 513,
    channels: 1,
    data: Array::make(1024 * 513, b'\x00'),
  }
  expect_interop_error(() => ignore(@interop.images_to_tensor([large, large])))
  let tensor = @interop.image_to_tensor(image)
  for std in [[0.0], [-1.0], [1.0 / 0.0], [0.0 / 0.0], []] {
    expect_interop_error(() => ignore(@interop.normalize(tensor, [0.5], std)))
  }
  expect_interop_error(() => ignore(@interop.normalize(tensor, [], [1.0])))
  expect_interop_error(() => {
    ignore(@interop.normalize(tensor, [1.0 / 0.0], [1.0]))
  })
  expect_interop_error(() => {
    ignore(@interop.normalize(tensor, [0.0 / 0.0], [1.0]))
  })
  let huge = @moonwarp.Tensor::new([1, 1, 1, 1], [1.0e308])
  expect_interop_error(() => ignore(@interop.denormalize(huge, [0.0], [2.0])))
  expect_interop_error(() => ignore(@interop.normalize(huge, [0.0], [0.1])))
  let many = @moonwarp.Tensor::new([65, 1, 1, 1], Array::make(65, 0.0))
  expect_interop_error(() => ignore(@interop.tensor_to_images(many)))
  assert_eq(
    @interop.tensor_to_images(@interop.images_to_tensor(Array::make(64, image))).length(),
    64,
  )
}

///|
test "integer masks preserve full nonnegative int32 range and reject mixed labels" {
  let labels = [0, 1, 16777217, 2147483647]
  let tensor = @interop.mask_to_tensor(labels, 2, 2)
  assert_eq(tensor.data(), [0.0, 1.0, 16777217.0, 2147483647.0])
  let restored = @interop.tensor_to_mask(tensor)
  assert_eq(restored, labels)
  restored[0] = 7
  assert_eq(labels[0], 0)
  for value in [0.5, -1.0, 2147483648.0] {
    let invalid = @moonwarp.Tensor::new([1, 1, 1, 1], [value])
    expect_interop_error(() => ignore(@interop.tensor_to_mask(invalid)))
  }
  expect_interop_error(() => ignore(@interop.mask_to_tensor([-1], 1, 1)))
  expect_interop_error(() => ignore(@interop.mask_to_tensor([1], 2, 1)))
  expect_interop_error(() => {
    ignore(@interop.tensor_to_mask(tensor, batch_index=-1))
  })
  let batch = @moonwarp.Tensor::new([2, 1, 1, 2], [
    0.0, 1.0, 16777217.0, 2147483647.0,
  ])
  assert_eq(@interop.tensor_to_mask(batch, batch_index=1), [
    16777217, 2147483647,
  ])
}

///|
test "right translation synchronizes image mask clipped boxes and point visibility" {
  let sample : @moonaug.Sample = {
    image: {
      width: 4,
      height: 3,
      channels: 1,
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    mask: Some([0, 16777217, 2147483647, 9, 1, 2, 3, 4, 5, 6, 7, 8]),
    boxes: [
      { x_min: 0.0, y_min: 0.0, x_max: 4.0, y_max: 3.0, label: "whole", },
      { x_min: 3.0, y_min: 0.0, x_max: 4.0, y_max: 1.0, label: "outside", },
    ],
  }
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 0.5, y: 0.5, label: "center", visible: true, },
    { x: 3.0, y: 3.0, label: "edge", visible: true, },
    { x: 4.0, y: 1.0, label: "lost", visible: true, },
    { x: 0.0, y: 0.5, label: "hidden", visible: false, },
  ]
  for align_corners in [false, true] {
    let theta = @moonwarp.pixel_affine_to_theta(
      [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
      3,
      4,
      3,
      4,
      align_corners~,
    )
    let result = @interop.warp_extended(
      { sample, keypoints, },
      theta,
      3,
      4,
      align_corners~,
      min_visibility=0.75,
      mask_fill=99,
    )
    assert_eq(result.sample.image.data, [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11])
    assert_eq(
      result.sample.mask,
      Some([99, 0, 16777217, 2147483647, 99, 1, 2, 3, 99, 5, 6, 7]),
    )
    assert_eq(result.sample.boxes.length(), 1)
    assert_eq(result.sample.boxes[0].label, "whole")
    interop_close(result.sample.boxes[0].x_min, 1.0)
    interop_close(result.sample.boxes[0].x_max, 4.0)
    assert_eq(result.keypoints.map(point => point.label), [
      "center", "edge", "lost", "hidden",
    ])
    assert_eq(result.keypoints.map(point => point.visible), [
      true, true, false, false,
    ])
    interop_close(result.keypoints[0].x, 1.5)
    interop_close(result.keypoints[1].x, 4.0)
    interop_close(result.keypoints[2].x, 5.0)
    interop_close(result.keypoints[3].x, 1.0)
    assert_eq(
      @interop.warp_sample(
        sample,
        theta,
        3,
        4,
        align_corners~,
        min_visibility=0.750001,
      ).boxes,
      [],
    )
  }
  assert_eq(sample.image.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  assert_eq(keypoints[2].visible, true)
}

///|
test "clockwise rotation of a nonsquare sample respects half pixel edge conversion" {
  let sample : @moonaug.Sample = {
    image: { width: 3, height: 2, channels: 1, data: [1, 2, 3, 4, 5, 6], },
    mask: Some([10, 20, 30, 40, 50, 60]),
    boxes: [
      { x_min: 0.0, y_min: 0.0, x_max: 1.0, y_max: 1.0, label: "corner", },
    ],
  }
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 0.5, y: 1.5, label: "pixel", visible: true, },
  ]
  for align_corners in [false, true] {
    let theta = @moonwarp.pixel_affine_to_theta(
      [0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
      2,
      3,
      3,
      2,
      align_corners~,
    )
    let result = @interop.warp_extended(
      { sample, keypoints, },
      theta,
      3,
      2,
      align_corners~,
    )
    assert_eq((result.sample.image.width, result.sample.image.height), (2, 3))
    assert_eq(result.sample.image.data, [4, 1, 5, 2, 6, 3])
    assert_eq(result.sample.mask, Some([40, 10, 50, 20, 60, 30]))
    let box = result.sample.boxes[0]
    interop_close(box.x_min, 1.0)
    interop_close(box.y_min, 0.0)
    interop_close(box.x_max, 2.0)
    interop_close(box.y_max, 1.0)
    interop_close(result.keypoints[0].x, 0.5)
    interop_close(result.keypoints[0].y, 0.5)
    assert_eq(result.keypoints[0].label, "pixel")
    assert_eq(result.keypoints[0].visible, true)
  }
}

///|
test "normalized identity scales edges with output size without half pixel drift" {
  let sample : @moonaug.Sample = {
    image: { width: 3, height: 2, channels: 1, data: [10, 20, 30, 40, 50, 60], },
    mask: None,
    boxes: [{ x_min: 0.0, y_min: 0.0, x_max: 3.0, y_max: 2.0, label: "whole", }],
  }
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 0.5, y: 0.5, label: "first", visible: true, },
  ]
  let result = @interop.warp_extended(
    { sample, keypoints, },
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    4,
    6,
  )
  assert_eq(result.sample.boxes, [
    { x_min: 0.0, y_min: 0.0, x_max: 6.0, y_max: 4.0, label: "whole", },
  ])
  interop_close(result.keypoints[0].x, 1.0)
  interop_close(result.keypoints[0].y, 1.0)
}

///|
test "matrix roundoff at closed image edges preserves keypoint visibility" {
  let sample : @moonaug.Sample = {
    image: { width: 4, height: 3, channels: 1, data: Array::make(12, b'\x80'), },
    mask: None,
    boxes: [],
  }
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 0.0, y: 0.0, label: "origin", visible: true, },
    { x: 4.0, y: 3.0, label: "far", visible: true, },
    { x: 0.0, y: 0.0, label: "hidden", visible: false, },
  ]
  let resized = @interop.warp_extended(
    { sample, keypoints, },
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    5,
    6,
  )
  assert_eq(resized.keypoints.map(point => point.visible), [true, true, false])
  assert_eq(resized.keypoints[0].x, 0.0)
  assert_eq(resized.keypoints[0].y, 0.0)
  let reflected = @interop.warp_extended(
    { sample, keypoints, },
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    3,
    4,
  )
  assert_eq(reflected.keypoints.map(point => point.visible), [true, true, false])
  let shifted = @interop.warp_extended(
    { sample, keypoints, },
    [1.0, 0.0, 1.0e-6, 0.0, 1.0, 0.0],
    3,
    4,
  )
  assert_eq(shifted.keypoints[0].visible, false)
  assert_true(shifted.keypoints[0].x < -1.0e-7)
}

///|
test "annotated warps reject singular nonfinite unsafe geometry and resource shapes" {
  let sample : @moonaug.Sample = {
    image: { width: 2, height: 2, channels: 1, data: [1, 2, 3, 4], },
    mask: None,
    boxes: [],
  }
  let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  for
    theta in [
      [],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0 / 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [1.0 / 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ] {
    expect_interop_error(() => ignore(@interop.warp_sample(sample, theta, 2, 2)))
  }
  for size in [(0, 2), (4097, 1), (1024, 1025), (1024, 513)] {
    expect_interop_error(() => {
      ignore(@interop.warp_sample(sample, identity, size.0, size.1))
    })
  }
  for minimum in [-0.1, 1.1, 0.0 / 0.0] {
    expect_interop_error(() => {
      ignore(
        @interop.warp_sample(sample, identity, 2, 2, min_visibility=minimum),
      )
    })
  }
  expect_interop_error(() => {
    ignore(@interop.warp_sample(sample, identity, 2, 2, mask_fill=-1))
  })
  expect_interop_error(() => {
    ignore(@interop.warp_sample(sample, identity, 1, 2, align_corners=true))
  })
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 1.0e308, y: 0.0, label: "extreme", visible: false, },
  ]
  expect_interop_error(() => {
    ignore(
      @interop.warp_extended(
        { sample, keypoints, },
        [0.1, 0.0, 0.0, 0.0, 1.0, 0.0],
        2,
        2,
      ),
    )
  })
  let tiny : @moonaug.Sample = {
    image: { width: 1, height: 1, channels: 1, data: [127], },
    mask: Some([2147483647]),
    boxes: [],
  }
  assert_eq(@interop.warp_sample(tiny, identity, 1, 1), tiny)
}

///|
test "JSON batch normalization round trip and annotated translation match typed contracts" {
  let images = Json::array([
    Json::object({
      "width": (2).to_json(),
      "height": (1).to_json(),
      "channels": (1).to_json(),
      "data": [0, 255].to_json(),
    }),
    Json::object({
      "width": (2).to_json(),
      "height": (1).to_json(),
      "channels": (1).to_json(),
      "data": [64, 128].to_json(),
    }),
  ])
  let packed = invoke_interop(
    Json::object({
      "action": "pack".to_json(),
      "images": images,
      "mean": [0.5].to_json(),
      "std": [0.25].to_json(),
    }),
  )
  let tensor = response_field(packed, "tensor")
  assert_eq(response_field(tensor, "shape"), [2, 1, 1, 2].to_json())
  let unpacked = invoke_interop(
    Json::object({
      "action": "unpack".to_json(),
      "tensor": tensor,
      "mean": [0.5].to_json(),
      "std": [0.25].to_json(),
    }),
  )
  assert_eq(response_field(unpacked, "images"), images)
  let sample : @moonaug.Sample = {
    image: { width: 2, height: 2, channels: 1, data: [10, 20, 30, 40], },
    mask: Some([0, 2147483647, 2, 3]),
    boxes: [
      { x_min: 0.0, y_min: 0.0, x_max: 2.0, y_max: 2.0, label: "object", },
    ],
  }
  let keypoints : Array[@moonaug.Keypoint] = [
    { x: 1.5, y: 0.5, label: "point", visible: true, },
  ]
  let theta = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]
  let expected = @interop.warp_extended(
    { sample, keypoints, },
    theta,
    2,
    2,
    mask_fill=9,
  )
  let response = invoke_interop(
    Json::object({
      "action": "warp_sample".to_json(),
      "sample": @moonaug.sample_to_json(sample),
      "keypoints": keypoints.to_json(),
      "theta": theta.to_json(),
      "out_height": (2).to_json(),
      "out_width": (2).to_json(),
      "mask_fill": (9).to_json(),
    }),
  )
  assert_eq(
    response_field(response, "sample"),
    @moonaug.sample_to_json(expected.sample),
  )
  assert_eq(response_field(response, "keypoints"), expected.keypoints.to_json())
}

///|
test "JSON invalid requests fail structurally including preflight aggregate limits" {
  for
    request in [
      "{", "null", "{}", "{\"action\":\"other\"}", "{\"action\":\"pack\",\"images\":[]}",
      "{\"action\":\"unpack\",\"tensor\":{\"shape\":[1,1,1.5,1],\"data\":[1]}}",
    ] {
    let response = @json.parse(@interop.process_json(request))
    assert_eq(response_field(response, "ok"), false.to_json())
    assert_true(response_field(response, "error") is String(_))
  }
  let image = Json::object({
    "width": (1).to_json(),
    "height": (1).to_json(),
    "channels": (1).to_json(),
    "data": [0].to_json(),
  })
  let incomplete = Json::object({
    "action": "pack".to_json(),
    "images": [image].to_json(),
    "mean": [0.5].to_json(),
  })
  assert_eq(
    response_field(
      @json.parse(@interop.process_json(incomplete.stringify())),
      "ok",
    ),
    false.to_json(),
  )
  // The second image exceeds the aggregate JSON budget. Pixel bytes are not decoded.
  let large = Json::object({
    "width": (512).to_json(),
    "height": (512).to_json(),
    "channels": (1).to_json(),
    "data": Array::make(262144, 0).to_json(),
  })
  let request = Json::object({
    "action": "pack".to_json(),
    "images": [large, large].to_json(),
  })
  let response = @json.parse(@interop.process_json(request.stringify()))
  assert_eq(response_field(response, "ok"), false.to_json())
  assert_eq(
    response_field(response, "error"),
    "JSON packed tensors are limited to 262144 values".to_json(),
  )
}
