///|
fn fixture() -> Sample {
  {
    image: {
      width: 3,
      height: 2,
      channels: 1,
      data: [b'\x0A', b'\x14', b'\x1E', b'\x28', b'\x32', b'\x3C'],
    },
    mask: Some([1, 2, 3, 4, 5, 6]),
    boxes: [
      { x_min: 0.0, y_min: 0.0, x_max: 2.0, y_max: 1.0, label: "object", },
    ],
  }
}

///|
fn run_transform(
  sample : Sample,
  transform : Transform,
) -> Sample raise AugError {
  augment(sample, [{ transform, probability: 1.0, }], 42, "fixture").sample
}

///|
fn expect_invalid(action : () -> Unit raise AugError) -> Unit raise Error {
  let raised = try {
    action()
    false
  } catch {
    Invalid(_) => true
  }
  assert_true(raised)
}

///|
test "hflip and vflip keep bytes masks and labels aligned" {
  let input = fixture()
  let flipped = run_transform(input, HFlip)
  assert_eq(flipped.image.data, [
    b'\x1E', b'\x14', b'\x0A', b'\x3C', b'\x32', b'\x28',
  ])
  assert_eq(flipped.mask, Some([3, 2, 1, 6, 5, 4]))
  assert_eq(flipped.boxes, [
    { x_min: 1.0, y_min: 0.0, x_max: 3.0, y_max: 1.0, label: "object", },
  ])
  assert_eq(run_transform(flipped, HFlip), input)
  let vertical = run_transform(input, VFlip)
  assert_eq(vertical.mask, Some([4, 5, 6, 1, 2, 3]))
  assert_eq(vertical.boxes[0].y_min, 1.0)
  assert_eq(run_transform(vertical, VFlip), input)
}

///|
test "clockwise rotation handles nonsquare images and negative turns" {
  let input = fixture()
  let rotated = run_transform(input, Rotate90(1))
  assert_eq((rotated.image.width, rotated.image.height), (2, 3))
  assert_eq(rotated.image.data, [
    b'\x28', b'\x0A', b'\x32', b'\x14', b'\x3C', b'\x1E',
  ])
  assert_eq(rotated.mask, Some([4, 1, 5, 2, 6, 3]))
  assert_eq(rotated.boxes, [
    { x_min: 1.0, y_min: 0.0, x_max: 2.0, y_max: 2.0, label: "object", },
  ])
  assert_eq(run_transform(rotated, Rotate90(-1)), input)
  assert_eq(run_transform(input, Rotate90(-2147483648)), input)
}

///|
test "crop clips boxes and filters labels at inclusive visibility threshold" {
  let input = {
    ..fixture(),
    boxes: [
      { x_min: 0.0, y_min: 0.0, x_max: 2.0, y_max: 1.0, label: "half", },
      { x_min: 0.0, y_min: 1.0, x_max: 0.5, y_max: 2.0, label: "gone", },
      { x_min: 1.0, y_min: 0.0, x_max: 3.0, y_max: 2.0, label: "full", },
    ],
  }
  let pipeline = [{ transform: Crop(1, 0, 2, 2), probability: 1.0, }]
  let output = augment(input, pipeline, 0, "", min_visibility=0.5).sample
  assert_eq(output.mask, Some([2, 3, 5, 6]))
  assert_eq(output.boxes.map(fn(box) { box.label }), ["half", "full"])
  assert_eq(output.boxes[0].x_min, 0.0)
  assert_eq(output.boxes[0].x_max, 1.0)
  let strict = augment(input, pipeline, 0, "", min_visibility=0.50001).sample
  assert_eq(strict.boxes.map(fn(box) { box.label }), ["full"])
}

///|
test "half-pixel bilinear resize and exact-nearest mask use different sampling" {
  let sample : Sample = {
    image: { width: 2, height: 1, channels: 1, data: [b'\x00', b'\x64'], },
    mask: Some([7, 42]),
    boxes: [{ x_min: 0.0, y_min: 0.0, x_max: 1.0, y_max: 1.0, label: "left", }],
  }
  let output = run_transform(sample, Resize(4, 1))
  assert_eq(output.image.data, [b'\x00', b'\x19', b'\x4B', b'\x64'])
  assert_eq(output.mask, Some([7, 7, 42, 42]))
  assert_eq(output.boxes[0].x_max, 2.0)
  let down = run_transform(sample, Resize(1, 1))
  assert_eq(down.image.data, [b'\x32'])
  assert_eq(down.mask, Some([42]))
}

///|
test "RGB RGBA and grayscale color transforms clamp round and preserve alpha" {
  let rgba : Sample = {
    image: {
      width: 1,
      height: 1,
      channels: 4,
      data: [b'\x32', b'\x64', b'\xC8', b'\x3F'],
    },
    mask: None,
    boxes: [],
  }
  assert_eq(run_transform(rgba, Brightness(1.5)).image.data, [
    b'\x4B', b'\x96', b'\xFF', b'\x3F',
  ])
  assert_eq(run_transform(rgba, Contrast(0.0)).image.data, [
    b'\x80', b'\x80', b'\x80', b'\x3F',
  ])
  let rgb : Sample = {
    ..rgba,
    image: {
      width: 1,
      height: 1,
      channels: 3,
      data: [b'\x01', b'\x03', b'\x05'],
    },
  }
  assert_eq(run_transform(rgb, Brightness(0.5)).image.data, [
    b'\x01', b'\x02', b'\x03',
  ])
  assert_eq(
    run_transform(fixture(), Brightness(0.0)).image.data,
    Array::make(6, b'\x00'),
  )
}

///|
test "seed and sample ID are deterministic and random operations resolve into replay" {
  let input = fixture()
  let pipeline = [
    { transform: HFlip, probability: 0.4, },
    { transform: RandomCrop(2, 1), probability: 1.0, },
    { transform: VFlip, probability: 0.0, },
    { transform: Brightness(1.2), probability: 0.7, },
  ]
  let a = augment(input, pipeline, 123, "sample-你好")
  assert_eq(a, augment(input, pipeline, 123, "sample-你好"))
  assert_eq(replay(input, a.replay), a.sample)
  let record = replay_from_json(
    @json.parse(replay_to_json(a.replay).stringify()),
  )
  assert_eq(record, a.replay)
  assert_eq(replay(input, record), a.sample)
  let variants : Array[String] = []
  for seed in 0..<24 {
    let result = augment(input, pipeline, seed, "sample")
    let encoded = replay_to_json(result.replay).stringify()
    if !variants.contains(encoded) {
      variants.push(encoded)
    }
  }
  assert_true(variants.length() >= 4)
}

///|
test "version 1 random stream is pinned across compilation targets" {
  let rng = Rng::new(123, "sample")
  assert_eq([rng.next(), rng.next(), rng.next(), rng.next(), rng.next()], [
    346799820U, 221503957U, 1344689009U, 3545232629U, 2549678525U,
  ])
}

///|
test "operation limits accept the boundary and reject excess in typed and JSON APIs" {
  let input = fixture()
  let skipped = { transform: HFlip, probability: 0.0, }
  assert_eq(augment(input, Array::make(256, skipped), 0, "").sample, input)
  expect_invalid(fn() raise AugError {
    ignore(augment(input, Array::make(257, skipped), 0, ""))
  })
  let record = augment(input, [], 0, "").replay
  assert_eq(
    replay(input, { ..record, operations: Array::make(256, HFlip), }),
    input,
  )
  let oversized = { ..record, operations: Array::make(257, HFlip), }
  expect_invalid(fn() raise AugError { ignore(replay(input, oversized)) })
  expect_invalid(fn() raise AugError {
    ignore(replay_from_json(replay_to_json(oversized)))
  })
  let request = Json::object({
    "action": "augment".to_json(),
    "sample": sample_to_json(input),
    "pipeline": Json::array(
      Array::make(257, Json::object({ "op": "hflip".to_json() })),
    ),
  })
  let response = @json.parse(process_json(request.stringify()))
  assert_eq(field(response, "ok"), false.to_json())
  assert_eq(
    string(field(response, "error")),
    "pipelines and replay records are limited to 256 operations",
  )
}

///|
test "box limit checks both decoded samples and JSON arrays" {
  let input = fixture()
  validate({ ..input, boxes: Array::make(100000, input.boxes[0]), })
  let oversized = { ..input, boxes: Array::make(100001, input.boxes[0]), }
  expect_invalid(fn() raise AugError { validate(oversized) })
  expect_invalid(fn() raise AugError {
    ignore(sample_from_json(sample_to_json(oversized)))
  })
}

///|
test "JSON character limit rejects a real oversized string before parsing" {
  let oversized = " ".repeat(16777217)
  let result = @json.parse(process_json(oversized))
  assert_eq(field(result, "ok"), false.to_json())
  assert_eq(
    string(field(result, "error")),
    "JSON input is limited to 16777216 UTF-16 code units",
  )
}

///|
test "empty pipeline does not alias caller buffers" {
  let input = fixture()
  let output = augment(input, [], 0, "").sample
  output.image.data[0] = b'\xFF'
  if output.mask is Some(mask) {
    mask[0] = 99
  }
  assert_eq(input.image.data[0], b'\x0A')
  assert_eq(input.mask, Some([1, 2, 3, 4, 5, 6]))
}

///|
test "invalid samples operations probabilities and replay shapes raise catchable errors" {
  let input = fixture()
  expect_invalid(fn() raise AugError {
    validate({ ..input, image: { ..input.image, width: 0, }, })
  })
  expect_invalid(fn() raise AugError {
    validate({ ..input, image: { ..input.image, width: 2147483647, }, })
  })
  expect_invalid(fn() raise AugError {
    validate({ ..input, image: { ..input.image, channels: 2, }, })
  })
  expect_invalid(fn() raise AugError { validate({ ..input, mask: Some([1]), }) })
  expect_invalid(fn() raise AugError {
    validate({ ..input, mask: Some([1, 2, 3, 4, 5, -1]), })
  })
  expect_invalid(fn() raise AugError {
    validate({ ..input, boxes: [{ ..input.boxes[0], x_max: 4.0, }], })
  })
  expect_invalid(fn() raise AugError {
    ignore(run_transform(input, Crop(2147483647, 0, 1, 1)))
  })
  expect_invalid(fn() raise AugError {
    ignore(run_transform(input, RandomCrop(4, 1)))
  })
  expect_invalid(fn() raise AugError {
    ignore(run_transform(input, Brightness(-1.0)))
  })
  expect_invalid(fn() raise AugError {
    ignore(run_transform(input, Contrast(0.0 / 0.0)))
  })
  expect_invalid(fn() raise AugError {
    validate({ ..input, boxes: [{ ..input.boxes[0], x_min: 0.0 / 0.0, }], })
  })
  expect_invalid(fn() raise AugError {
    ignore(augment(input, [{ transform: HFlip, probability: 1.1, }], 0, ""))
  })
  expect_invalid(fn() raise AugError {
    ignore(augment(input, [], 0, "", min_visibility=1.1))
  })
  let record = augment(input, [], 0, "").replay
  expect_invalid(fn() raise AugError {
    ignore(replay(input, { ..record, version: 2, }))
  })
  expect_invalid(fn() raise AugError {
    ignore(replay(input, { ..record, input_width: 2, }))
  })
  expect_invalid(fn() raise AugError {
    ignore(replay(input, { ..record, input_channels: 3, }))
  })
  expect_invalid(fn() raise AugError {
    ignore(replay(input, { ..record, operations: [RandomCrop(1, 1)], }))
  })
}

///|
test "JSON bridge catches malformed input and replays complete annotated sample" {
  let request =
    #|{"action":"augment","sample":{"image":{"width":2,"height":1,"channels":1,"data":[10,20]},"mask":[1,2],"boxes":[{"x_min":0,"y_min":0,"x_max":1,"y_max":1,"label":"cat"}]},"pipeline":[{"op":"hflip"}],"seed":42,"sample_id":"demo"}
  let response = @json.parse(process_json(request))
  assert_eq(field(response, "ok"), true.to_json())
  let sample = sample_from_json(field(response, "sample"))
  assert_eq(sample.mask, Some([2, 1]))
  assert_eq(sample.boxes[0].label, "cat")
  let replay_request = Json::object({
    "action": "replay".to_json(),
    "sample": field(@json.parse(request), "sample"),
    "replay": field(response, "replay"),
  })
  assert_eq(
    field(@json.parse(process_json(replay_request.stringify())), "sample"),
    field(response, "sample"),
  )
  for
    malformed in [
      "{",
      "null",
      "{}",
      "{\"action\":\"unknown\"}",
      request.replace(old="[10,20]", new="[10.5,20]"),
      request.replace(old="[10,20]", new="[256,20]"),
    ] {
    let rejected = @json.parse(process_json(malformed))
    assert_eq(field(rejected, "ok"), false.to_json())
    assert_true(string(field(rejected, "error")).length() > 0)
  }
}
