///|
test "legacy stochastic pipeline retains its resolved version 1 golden record" {
  let result = augment(
    fixture(),
    [
      { transform: HFlip, probability: 0.4, },
      { transform: RandomCrop(2, 1), probability: 1.0, },
      { transform: VFlip, probability: 0.0, },
      { transform: Brightness(1.2), probability: 0.7, },
    ],
    123,
    "sample",
  )
  assert_eq(result.replay.version, 1)
  assert_eq(result.replay.operations, [HFlip, Crop(1, 1, 2, 1)])
  assert_eq(result.sample.image.data, [50, 40])
  assert_eq(result.sample.mask, Some([5, 4]))
}

///|
test "tiny identity crop preserves boxes and all geometric output boxes remain valid" {
  let input : Sample = {
    image: { width: 1, height: 1, channels: 1, data: [1], },
    mask: None,
    boxes: [
      {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0e-200,
        y_max: 1.0e-200,
        label: "tiny",
      },
    ],
  }
  assert_eq(run_transform(input, Crop(0, 0, 1, 1)).boxes, input.boxes)
  let output = run_transform(input, HFlip)
  validate(output)
  assert_eq(output.boxes.length(), 0)
  let smallest = {
    ..input,
    boxes: [{ ..input.boxes[0], x_max: 5.0e-324, y_max: 1.0, }],
  }
  validate(run_transform(smallest, Resize(1, 1)))
}

///|
test "Pad snapshots caller fill and new mask fill is explicit" {
  let input = fixture()
  let fill : Array[Byte] = [99]
  let result = augment(
    input,
    [
      {
        transform: Pad({
          left: 1,
          top: 1,
          right: 0,
          bottom: 1,
          fill,
          mask_value: 42,
        }),
        probability: 1.0,
      },
    ],
    0,
    "pad",
  )
  assert_eq(result.sample.image.data, [
    99, 99, 99, 99, 99, 10, 20, 30, 99, 40, 50, 60, 99, 99, 99, 99,
  ])
  assert_eq(
    result.sample.mask,
    Some([42, 42, 42, 42, 42, 1, 2, 3, 42, 4, 5, 6, 42, 42, 42, 42]),
  )
  assert_eq(result.replay.version, 2)
  fill[0] = 0
  assert_eq(replay(input, result.replay), result.sample)
  assert_eq(replay_from_json(replay_to_json(result.replay)), result.replay)
}

///|
test "high-level operations resolve deterministic geometry and expanded count is bounded" {
  let input = fixture()
  let letterbox : Step = {
    transform: Letterbox({
      width: 8,
      height: 8,
      fill: [114],
      mask_value: 255,
      scale_up: true,
    }),
    probability: 1.0,
  }
  let result = augment(input, [letterbox], 0, "letterbox")
  assert_eq(result.replay.operations, [
    Resize(8, 5),
    Pad({ left: 0, top: 1, right: 0, bottom: 2, fill: [114], mask_value: 255, }),
  ])
  assert_eq(result.sample.boxes[0].y_max, 3.5)
  let small : Step = {
    transform: Letterbox({
      width: 3,
      height: 2,
      fill: [0],
      mask_value: 0,
      scale_up: false,
    }),
    probability: 1.0,
  }
  assert_eq(
    augment(input, Array::make(128, small), 0, "").replay.operations.length(),
    256,
  )
  expect_invalid(() => ignore(augment(input, Array::make(129, small), 0, "")))
  let source : Sample = {
    image: { width: 4, height: 2, channels: 1, data: [1, 2, 3, 4, 5, 6, 7, 8], },
    mask: None,
    boxes: [],
  }
  let options : ResizedCropOptions = {
    width: 2,
    height: 2,
    scale_min: 1.0,
    scale_max: 1.0,
    ratio_min: 1.0,
    ratio_max: 1.0,
  }
  let pipeline = [{ transform: RandomResizedCrop(options), probability: 1.0, }]
  let result = augment(source, pipeline, 11, "fallback")
  assert_eq(result.replay.operations, [Crop(1, 0, 2, 2), Resize(2, 2)])
  assert_eq(result.sample.image.data, [2, 3, 6, 7])
  assert_eq(replay(source, result.replay), result.sample)
  let extreme = { ..options, ratio_min: 1.0e-300, ratio_max: 1.0e300, }
  validate(
    augment(
      source,
      [{ transform: RandomResizedCrop(extreme), probability: 1.0, }],
      0,
      "extreme",
    ).sample,
  )
}

///|
test "replay preflight checks late invalid geometry before large allocations" {
  let sample = fixture()
  let record = augment(sample, [], 0, "").replay
  expect_invalid(() => {
    ignore(
      replay(sample, {
        ..record,
        operations: [Resize(4096, 4096), Crop(4096, 0, 1, 1)],
      }),
    )
  })
  let pad = Pad({
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    fill: [0],
    mask_value: 0,
  })
  expect_invalid(() => ignore(replay(sample, { ..record, operations: [pad], })))
  expect_invalid(() => {
    ignore(replay_from_json(replay_to_json({ ..record, operations: [pad], })))
  })
  for
    padding in [
      {
        left: 2147483647,
        top: 0,
        right: 1,
        bottom: 0,
        fill: [0],
        mask_value: 0,
      },
      { left: -1, top: 0, right: 0, bottom: 0, fill: [0], mask_value: 0, },
      { left: 0, top: 0, right: 0, bottom: 0, fill: [0, 0, 0], mask_value: 0, },
      { left: 0, top: 0, right: 0, bottom: 0, fill: [0], mask_value: -1, },
    ] {
    expect_invalid(() => ignore(run_transform(sample, Pad(padding))))
  }
  let options : ResizedCropOptions = {
    width: 2,
    height: 2,
    scale_min: 0.2,
    scale_max: 0.8,
    ratio_min: 0.5,
    ratio_max: 2.0,
  }
  for
    bad in [
      { ..options, scale_min: 0.0, },
      { ..options, scale_max: 1.1, },
      { ..options, scale_min: 0.9, },
      { ..options, ratio_min: 0.0, },
      { ..options, ratio_max: 0.1, },
      { ..options, ratio_max: 1.0 / 0.0, },
    ] {
    expect_invalid(() => {
      ignore(
        augment(
          sample,
          [{ transform: RandomResizedCrop(bad), probability: 0.0, }],
          0,
          "",
        ),
      )
    })
  }
}

///|
test "keypoints retain labels indices and irreversible visibility across crop pad flip" {
  let input : ExtendedSample = {
    sample: fixture(),
    keypoints: [
      { x: 0.5, y: 0.5, label: "cropped", visible: true, },
      { x: 1.5, y: 1.5, label: "inside", visible: true, },
      { x: 0.0, y: 0.0, label: "hidden", visible: false, },
    ],
  }
  let result = augment_with_keypoints(
    input,
    [
      { transform: Crop(1, 0, 2, 2), probability: 1.0, },
      {
        transform: Pad({
          left: 1,
          top: 0,
          right: 0,
          bottom: 0,
          fill: [0],
          mask_value: 0,
        }),
        probability: 1.0,
      },
      { transform: HFlip, probability: 1.0, },
    ],
    0,
    "points",
  )
  assert_eq(result.sample.keypoints, [
    { x: 2.5, y: 0.5, label: "cropped", visible: false, },
    { x: 1.5, y: 1.5, label: "inside", visible: true, },
    { x: 3.0, y: 0.0, label: "hidden", visible: false, },
  ])
  assert_eq(replay_with_keypoints(input, result.replay), result.sample)
  assert_eq(input.keypoints[0].visible, true)
  validate_extended(result.sample)
}

///|
test "edge-coordinate points resize rotate and reject nonfinite arithmetic" {
  let input : ExtendedSample = {
    sample: fixture(),
    keypoints: [
      { x: 0.0, y: 0.0, label: "origin", visible: true, },
      { x: 3.0, y: 2.0, label: "far-edge", visible: true, },
    ],
  }
  let result = augment_with_keypoints(
    input,
    [
      { transform: Rotate90(1), probability: 1.0, },
      { transform: Resize(4, 6), probability: 1.0, },
    ],
    0,
    "edges",
  )
  assert_eq(result.sample.keypoints, [
    { x: 4.0, y: 0.0, label: "origin", visible: true, },
    { x: 0.0, y: 6.0, label: "far-edge", visible: true, },
  ])
  validate_extended(result.sample)
  expect_invalid(() => {
    validate_extended({
      ..input,
      keypoints: [{ x: 4.0, y: 0.0, label: "bad", visible: true, }],
    })
  })
  expect_invalid(() => {
    validate_extended({
      ..input,
      keypoints: [{ x: 0.0 / 0.0, y: 0.0, label: "nan", visible: false, }],
    })
  })
  let extreme = {
    ..input,
    keypoints: [{ x: 1.0e308, y: 0.0, label: "extreme", visible: false, }],
  }
  expect_invalid(() => {
    ignore(
      augment_with_keypoints(
        extreme,
        [{ transform: Resize(30, 2), probability: 1.0, }],
        0,
        "",
      ),
    )
  })
}

///|
test "batch results remain stable across order partition and ID-based replay" {
  let base : ExtendedSample = {
    sample: fixture(),
    keypoints: [{ x: 0.5, y: 0.5, label: "p", visible: true, }],
  }
  let a : BatchItem = { sample_id: "a", input: base, }
  let b : BatchItem = { sample_id: "b", input: base, }
  let pipeline = [
    { transform: RandomCrop(2, 1), probability: 1.0, },
    { transform: HFlip, probability: 0.5, },
  ]
  let ab = augment_batch([a, b], pipeline, 17)
  let ba = augment_batch([b, a], pipeline, 17)
  assert_eq(ab[0], ba[1])
  assert_eq(ab[1], ba[0])
  assert_eq(augment_batch([a], pipeline, 17)[0], ab[0])
  let records = [
    { sample_id: "b", replay: ab[1].output.replay, },
    { sample_id: "a", replay: ab[0].output.replay, },
  ]
  assert_eq(replay_batch([a, b], records), ab)
  expect_invalid(() => ignore(augment_batch([a, a], pipeline, 0)))
  expect_invalid(() => {
    ignore(augment_batch([{ ..a, sample_id: "", }], pipeline, 0))
  })
  expect_invalid(() => {
    ignore(
      replay_batch([a, b], [
        { sample_id: "a", replay: records[0].replay, },
        { sample_id: "a", replay: records[1].replay, },
      ]),
    )
  })
  expect_invalid(() => ignore(replay_batch([a], [])))
  expect_invalid(() => {
    ignore(
      augment_batch(
        [a, b],
        [{ transform: Resize(4096, 4096), probability: 1.0, }],
        0,
      ),
    )
  })
  expect_invalid(() => ignore(augment_batch([], [], 0)))
}

///|
test "batch totals reject shared large inputs and excessive annotations" {
  let large : Sample = {
    image: {
      width: 4096,
      height: 65,
      channels: 1,
      data: Array::make(4096 * 65, b'\x00'),
    },
    mask: None,
    boxes: [],
  }
  let items = Array::makei(64, i => {
    sample_id: i.to_string(),
    input: { sample: large, keypoints: [], },
  })
  expect_invalid(() => ignore(augment_batch(items, [], 0)))
  let crowded : ExtendedSample = {
    sample: { ..fixture(), boxes: Array::make(50001, fixture().boxes[0]), },
    keypoints: [],
  }
  expect_invalid(() => {
    ignore(
      augment_batch(
        [
          { sample_id: "a", input: crowded, },
          { sample_id: "b", input: crowded, },
        ],
        [],
        0,
      ),
    )
  })
}

///|
test "extended JSON schemas roundtrip keypoints batch records and retain legacy shape" {
  let input = sample_to_json(fixture())
  let points = [{ x: 0.5, y: 0.5, label: "p", visible: true, }]
  let pipeline = Json::array([
    Json::object({
      "op": "pad".to_json(),
      "left": (1).to_json(),
      "top": (0).to_json(),
      "right": (0).to_json(),
      "bottom": (0).to_json(),
      "fill": [7].to_json(),
      "mask_value": (255).to_json(),
    }),
  ])
  let request = Json::object({
    "action": "augment".to_json(),
    "sample": input,
    "pipeline": pipeline,
    "keypoints": points.to_json(),
  })
  let response = @json.parse(process_json(request.stringify()))
  assert_eq(field(response, "ok"), true.to_json())
  assert_eq(keypoints_from_json(Some(field(response, "keypoints")))[0].x, 1.5)
  let replay_request = Json::object({
    "action": "replay".to_json(),
    "sample": input,
    "keypoints": points.to_json(),
    "replay": field(response, "replay"),
  })
  assert_eq(
    field(@json.parse(process_json(replay_request.stringify())), "keypoints"),
    field(response, "keypoints"),
  )
  let old = Json::object({
    "action": "augment".to_json(),
    "sample": input,
    "pipeline": Json::array([]),
  })
  assert_eq(
    optional_field(@json.parse(process_json(old.stringify())), "keypoints"),
    None,
  )
  let batch = Json::object({
    "action": "augment_batch".to_json(),
    "items": Json::array([
      Json::object({
        "sample_id": "one".to_json(),
        "sample": input,
        "keypoints": points.to_json(),
      }),
      Json::object({ "sample_id": "two".to_json(), "sample": input }),
    ]),
    "pipeline": pipeline,
  })
  let response = @json.parse(process_json(batch.stringify()))
  assert_eq(field(response, "ok"), true.to_json())
  let records = []
  for item in array(field(response, "items")) {
    records.push(
      Json::object({
        "sample_id": field(item, "sample_id"),
        "replay": field(item, "replay"),
      }),
    )
  }
  let batch_replay = Json::object({
    "action": "replay_batch".to_json(),
    "items": field(batch, "items"),
    "records": Json::array(records),
  })
  assert_eq(
    field(@json.parse(process_json(batch_replay.stringify())), "items"),
    field(response, "items"),
  )
}

///|
test "JSON budgets reject decoded or enlarged images before output execution" {
  let input = sample_to_json(fixture())
  let request = Json::object({
    "action": "augment".to_json(),
    "sample": input,
    "pipeline": Json::array([
      Json::object({
        "op": "resize".to_json(),
        "width": (4096).to_json(),
        "height": (4096).to_json(),
      }),
    ]),
  })
  assert_eq(
    field(@json.parse(process_json(request.stringify())), "ok"),
    false.to_json(),
  )
  let over = Json::object({
    "image": Json::object({
      "width": (1025).to_json(),
      "height": (1024).to_json(),
      "channels": (1).to_json(),
      "data": Json::array([]),
    }),
  })
  expect_invalid(() => ignore(sample_from_json(over)))
  let batch = Json::object({
    "action": "augment_batch".to_json(),
    "items": Json::array([
      Json::object({ "sample_id": "a".to_json(), "sample": input }),
      Json::object({ "sample_id": "b".to_json(), "sample": input }),
    ]),
    "pipeline": Json::array([
      Json::object({
        "op": "resize".to_json(),
        "width": (1024).to_json(),
        "height": (1024).to_json(),
      }),
    ]),
  })
  assert_eq(
    field(@json.parse(process_json(batch.stringify())), "ok"),
    false.to_json(),
  )
}
