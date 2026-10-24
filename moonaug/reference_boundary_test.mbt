///|
fn reference_aug_reject(request : String) -> Unit raise {
  assert_true(
    @json.parse(@moonaug.process_json(request))
    is Object({ "ok": False, "error": String(_), .. }),
  )
}

///|
fn reference_aug_typed_reject(
  action : () -> Unit raise @moonaug.AugError,
) -> Unit raise {
  let rejected = try {
    action()
    false
  } catch {
    @moonaug.Invalid(_) => true
  }
  assert_true(rejected)
}

///|
test "JSON mask IDs and byte fields are exact integers, never truncated" {
  let valid =
    #|{"action":"augment","sample":{"image":{"width":2,"height":1,"channels":1,"data":[10,20]},"mask":[1,2],"boxes":[]},"pipeline":[]}
  assert_true(
    @json.parse(@moonaug.process_json(valid)) is Object({ "ok": True, .. }),
  )
  for
    invalid in [
      valid.replace(old="\"mask\":[1,2]", new="\"mask\":[1.5,2]"),
      valid.replace(old="\"mask\":[1,2]", new="\"mask\":[-1,2]"),
      valid.replace(old="\"mask\":[1,2]", new="\"mask\":[2147483648,2]"),
      valid.replace(old="\"mask\":[1,2]", new="\"mask\":[true,2]"),
      valid.replace(old="\"width\":2", new="\"width\":2.9"),
      valid.replace(old="[10,20]", new="[10.1,20]"),
      valid.replace(
        old="\"pipeline\":[]",
        new="\"pipeline\":[{\"op\":\"brightness\",\"factor\":1e999}]",
      ),
      valid.replace(
        old="\"pipeline\":[]",
        new="\"pipeline\":[{\"op\":\"typo_hflip\"}]",
      ),
    ] {
    reference_aug_reject(invalid)
  }
}

///|
test "typed API rejects nonfinite parameters and protects shape arithmetic" {
  let sample : @moonaug.Sample = {
    image: { width: 2, height: 1, channels: 1, data: [10, 20], },
    mask: Some([1, 2]),
    boxes: [],
  }
  let infinity = 1.0 / 0.0
  let nan = 0.0 / 0.0
  for bad_factor in [infinity, -infinity, nan] {
    reference_aug_typed_reject(() => {
      ignore(
        @moonaug.augment(
          sample,
          [{ transform: @moonaug.Brightness(bad_factor), probability: 1.0, }],
          0,
          "nonfinite",
        ),
      )
    })
    reference_aug_typed_reject(() => {
      ignore(
        @moonaug.augment(
          sample,
          [{ transform: @moonaug.HFlip, probability: bad_factor, }],
          0,
          "nonfinite",
        ),
      )
    })
  }
  reference_aug_typed_reject(() => {
    @moonaug.validate({
      ..sample,
      image: { ..sample.image, width: 65536, height: 65536, },
    })
  })
  reference_aug_typed_reject(() => {
    ignore(
      @moonaug.augment(
        sample,
        [
          {
            transform: @moonaug.Resize(2147483647, 2147483647),
            probability: 1.0,
          },
        ],
        0,
        "huge",
      ),
    )
  })
}

///|
test "operation resource bound applies equally to pipelines and saved replays" {
  let sample : @moonaug.Sample = {
    image: { width: 1, height: 1, channels: 1, data: [17], },
    mask: Some([3]),
    boxes: [],
  }
  let allowed = Array::make(256, @moonaug.Step::{
    transform: @moonaug.Rotate90(0),
    probability: 1.0,
  })
  assert_eq(@moonaug.augment(sample, allowed, 0, "limit").sample, sample)
  reference_aug_typed_reject(() => {
    ignore(
      @moonaug.augment(
        sample,
        Array::make(257, @moonaug.Step::{
          transform: @moonaug.HFlip,
          probability: 0.0,
        }),
        0,
        "over-limit",
      ),
    )
  })
  let record = @moonaug.augment(sample, [], 0, "record").replay
  reference_aug_typed_reject(() => {
    ignore(
      @moonaug.replay(sample, {
        ..record,
        operations: Array::make(257, @moonaug.Rotate90(0)),
      }),
    )
  })
  reference_aug_typed_reject(() => {
    ignore(
      @moonaug.replay_from_json(
        @moonaug.replay_to_json({
          ..record,
          operations: Array::make(257, @moonaug.Rotate90(0)),
        }),
      ),
    )
  })
}

///|
test "replay JSON rejects unresolved randomness and future versions" {
  let request =
    #|{"action":"replay","sample":{"image":{"width":1,"height":1,"channels":1,"data":[17]},"mask":[3]},"replay":{"version":1,"input_width":1,"input_height":1,"input_channels":1,"min_visibility":0,"operations":[]}}
  assert_true(
    @json.parse(@moonaug.process_json(request)) is Object({ "ok": True, .. }),
  )
  for
    invalid in [
      request.replace(old="\"version\":1", new="\"version\":2"),
      request.replace(old="\"input_width\":1", new="\"input_width\":2"),
      request.replace(
        old="\"operations\":[]",
        new="\"operations\":[{\"op\":\"random_crop\",\"width\":1,\"height\":1}]",
      ),
      request.replace(
        old="\"operations\":[]",
        new="\"operations\":[{\"op\":\"unknown\"}]",
      ),
      request.replace(
        old="\"min_visibility\":0",
        new="\"min_visibility\":1e999",
      ),
    ] {
    reference_aug_reject(invalid)
  }
}
