///|
fn boolean(value : Json) -> Bool raise AugError {
  match value {
    True => true
    False => false
    _ => raise Invalid("expected JSON boolean")
  }
}

///|
fn fill_from_json(value : Json) -> Array[Byte] raise AugError {
  let entries = array(value)
  guard entries.length() == 1 || entries.length() == 3 || entries.length() == 4 else {
    raise Invalid("padding fill must have 1, 3, or 4 channel values")
  }
  let values = []
  for item in entries {
    let value = integer(item)
    guard value >= 0 && value <= 255 else {
      raise Invalid("padding byte values must be in [0, 255]")
    }
    values.push(value.to_byte())
  }
  values
}

///|
fn json_value_count(count : Int) -> Unit raise AugError {
  guard count <= MAX_JSON_VALUES else {
    raise Invalid(
      "JSON decoded/output images are limited to 1048576 total channel values",
    )
  }
}

///|
fn pipeline_from_json(request : Json) -> Array[Step] raise AugError {
  let entries = array(field(request, "pipeline"))
  operation_count(entries.length())
  let pipeline = []
  for entry in entries {
    pipeline.push({
      transform: transform_from_json(entry),
      probability: optional_number(entry, "p", 1.0),
    })
  }
  pipeline
}

///|
fn seed_from_json(request : Json) -> Int raise AugError {
  match optional_field(request, "seed") {
    Some(value) => integer(value)
    None => 0
  }
}

///|
fn keypoints_from_json(value : Json?) -> Array[Keypoint] raise AugError {
  guard value is Some(value) else { return [] }
  let entries = array(value)
  guard entries.length() <= MAX_KEYPOINTS else {
    raise Invalid("samples are limited to 100000 keypoints")
  }
  let points = []
  for point in entries {
    points.push({
      x: number(field(point, "x")),
      y: number(field(point, "y")),
      label: string(field(point, "label")),
      visible: boolean(field(point, "visible")),
    })
  }
  points
}

///|
fn json_output_budget(records : Array[Replay]) -> Unit raise AugError {
  let mut values = 0
  for record in records {
    let (width, height) = validate_record(record)
    let mut stage_width = record.input_width
    let mut stage_height = record.input_height
    for operation in record.operations {
      let (w, h) = resolved_shape(
        stage_width,
        stage_height,
        record.input_channels,
        operation,
        record.version,
      )
      json_value_count(w * h * record.input_channels)
      stage_width = w
      stage_height = h
    }
    let count = width * height * record.input_channels
    guard count <= MAX_JSON_VALUES - values else {
      raise Invalid(
        "JSON decoded/output images are limited to 1048576 total channel values",
      )
    }
    values = values + count
  }
}

///|
fn single_request(request : Json, action : String) -> Json raise AugError {
  let sample = sample_from_json(field(request, "sample"))
  let points = optional_field(request, "keypoints")
  let input = { sample, keypoints: keypoints_from_json(points), }
  validate_extended(input)
  let record = if action == "augment" {
    let sample_id = match optional_field(request, "sample_id") {
      Some(value) => string(value)
      None => ""
    }
    resolve_pipeline(
      sample.image,
      pipeline_from_json(request),
      seed_from_json(request),
      sample_id,
      optional_number(request, "min_visibility", 0.0),
    )
  } else {
    let record = replay_from_json(field(request, "replay"))
    check_replay_shape(sample, record)
    record
  }
  json_output_budget([record])
  let result = execute_extended(input, record)
  let response : Map[String, Json] = {
    "ok": true.to_json(),
    "sample": sample_to_json(result.sample.sample),
    "replay": replay_to_json(result.replay),
  }
  if points is Some(_) {
    response["keypoints"] = result.sample.keypoints.to_json()
  }
  Json::object(response)
}

// Count every batch image before creating any decoded byte arrays.

///|
fn json_batch_input_budget(entries : Array[Json]) -> Unit raise AugError {
  guard entries.length() > 0 && entries.length() <= MAX_BATCH_ITEMS else {
    raise Invalid("batches must contain 1 to 64 samples")
  }
  let mut values = 0
  for entry in entries {
    let image = field(field(entry, "sample"), "image")
    let width = integer(field(image, "width"))
    let height = integer(field(image, "height"))
    let channels = integer(field(image, "channels"))
    dimensions(width, height)
    guard channels == 1 || channels == 3 || channels == 4 else {
      raise Invalid("image channels must be 1, 3, or 4")
    }
    let count = width * height * channels
    guard count <= MAX_JSON_VALUES - values else {
      raise Invalid(
        "JSON decoded/output images are limited to 1048576 total channel values",
      )
    }
    values = values + count
  }
}

///|
fn batch_request(request : Json, action : String) -> Json raise AugError {
  let entries = array(field(request, "items"))
  json_batch_input_budget(entries)
  let items = []
  for entry in entries {
    items.push({
      sample_id: string(field(entry, "sample_id")),
      input: {
        sample: sample_from_json(field(entry, "sample")),
        keypoints: keypoints_from_json(optional_field(entry, "keypoints")),
      },
    })
  }
  let records = if action == "augment_batch" {
    plan_batch(
      items,
      pipeline_from_json(request),
      seed_from_json(request),
      optional_number(request, "min_visibility", 0.0),
    )
  } else {
    let entries = array(field(request, "records"))
    guard entries.length() <= MAX_BATCH_ITEMS else {
      raise Invalid("batches must contain 1 to 64 samples")
    }
    let records = []
    for entry in entries {
      records.push({
        sample_id: string(field(entry, "sample_id")),
        replay: replay_from_json(field(entry, "replay")),
      })
    }
    align_batch_replays(items, records)
  }
  json_output_budget(records)
  let results = execute_batch(items, records)
  let output = results.map(fn(result) {
    Json::object({
      "sample_id": result.sample_id.to_json(),
      "sample": sample_to_json(result.output.sample.sample),
      "keypoints": result.output.sample.keypoints.to_json(),
      "replay": replay_to_json(result.output.replay),
    })
  })
  Json::object({ "ok": true.to_json(), "items": Json::array(output) })
}

///|
fn process_request(request : Json) -> Json raise AugError {
  let action = string(field(request, "action"))
  match action {
    "augment" | "replay" => single_request(request, action)
    "augment_batch" | "replay_batch" => batch_request(request, action)
    _ =>
      raise Invalid(
        "action must be augment, replay, augment_batch, or replay_batch",
      )
  }
}
