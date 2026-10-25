///|
fn field(value : Json, key : String) -> Json raise AugError {
  guard value is Object(fields) else { raise Invalid("expected JSON object") }
  guard fields.get(key) is Some(item) else {
    raise Invalid("missing field: " + key)
  }
  item
}

///|
fn optional_field(value : Json, key : String) -> Json? raise AugError {
  guard value is Object(fields) else { raise Invalid("expected JSON object") }
  fields.get(key)
}

///|
fn number(value : Json) -> Double raise AugError {
  guard value is Number(n, ..) && finite(n) else {
    raise Invalid("expected finite JSON number")
  }
  n
}

///|
fn integer(value : Json) -> Int raise AugError {
  let n = number(value)
  guard n >= -2147483648.0 && n <= 2147483647.0 && @math.floor(n) == n else {
    raise Invalid("expected 32-bit JSON integer")
  }
  n.to_int()
}

///|
fn string(value : Json) -> String raise AugError {
  guard value is String(text) else { raise Invalid("expected JSON string") }
  text
}

///|
fn array(value : Json) -> Array[Json] raise AugError {
  guard value is Array(items) else { raise Invalid("expected JSON array") }
  items
}

///|
fn optional_number(
  value : Json,
  key : String,
  default : Double,
) -> Double raise AugError {
  match optional_field(value, key) {
    Some(item) => number(item)
    None => default
  }
}

///|
/// Parse a packed-byte sample, checking the 1048576-channel-value JSON budget
/// before allocating its byte buffer. Invalid samples raise AugError.
pub fn sample_from_json(value : Json) -> Sample raise AugError {
  let raw_image = field(value, "image")
  let width = integer(field(raw_image, "width"))
  let height = integer(field(raw_image, "height"))
  let channels = integer(field(raw_image, "channels"))
  dimensions(width, height)
  guard channels == 1 || channels == 3 || channels == 4 else {
    raise Invalid("image channels must be 1, 3, or 4")
  }
  json_value_count(width * height * channels)
  let raw_data = array(field(raw_image, "data"))
  guard raw_data.length() == width * height * channels else {
    raise Invalid(
      "image buffer length does not match width * height * channels",
    )
  }
  let data = []
  for item in raw_data {
    let value = integer(item)
    guard value >= 0 && value <= 255 else {
      raise Invalid("image byte values must be in [0, 255]")
    }
    data.push(value.to_byte())
  }
  let mask = match optional_field(value, "mask") {
    None | Some(Null) => None
    Some(raw_mask) => {
      let entries = array(raw_mask)
      guard entries.length() == width * height else {
        raise Invalid("mask buffer length does not match image dimensions")
      }
      let labels = []
      for entry in entries {
        labels.push(integer(entry))
      }
      Some(labels)
    }
  }
  let boxes = []
  if optional_field(value, "boxes") is Some(raw_boxes) {
    let entries = array(raw_boxes)
    box_count(entries.length())
    for box in entries {
      boxes.push({
        x_min: number(field(box, "x_min")),
        y_min: number(field(box, "y_min")),
        x_max: number(field(box, "x_max")),
        y_max: number(field(box, "y_max")),
        label: string(field(box, "label")),
      })
    }
  }
  let sample = { image: { width, height, channels, data, }, mask, boxes, }
  validate(sample)
  sample
}

///|
/// Serialize an already valid Sample. Validate untrusted typed input first.
/// This low-level serializer does not impose process_json's response budget.
pub fn sample_to_json(sample : Sample) -> Json {
  Json::object({
    "image": Json::object({
      "width": sample.image.width.to_json(),
      "height": sample.image.height.to_json(),
      "channels": sample.image.channels.to_json(),
      "data": sample.image.data.map(fn(value) { value.to_int() }).to_json(),
    }),
    "mask": match sample.mask {
      Some(mask) => mask.to_json()
      None => Json::null()
    },
    "boxes": sample.boxes.to_json(),
  })
}

///|
fn transform_from_json(value : Json) -> Transform raise AugError {
  let name = string(field(value, "op"))
  let transform = match name {
    "resize" =>
      Resize(integer(field(value, "width")), integer(field(value, "height")))
    "crop" =>
      Crop(
        integer(field(value, "x")),
        integer(field(value, "y")),
        integer(field(value, "width")),
        integer(field(value, "height")),
      )
    "random_crop" =>
      RandomCrop(
        integer(field(value, "width")),
        integer(field(value, "height")),
      )
    "hflip" => HFlip
    "vflip" => VFlip
    "rotate90" => Rotate90(integer(field(value, "turns")))
    "brightness" => Brightness(number(field(value, "factor")))
    "contrast" => Contrast(number(field(value, "factor")))
    "pad" =>
      Pad({
        left: integer(field(value, "left")),
        top: integer(field(value, "top")),
        right: integer(field(value, "right")),
        bottom: integer(field(value, "bottom")),
        fill: fill_from_json(field(value, "fill")),
        mask_value: integer(field(value, "mask_value")),
      })
    "letterbox" =>
      Letterbox({
        width: integer(field(value, "width")),
        height: integer(field(value, "height")),
        fill: fill_from_json(field(value, "fill")),
        mask_value: integer(field(value, "mask_value")),
        scale_up: match optional_field(value, "scale_up") {
          None => true
          Some(item) => boolean(item)
        },
      })
    "random_resized_crop" =>
      RandomResizedCrop({
        width: integer(field(value, "width")),
        height: integer(field(value, "height")),
        scale_min: number(field(value, "scale_min")),
        scale_max: number(field(value, "scale_max")),
        ratio_min: number(field(value, "ratio_min")),
        ratio_max: number(field(value, "ratio_max")),
      })
    _ => raise Invalid("unsupported transform: " + name)
  }
  validate_transform(transform)
  transform
}

///|
fn transform_to_json(transform : Transform) -> Json {
  match transform {
    Resize(w, h) =>
      Json::object({
        "op": "resize".to_json(),
        "width": w.to_json(),
        "height": h.to_json(),
      })
    Crop(x, y, w, h) =>
      Json::object({
        "op": "crop".to_json(),
        "x": x.to_json(),
        "y": y.to_json(),
        "width": w.to_json(),
        "height": h.to_json(),
      })
    RandomCrop(w, h) =>
      Json::object({
        "op": "random_crop".to_json(),
        "width": w.to_json(),
        "height": h.to_json(),
      })
    HFlip => Json::object({ "op": "hflip".to_json() })
    VFlip => Json::object({ "op": "vflip".to_json() })
    Rotate90(turns) =>
      Json::object({ "op": "rotate90".to_json(), "turns": turns.to_json() })
    Brightness(factor) =>
      Json::object({ "op": "brightness".to_json(), "factor": factor.to_json() })
    Contrast(factor) =>
      Json::object({ "op": "contrast".to_json(), "factor": factor.to_json() })
    Pad(padding) =>
      Json::object({
        "op": "pad".to_json(),
        "left": padding.left.to_json(),
        "top": padding.top.to_json(),
        "right": padding.right.to_json(),
        "bottom": padding.bottom.to_json(),
        "fill": padding.fill.map(fn(value) { value.to_int() }).to_json(),
        "mask_value": padding.mask_value.to_json(),
      })
    Letterbox(options) =>
      Json::object({
        "op": "letterbox".to_json(),
        "width": options.width.to_json(),
        "height": options.height.to_json(),
        "fill": options.fill.map(fn(value) { value.to_int() }).to_json(),
        "mask_value": options.mask_value.to_json(),
        "scale_up": options.scale_up.to_json(),
      })
    RandomResizedCrop(options) =>
      Json::object({
        "op": "random_resized_crop".to_json(),
        "width": options.width.to_json(),
        "height": options.height.to_json(),
        "scale_min": options.scale_min.to_json(),
        "scale_max": options.scale_max.to_json(),
        "ratio_min": options.ratio_min.to_json(),
        "ratio_max": options.ratio_max.to_json(),
      })
  }
}

///|
/// Serialize a valid resolved replay. Version 2 extends version 1 with Pad.
pub fn replay_to_json(record : Replay) -> Json {
  Json::object({
    "version": record.version.to_json(),
    "input_width": record.input_width.to_json(),
    "input_height": record.input_height.to_json(),
    "input_channels": record.input_channels.to_json(),
    "min_visibility": record.min_visibility.to_json(),
    "operations": Json::array(record.operations.map(transform_to_json)),
  })
}

///|
/// Parse and preflight a version 1 or 2 replay and all intermediate shapes.
pub fn replay_from_json(value : Json) -> Replay raise AugError {
  let version = integer(field(value, "version"))
  guard version == 1 || version == 2 else {
    raise Invalid("unsupported replay version")
  }
  let input_width = integer(field(value, "input_width"))
  let input_height = integer(field(value, "input_height"))
  let input_channels = integer(field(value, "input_channels"))
  dimensions(input_width, input_height)
  guard input_channels == 1 || input_channels == 3 || input_channels == 4 else {
    raise Invalid("replay input_channels must be 1, 3, or 4")
  }
  let min_visibility = number(field(value, "min_visibility"))
  visibility_parameter(min_visibility)
  let operations = []
  let raw_operations = array(field(value, "operations"))
  operation_count(raw_operations.length())
  for item in raw_operations {
    let operation = transform_from_json(item)
    operations.push(operation)
  }
  let record = {
    version,
    input_width,
    input_height,
    input_channels,
    min_visibility,
    operations,
  }
  ignore(validate_record(record))
  record
}

///|
/// Browser/CLI bridge for single-item and ID-keyed batch augmentation/replay.
/// Single-item responses preserve {ok:true,sample,replay}; optional keypoints
/// are returned when supplied. Batch success returns {ok:true,items:[...]}.
/// Errors always return {ok:false,error}.
/// Malformed JSON, invalid input, and unsupported operations are caught.
/// Requests are limited to 16777216 UTF-16 code units, checked before JSON parsing.
pub fn process_json(input : String) -> String {
  if input.length() > MAX_JSON_CHARS {
    return Json::object({
      "ok": false.to_json(),
      "error": "JSON input is limited to 16777216 UTF-16 code units".to_json(),
    }).stringify()
  }
  let result = try {
    let request = @json.parse(input)
    process_request(request)
  } catch {
    Invalid(message) =>
      Json::object({ "ok": false.to_json(), "error": message.to_json() })
    _ =>
      Json::object({
        "ok": false.to_json(),
        "error": "invalid JSON input".to_json(),
      })
  }
  result.stringify()
}
