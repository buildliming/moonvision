///|
fn json_field(value : Json, key : String) -> Json raise InteropError {
  guard value is Object(fields) else { raise Invalid("expected a JSON object") }
  guard fields.get(key) is Some(item) else {
    raise Invalid("missing field: " + key)
  }
  item
}

///|
fn json_optional(value : Json, key : String) -> Json? raise InteropError {
  guard value is Object(fields) else { raise Invalid("expected a JSON object") }
  fields.get(key)
}

///|
fn json_number(value : Json) -> Double raise InteropError {
  guard value is Number(number, ..) &&
    number == number &&
    number >= -1.7976931348623157e308 &&
    number <= 1.7976931348623157e308 else {
    raise Invalid("expected a finite number")
  }
  number
}

///|
fn json_integer(value : Json) -> Int raise InteropError {
  let number = json_number(value)
  guard number >= -2147483648.0 &&
    number <= 2147483647.0 &&
    @math.floor(number) == number else {
    raise Invalid("expected an int32 integer")
  }
  number.to_int()
}

///|
fn json_boolean(value : Json) -> Bool raise InteropError {
  match value {
    True => true
    False => false
    _ => raise Invalid("expected a boolean")
  }
}

///|
fn json_numbers(value : Json) -> Array[Double] raise InteropError {
  guard value is Array(items) && items.length() <= 4096 else {
    raise Invalid("expected a bounded numeric array")
  }
  let values = []
  for item in items {
    values.push(json_number(item))
  }
  values
}

///|
fn image_header(value : Json) -> Array[Int] raise InteropError {
  let width = json_integer(json_field(value, "width"))
  let height = json_integer(json_field(value, "height"))
  let channels = json_integer(json_field(value, "channels"))
  let shape = [1, channels, height, width]
  let size = compatible_size(shape)
  guard json_field(value, "data") is Array(data) && data.length() == size else {
    raise Invalid("image data length does not match shape")
  }
  shape
}

///|
fn decoded_sample(value : Json) -> @moonaug.Sample raise InteropError {
  ignore(image_header(json_field(value, "image")))
  @moonaug.sample_from_json(value) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
}

///|
fn decoded_points(
  request : Json,
) -> Array[@moonaug.Keypoint] raise InteropError {
  let entries = match json_optional(request, "keypoints") {
    None => return []
    Some(value) => {
      guard value is Array(items) && items.length() <= 100000 else {
        raise Invalid("keypoints must be an array of at most 100000 points")
      }
      items
    }
  }
  let points : Array[@moonaug.Keypoint] = []
  for point in entries {
    guard json_field(point, "label") is String(label) else {
      raise Invalid("point label must be a string")
    }
    let visible = json_boolean(json_field(point, "visible"))
    points.push({
      x: json_number(json_field(point, "x")),
      y: json_number(json_field(point, "y")),
      label,
      visible,
    })
  }
  points
}

///|
fn encode_points(points : Array[@moonaug.Keypoint]) -> Json {
  points
  .map(fn(point) {
    Json::object({
      "x": point.x.to_json(),
      "y": point.y.to_json(),
      "label": point.label.to_json(),
      "visible": point.visible.to_json(),
    })
  })
  .to_json()
}

///|
fn normalization_parameters(
  request : Json,
  channels : Int,
) -> (Array[Double], Array[Double])? raise InteropError {
  let mean = json_optional(request, "mean")
  let std = json_optional(request, "std")
  match (mean, std) {
    (None, None) => None
    (Some(m), Some(s)) => {
      let mean = json_numbers(m)
      let std = json_numbers(s)
      channel_parameters(channels, mean, std)
      Some((mean, std))
    }
    _ => raise Invalid("mean and std must be provided together")
  }
}

///|
fn process_request(request : Json) -> Json raise InteropError {
  guard json_field(request, "action") is String(action) else {
    raise Invalid("action must be a string")
  }
  if action == "warp_sample" {
    let sample = decoded_sample(json_field(request, "sample"))
    let keypoints = decoded_points(request)
    let height = json_integer(json_field(request, "out_height"))
    let width = json_integer(json_field(request, "out_width"))
    let raw_theta = json_field(request, "theta")
    guard raw_theta is Array(coefficients) && coefficients.length() == 6 else {
      raise Invalid("sample warp requires exactly six theta coefficients")
    }
    let theta = json_numbers(raw_theta)
    let align_corners = match json_optional(request, "align_corners") {
      None => false
      Some(value) => json_boolean(value)
    }
    let min_visibility = match json_optional(request, "min_visibility") {
      None => 0.0
      Some(value) => json_number(value)
    }
    let mask_fill = match json_optional(request, "mask_fill") {
      None => 0
      Some(value) => json_integer(value)
    }
    let output = warp_extended(
      { sample, keypoints, },
      theta,
      height,
      width,
      align_corners~,
      min_visibility~,
      mask_fill~,
    )
    Json::object({
      "ok": true.to_json(),
      "sample": @moonaug.sample_to_json(output.sample),
      "keypoints": encode_points(output.keypoints),
    })
  } else if action == "pack" {
    guard json_field(request, "images") is Array(entries) &&
      !entries.is_empty() &&
      entries.length() <= 64 else {
      raise Invalid("pack requires between 1 and 64 images")
    }
    let shape = image_header(entries[0])
    let mut total = 0
    for entry in entries {
      guard image_header(entry) == shape else {
        raise Invalid("batch images must have equal width, height and channels")
      }
      guard json_field(entry, "data") is Array(data) else {
        raise Invalid("image data must be an array")
      }
      guard data.length() <= 262144 - total else {
        raise Invalid("JSON packed tensors are limited to 262144 values")
      }
      total = total + data.length()
    }
    let parameters = normalization_parameters(request, shape[1])
    let images : Array[@moonaug.Image] = []
    // Inspect every shape and the aggregate JSON budget before copying pixels.
    for entry in entries {
      images.push(decoded_sample(Json::object({ "image": entry })).image)
    }
    let packed = images_to_tensor(images)
    let tensor = match parameters {
      None => packed
      Some((mean, std)) => normalize(packed, mean, std)
    }
    Json::object({
      "ok": true.to_json(),
      "tensor": @moonwarp.tensor_to_json(tensor),
    })
  } else if action == "unpack" {
    let raw_tensor = json_field(request, "tensor")
    let tensor = @moonwarp.tensor_from_json(raw_tensor) catch {
      error => raise convert_error(error)
    }
    let parameters = normalization_parameters(request, tensor.shape()[1])
    let restored = match parameters {
      None => tensor
      Some((mean, std)) => denormalize(tensor, mean, std)
    }
    let images = tensor_to_images(restored)
    let result = images.map(fn(image) {
      match @moonaug.sample_to_json({ image, mask: None, boxes: [], }) {
        Object(fields) => fields.get("image").unwrap()
        _ => Json::null()
      }
    })
    Json::object({ "ok": true.to_json(), "images": result.to_json() })
  } else {
    raise Invalid("action must be warp_sample, pack, or unpack")
  }
}

///|
/// JSON integration for annotated affine warps and normalized image batches.
/// Requests are bounded to 16M UTF-16 units; all failures return ok:false/error.
/// pack/unpack use a 262144-value JSON tensor budget; typed APIs allow 1048576.
pub fn process_json(request : String) -> String {
  let result = try {
    guard request.length() <= 16000000 else {
      raise Invalid("request exceeds 16000000 characters")
    }
    let value = @json.parse(request) catch {
      _ => raise Invalid("invalid JSON")
    }
    process_request(value)
  } catch {
    Invalid(message) =>
      Json::object({ "ok": false.to_json(), "error": message.to_json() })
  }
  let serialized = result.stringify()
  if serialized.length() > 16000000 {
    Json::object({
      "ok": false.to_json(),
      "error": "response exceeds 16000000 characters".to_json(),
    }).stringify()
  } else {
    serialized
  }
}
