///|
fn required(value : Json, key : String) -> Json raise WarpError {
  guard value is Object(fields) else {
    raise Invalid("request must be a JSON object")
  }
  match fields.get(key) {
    Some(field) => field
    None => raise Invalid("missing field: \{key}")
  }
}

///|
fn optional(value : Json, key : String) -> Json? {
  if value is Object(fields) {
    fields.get(key)
  } else {
    None
  }
}

///|
fn decode_int(value : Json) -> Int raise WarpError {
  guard value is Number(number, ..) else {
    raise Invalid("expected an integer")
  }
  if !finite(number) ||
    number < -2147483648.0 ||
    number > 2147483647.0 ||
    @double.floor(number) != number {
    raise Invalid("expected an exact signed 32-bit integer")
  }
  number.to_int()
}

///|
fn decode_shape(value : Json) -> Array[Int] raise WarpError {
  guard value is Array(items) else {
    raise Invalid("shape must be an integer array")
  }
  if items.length() != 4 {
    raise Invalid("tensor/grid shape must have exactly four dimensions")
  }
  let shape = []
  for item in items {
    shape.push(decode_int(item))
  }
  shape
}

///|
fn decode_double(value : Json) -> Double raise WarpError {
  let number : Double = @json.from_json(value) catch {
    _ => raise Invalid("expected a number")
  }
  if !finite(number) {
    raise Invalid("expected a finite number")
  }
  number
}

///|
fn decode_bool(value : Json) -> Bool raise WarpError {
  @json.from_json(value) catch {
    _ => raise Invalid("expected a boolean")
  }
}

///|
fn decode_numbers(value : Json) -> Array[Double] raise WarpError {
  let data : Array[Double] = @json.from_json(value) catch {
    _ => raise Invalid("expected an array of numbers")
  }
  validate_values(data)
  data
}

///|
fn decode_tensor(value : Json) -> Tensor raise WarpError {
  let shape = decode_shape(required(value, "shape"))
  let size = checked_size(shape, 4)
  json_budget(size)
  let raw_data = required(value, "data")
  preflight_array(raw_data, size, "tensor data")
  let data = decode_numbers(raw_data)
  { dims: shape, values: data, }
}

///|
fn json_budget(size : Int) -> Unit raise WarpError {
  if size > 262144 {
    raise Invalid("JSON tensors and grids are limited to 262144 values each")
  }
}

///|
fn preflight_array(
  value : Json,
  expected : Int,
  name : String,
) -> Unit raise WarpError {
  guard value is Array(items) else { raise Invalid("\{name} must be an array") }
  if items.length() != expected {
    raise Invalid("\{name} length does not match its shape")
  }
}

///|
fn decode_grid(value : Json) -> Grid raise WarpError {
  let shape = decode_shape(required(value, "shape"))
  let size = checked_size(shape, 4)
  if shape[3] != 2 {
    raise Invalid("grid last dimension must be 2")
  }
  json_budget(size)
  let raw_data = required(value, "data")
  preflight_array(raw_data, size, "grid data")
  { dims: shape, values: decode_numbers(raw_data), }
}

///|
fn decode_theta(value : Json, batch : Int) -> Array[Double] raise WarpError {
  preflight_array(value, batch * 6, "theta")
  decode_numbers(value)
}

///|
/// Decode a validated NCHW tensor from a {shape,data} JSON object. Rank, shape
/// size, exact integers, and raw array length are checked before numeric copies.
/// JSON tensors are limited to 262144 values; typed Tensor::new supports more.
pub fn tensor_from_json(value : Json) -> Tensor raise WarpError {
  decode_tensor(value)
}

///|
/// Encode a validated tensor as a {shape,data} JSON object. New arrays are
/// created; mutating this JSON cannot mutate the opaque input tensor.
pub fn tensor_to_json(tensor : Tensor) -> Json {
  encode_tensor(tensor)
}

///|
fn encode_tensor(tensor : Tensor) -> Json {
  Json::object({
    "shape": tensor.dims.to_json(),
    "data": tensor.values.to_json(),
  })
}

///|
fn encode_grid(grid : Grid) -> Json {
  Json::object({ "shape": grid.dims.to_json(), "data": grid.values.to_json() })
}

///|
fn request_align(request : Json) -> Bool raise WarpError {
  match optional(request, "align_corners") {
    Some(value) => decode_bool(value)
    None => false
  }
}

///|
fn request_mode(request : Json) -> SamplingMode raise WarpError {
  match optional(request, "mode") {
    None => Bilinear
    Some(String("bilinear")) => Bilinear
    Some(String("nearest")) => Nearest
    _ => raise Invalid("mode must be bilinear or nearest")
  }
}

///|
fn request_padding(
  request : Json,
  fallback : PaddingMode,
) -> PaddingMode raise WarpError {
  match optional(request, "padding") {
    None => fallback
    Some(String("zeros")) => Zeros
    Some(String("border")) => Border
    Some(String("reflection")) => Reflection
    _ => raise Invalid("padding must be zeros, border, or reflection")
  }
}

///|
fn json_output_shape(
  input : Tensor,
  height : Int,
  width : Int,
) -> Unit raise WarpError {
  json_budget(checked_size([input.dims[0], input.dims[1], height, width], 4))
  json_budget(checked_size([input.dims[0], height, width, 2], 4))
}

///|
fn process_request(request : Json) -> Json raise WarpError {
  let action : String = @json.from_json(required(request, "action")) catch {
    _ => raise Invalid("action must be a string")
  }
  let input = decode_tensor(required(request, "input"))
  let align_corners = request_align(request)
  if action == "sample" {
    let mode = request_mode(request)
    let padding = request_padding(request, Zeros)
    let grid = match optional(request, "grid") {
      Some(value) => {
        let grid = decode_grid(value)
        json_output_shape(input, grid.dims[1], grid.dims[2])
        grid
      }
      None => {
        let height = decode_int(required(request, "out_height"))
        let width = decode_int(required(request, "out_width"))
        json_output_shape(input, height, width)
        let theta = decode_theta(required(request, "theta"), input.dims[0])
        affine_grid(theta, input.dims[0], height, width, align_corners~)
      }
    }
    let output = grid_sample(input, grid, align_corners~, mode~, padding~)
    Json::object({
      "ok": Json::boolean(true),
      "output": encode_tensor(output),
      "grid": encode_grid(grid),
    })
  } else if action == "register" {
    let padding = request_padding(request, Zeros)
    let target = decode_tensor(required(request, "target"))
    let iterations = match optional(request, "iterations") {
      Some(v) => decode_int(v)
      None => 120
    }
    let learning_rate = match optional(request, "learning_rate") {
      Some(v) => decode_double(v)
      None => 0.5
    }
    let initial_theta = match optional(request, "theta") {
      Some(v) => Some(decode_theta(v, input.dims[0]))
      None => None
    }
    let result = register_affine(
      input,
      target,
      iterations~,
      learning_rate~,
      initial_theta?,
      align_corners~,
      padding~,
    )
    Json::object({
      "ok": Json::boolean(true),
      "output": encode_tensor(result.output),
      "theta": result.theta.to_json(),
      "losses": result.losses.to_json(),
      "initial_loss": result.losses[0].to_json(),
      "final_loss": result.losses[result.losses.length() - 1].to_json(),
    })
  } else if action == "register_multiscale" {
    let padding = request_padding(request, Border)
    let target = decode_tensor(required(request, "target"))
    let levels = match optional(request, "levels") {
      Some(v) => decode_int(v)
      None => 3
    }
    let iterations_per_level = match optional(request, "iterations_per_level") {
      Some(v) => decode_int(v)
      None => 80
    }
    let learning_rate = match optional(request, "learning_rate") {
      Some(v) => decode_double(v)
      None => 0.5
    }
    let initial_theta = match optional(request, "theta") {
      Some(v) => Some(decode_theta(v, input.dims[0]))
      None => None
    }
    let result = register_affine_multiscale(
      input,
      target,
      levels~,
      iterations_per_level~,
      learning_rate~,
      initial_theta?,
      align_corners~,
      padding~,
    )
    let traces = result.levels.map(level => {
      Json::object({
        "height": level.height.to_json(),
        "width": level.width.to_json(),
        "theta": level.theta.to_json(),
        "losses": level.losses.to_json(),
      })
    })
    Json::object({
      "ok": Json::boolean(true),
      "output": encode_tensor(result.output),
      "theta": result.theta.to_json(),
      "levels": Json::array(traces),
      "initial_loss": result.initial_loss.to_json(),
      "final_loss": result.final_loss.to_json(),
    })
  } else if action == "backward" {
    let mode = request_mode(request)
    let padding = request_padding(request, Zeros)
    let grad_output = decode_tensor(required(request, "grad_output"))
    match optional(request, "grid") {
      Some(raw) => {
        let grid = decode_grid(raw)
        json_output_shape(input, grid.dims[1], grid.dims[2])
        let gradient = grid_sample_backward(
          input,
          grid,
          grad_output,
          align_corners~,
          mode~,
          padding~,
        )
        Json::object({
          "ok": Json::boolean(true),
          "input_grad": encode_tensor(gradient.input_grad),
          "grid_grad": encode_grid(gradient.grid_grad),
        })
      }
      None => {
        json_output_shape(input, grad_output.dims[2], grad_output.dims[3])
        let theta = decode_theta(required(request, "theta"), input.dims[0])
        let gradient = warp_affine_backward(
          input,
          theta,
          grad_output,
          align_corners~,
          mode~,
          padding~,
        )
        Json::object({
          "ok": Json::boolean(true),
          "input_grad": encode_tensor(gradient.input_grad),
          "theta_grad": gradient.theta_grad.to_json(),
        })
      }
    }
  } else {
    raise Invalid(
      "unknown action: use sample, register, register_multiscale, or backward",
    )
  }
}

///|
/// JSON bridge for JS/Wasm demos and external integrations. Returns JSON with
/// `ok:true` or `ok:false,error`; invalid user inputs never escape as exceptions.
/// Supported actions are sample, backward, register, and register_multiscale.
pub fn process_json(request : String) -> String {
  let result = try {
    if request.length() > 16000000 {
      raise Invalid("request exceeds 16000000 characters")
    }
    let parsed = @json.parse(request) catch {
      _ => raise Invalid("invalid JSON")
    }
    process_request(parsed)
  } catch {
    Invalid(message) =>
      Json::object({
        "ok": Json::boolean(false),
        "error": Json::string(message),
      })
  }
  let serialized = result.stringify()
  if serialized.length() > 16000000 {
    Json::object({
      "ok": Json::boolean(false),
      "error": Json::string("response exceeds 16000000 characters"),
    }).stringify()
  } else {
    serialized
  }
}
