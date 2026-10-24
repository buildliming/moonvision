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
  Tensor::new(shape, decode_numbers(required(value, "data")))
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
fn process_request(request : Json) -> Json raise WarpError {
  let action : String = @json.from_json(required(request, "action")) catch {
    _ => raise Invalid("action must be a string")
  }
  let input = decode_tensor(required(request, "input"))
  let align_corners = request_align(request)
  if action == "sample" {
    let grid = match optional(request, "grid") {
      Some(value) => {
        let shape = decode_shape(required(value, "shape"))
        Grid::new(shape, decode_numbers(required(value, "data")))
      }
      None => {
        let theta = decode_numbers(required(request, "theta"))
        let height = decode_int(required(request, "out_height"))
        let width = decode_int(required(request, "out_width"))
        affine_grid(theta, input.dims[0], height, width, align_corners~)
      }
    }
    let output = grid_sample(input, grid, align_corners~)
    Json::object({
      "ok": Json::boolean(true),
      "output": encode_tensor(output),
      "grid": encode_grid(grid),
    })
  } else if action == "register" {
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
      Some(v) => Some(decode_numbers(v))
      None => None
    }
    let result = register_affine(
      input,
      target,
      iterations~,
      learning_rate~,
      initial_theta?,
      align_corners~,
    )
    Json::object({
      "ok": Json::boolean(true),
      "output": encode_tensor(result.output),
      "theta": result.theta.to_json(),
      "losses": result.losses.to_json(),
      "initial_loss": result.losses[0].to_json(),
      "final_loss": result.losses[result.losses.length() - 1].to_json(),
    })
  } else if action == "backward" {
    let theta = decode_numbers(required(request, "theta"))
    let grad_output = decode_tensor(required(request, "grad_output"))
    let gradient = warp_affine_backward(
      input,
      theta,
      grad_output,
      align_corners~,
    )
    Json::object({
      "ok": Json::boolean(true),
      "input_grad": encode_tensor(gradient.input_grad),
      "theta_grad": gradient.theta_grad.to_json(),
    })
  } else {
    raise Invalid("unknown action: use sample, register, or backward")
  }
}

///|
/// JSON bridge for JS/Wasm demos and external integrations. Returns JSON with
/// `ok:true` or `ok:false,error`; invalid user inputs never escape as exceptions.
/// Supported actions are sample, backward (affine chain), and register.
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
  result.stringify()
}
