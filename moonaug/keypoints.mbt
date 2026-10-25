///|
/// Validate the existing sample plus finite, labeled, index-preserving keypoints.
/// Visible points lie on or inside image edges; invisible coordinates may be outside.
pub fn validate_extended(input : ExtendedSample) -> Unit raise AugError {
  validate(input.sample)
  guard input.keypoints.length() <= MAX_KEYPOINTS else {
    raise Invalid("samples are limited to 100000 keypoints")
  }
  for point in input.keypoints {
    guard finite(point.x) && finite(point.y) else {
      raise Invalid("keypoint coordinates must be finite")
    }
    if point.visible {
      guard point.x >= 0.0 &&
        point.x <= input.sample.image.width.to_double() &&
        point.y >= 0.0 &&
        point.y <= input.sample.image.height.to_double() else {
        raise Invalid(
          "visible keypoints must lie inside the closed image rectangle",
        )
      }
    }
  }
}

///|
fn transformed_keypoints(
  points : Array[Keypoint],
  record : Replay,
) -> Array[Keypoint] raise AugError {
  let mut output = points.copy()
  let mut width = record.input_width
  let mut height = record.input_height
  for operation in record.operations {
    let (next_width, next_height) = resolved_shape(
      width,
      height,
      record.input_channels,
      operation,
      record.version,
    )
    let mapped = []
    for point in output {
      let (x, y) = match operation {
        Resize(w, h) =>
          (
            point.x * (w.to_double() / width.to_double()),
            point.y * (h.to_double() / height.to_double()),
          )
        Crop(x, y, _, _) => (point.x - x.to_double(), point.y - y.to_double())
        HFlip => (width.to_double() - point.x, point.y)
        VFlip => (point.x, height.to_double() - point.y)
        Rotate90(turns) =>
          match (turns % 4 + 4) % 4 {
            0 => (point.x, point.y)
            1 => (height.to_double() - point.y, point.x)
            2 => (width.to_double() - point.x, height.to_double() - point.y)
            _ => (point.y, width.to_double() - point.x)
          }
        Pad(padding) =>
          (
            point.x + padding.left.to_double(),
            point.y + padding.top.to_double(),
          )
        Brightness(_) | Contrast(_) => (point.x, point.y)
        RandomCrop(_, _) | Letterbox(_) | RandomResizedCrop(_) =>
          raise Invalid(
            "replay cannot contain unresolved high-level transforms",
          )
      }
      guard finite(x) && finite(y) else {
        raise Invalid("keypoint transform produced nonfinite coordinates")
      }
      // Clamp only resize rounding at the image edges; preserve genuinely outside coordinates.
      let (x, y) = if point.visible && operation is Resize(_, _) {
        (
          x.max(0.0).min(next_width.to_double()),
          y.max(0.0).min(next_height.to_double()),
        )
      } else {
        (x, y)
      }
      let visible = point.visible &&
        x >= 0.0 &&
        x <= next_width.to_double() &&
        y >= 0.0 &&
        y <= next_height.to_double()
      mapped.push({ x, y, label: point.label, visible, })
    }
    output = mapped
    width = next_width
    height = next_height
  }
  output
}

///|
fn execute_extended(
  input : ExtendedSample,
  record : Replay,
) -> ExtendedAugmented raise AugError {
  // Finish point arithmetic first, so malformed/extreme point geometry fails before image allocation.
  let keypoints = transformed_keypoints(input.keypoints, record)
  {
    sample: { sample: execute_record(input.sample, record), keypoints, },
    replay: record,
  }
}

///|
/// Apply one deterministic pipeline to the image, mask, boxes, and labeled keypoints.
/// Crop-out points remain in place with visible=false, which later operations never restore.
/// The replay records resolved geometry; the old Sample API remains unchanged.
pub fn augment_with_keypoints(
  input : ExtendedSample,
  pipeline : Array[Step],
  seed : Int,
  sample_id : String,
  min_visibility? : Double = 0.0,
) -> ExtendedAugmented raise AugError {
  validate_extended(input)
  let record = resolve_pipeline(
    input.sample.image,
    pipeline,
    seed,
    sample_id,
    min_visibility,
  )
  execute_extended(input, record)
}

///|
/// Replay the exact same geometry on keypoints as on image/mask/boxes.
/// All shapes and resolved operations are checked before any pixel work.
pub fn replay_with_keypoints(
  input : ExtendedSample,
  record : Replay,
) -> ExtendedSample raise AugError {
  validate_extended(input)
  check_replay_shape(input.sample, record)
  execute_extended(input, snapshot_replay(record)).sample
}
