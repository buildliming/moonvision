///|
fn snapshot_transform(operation : Transform) -> Transform {
  match operation {
    Pad(padding) => Pad({ ..padding, fill: padding.fill.copy(), })
    Letterbox(options) => Letterbox({ ..options, fill: options.fill.copy(), })
    other => other
  }
}

///|
fn snapshot_replay(record : Replay) -> Replay {
  { ..record, operations: record.operations.map(snapshot_transform), }
}

///|
fn resolved_shape(
  width : Int,
  height : Int,
  channels : Int,
  operation : Transform,
  version : Int,
) -> (Int, Int) raise AugError {
  validate_transform(operation)
  match operation {
    Resize(w, h) => (w, h)
    Crop(x, y, w, h) => {
      guard w <= width && h <= height && x <= width - w && y <= height - h else {
        raise Invalid("crop must lie entirely inside the current image")
      }
      (w, h)
    }
    Rotate90(turns) =>
      if turns % 2 == 0 {
        (width, height)
      } else {
        (height, width)
      }
    Pad(padding) => {
      guard version == 2 else {
        raise Invalid("padding requires replay version 2")
      }
      guard padding.fill.length() == channels else {
        raise Invalid("padding fill length must equal image channels")
      }
      padded_shape(width, height, padding)
    }
    HFlip | VFlip | Brightness(_) | Contrast(_) => (width, height)
    RandomCrop(_, _) | Letterbox(_) | RandomResizedCrop(_) =>
      raise Invalid("replay cannot contain unresolved high-level transforms")
  }
}

// Validate every operation and every intermediate shape before executing any pixel work.

///|
fn validate_record(record : Replay) -> (Int, Int) raise AugError {
  operation_count(record.operations.length())
  guard record.version == 1 || record.version == 2 else {
    raise Invalid("unsupported replay version")
  }
  dimensions(record.input_width, record.input_height)
  guard record.input_channels == 1 ||
    record.input_channels == 3 ||
    record.input_channels == 4 else {
    raise Invalid("replay input_channels must be 1, 3, or 4")
  }
  visibility_parameter(record.min_visibility)
  let mut width = record.input_width
  let mut height = record.input_height
  for operation in record.operations {
    let (w, h) = resolved_shape(
      width,
      height,
      record.input_channels,
      operation,
      record.version,
    )
    width = w
    height = h
  }
  (width, height)
}

///|
fn append_resolved(
  operations : Array[Transform],
  operation : Transform,
  width : Int,
  height : Int,
  channels : Int,
) -> (Int, Int) raise AugError {
  operation_count(operations.length() + 1)
  let shape = resolved_shape(width, height, channels, operation, 2)
  operations.push(snapshot_transform(operation))
  shape
}

///|
fn letterbox_operations(
  width : Int,
  height : Int,
  options : LetterboxOptions,
) -> Array[Transform] {
  let mut scale = (options.width.to_double() / width.to_double()).min(
    options.height.to_double() / height.to_double(),
  )
  if !options.scale_up {
    scale = scale.min(1.0)
  }
  let w = @math.floor(width.to_double() * scale + 0.5)
    .to_int()
    .max(1)
    .min(options.width)
  let h = @math.floor(height.to_double() * scale + 0.5)
    .to_int()
    .max(1)
    .min(options.height)
  let left = (options.width - w) / 2
  let top = (options.height - h) / 2
  [
    Resize(w, h),
    Pad({
      left,
      top,
      right: options.width - w - left,
      bottom: options.height - h - top,
      fill: options.fill.copy(),
      mask_value: options.mask_value,
    }),
  ]
}

///|
fn resized_crop_operations(
  width : Int,
  height : Int,
  options : ResizedCropOptions,
  rng : Rng,
) -> Array[Transform] {
  let area = width.to_double() * height.to_double()
  let log_min = @math.ln(options.ratio_min)
  let log_max = @math.ln(options.ratio_max)
  for _ in 0..<10 {
    let target = area *
      (options.scale_min + rng.draw() * (options.scale_max - options.scale_min))
    let ratio = @math.exp(log_min + rng.draw() * (log_max - log_min))
    let rw = @math.floor((target * ratio).sqrt() + 0.5)
    let rh = @math.floor((target / ratio).sqrt() + 0.5)
    if finite(rw) &&
      finite(rh) &&
      rw >= 1.0 &&
      rh >= 1.0 &&
      rw <= width.to_double() &&
      rh <= height.to_double() {
      let w = rw.to_int()
      let h = rh.to_int()
      let x = rng.bounded(width - w + 1)
      let y = rng.bounded(height - h + 1)
      return [Crop(x, y, w, h), Resize(options.width, options.height)]
    }
  }
  let input_ratio = width.to_double() / height.to_double()
  let (w, h) = if input_ratio < options.ratio_min {
    (
      width,
      @math.floor(width.to_double() / options.ratio_min + 0.5)
      .to_int()
      .max(1)
      .min(height),
    )
  } else if input_ratio > options.ratio_max {
    (
      @math.floor(height.to_double() * options.ratio_max + 0.5)
      .to_int()
      .max(1)
      .min(width),
      height,
    )
  } else {
    (width, height)
  }
  [
    Crop((width - w) / 2, (height - h) / 2, w, h),
    Resize(options.width, options.height),
  ]
}

///|
fn resolve_pipeline(
  image : Image,
  pipeline : Array[Step],
  seed : Int,
  sample_id : String,
  min_visibility : Double,
) -> Replay raise AugError {
  operation_count(pipeline.length())
  visibility_parameter(min_visibility)
  for step in pipeline {
    guard finite(step.probability) &&
      step.probability >= 0.0 &&
      step.probability <= 1.0 else {
      raise Invalid("step probability must be finite and in [0, 1]")
    }
    validate_transform(step.transform)
  }
  let rng = Rng::new(seed, sample_id)
  let operations = []
  let mut width = image.width
  let mut height = image.height
  let mut version = 1
  for step in pipeline {
    let applied = if step.probability == 1.0 {
      true
    } else if step.probability == 0.0 {
      false
    } else {
      rng.draw() < step.probability
    }
    if applied {
      let resolved = match step.transform {
        RandomCrop(w, h) => {
          guard w <= width && h <= height else {
            raise Invalid("random_crop must fit inside the current image")
          }
          let x = rng.bounded(width - w + 1)
          let y = rng.bounded(height - h + 1)
          [Crop(x, y, w, h)]
        }
        Letterbox(options) => letterbox_operations(width, height, options)
        RandomResizedCrop(options) =>
          resized_crop_operations(width, height, options, rng)
        Rotate90(turns) => [Rotate90((turns % 4 + 4) % 4)]
        other => [other]
      }
      for operation in resolved {
        let (w, h) = append_resolved(
          operations,
          operation,
          width,
          height,
          image.channels,
        )
        if operation is Pad(_) {
          version = 2
        }
        width = w
        height = h
      }
    }
  }
  {
    version,
    input_width: image.width,
    input_height: image.height,
    input_channels: image.channels,
    min_visibility,
    operations,
  }
}

///|
fn check_replay_shape(sample : Sample, record : Replay) -> Unit raise AugError {
  ignore(validate_record(record))
  guard sample.image.width == record.input_width &&
    sample.image.height == record.input_height &&
    sample.image.channels == record.input_channels else {
    raise Invalid("replay input shape does not match sample")
  }
}

///|
fn execute_record(sample : Sample, record : Replay) -> Sample raise AugError {
  let mut output = copy_sample(sample)
  for operation in record.operations {
    output = apply_transform(output, operation, record.min_visibility)
  }
  output
}
