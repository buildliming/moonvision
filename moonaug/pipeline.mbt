///|
priv struct Rng {
  mut state : UInt
}

///|
fn Rng::next(self : Rng) -> UInt {
  let mut value = self.state
  value = value ^ (value << 13)
  value = value ^ (value >> 17)
  value = value ^ (value << 5)
  self.state = value
  value
}

///|
fn Rng::new(seed : Int, sample_id : String) -> Rng {
  let mut state = seed.reinterpret_as_uint() ^ 2166136261U
  for unit in sample_id {
    state = (state ^ unit.to_uint()) * 16777619U
  }
  if state == 0U {
    state = 2654435769U
  }
  { state, }
}

///|
fn Rng::draw(self : Rng) -> Double {
  self.next().to_double() / 4294967296.0
}

///|
fn Rng::bounded(self : Rng, size : Int) -> Int {
  (self.next() % size.reinterpret_as_uint()).reinterpret_as_int()
}

///|
/// Apply a probabilistic pipeline with deterministic randomness keyed by seed and sample ID.
/// The same sample, pipeline, seed, and sample ID yield identical results on all targets.
/// Boxes are filtered by retained area at each crop; the threshold is inclusive.
/// All step parameters are checked, including steps with probability zero.
/// A pipeline may contain at most 256 steps.
pub fn augment(
  sample : Sample,
  pipeline : Array[Step],
  seed : Int,
  sample_id : String,
  min_visibility? : Double = 0.0,
) -> Augmented raise AugError {
  operation_count(pipeline.length())
  validate(sample)
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
  let mut output = copy_sample(sample)
  for step in pipeline {
    let apply = if step.probability == 1.0 {
      true
    } else if step.probability == 0.0 {
      false
    } else {
      rng.draw() < step.probability
    }
    if apply {
      let operation = match step.transform {
        RandomCrop(width, height) => {
          guard width <= output.image.width && height <= output.image.height else {
            raise Invalid("random_crop must fit inside the current image")
          }
          let x = rng.bounded(output.image.width - width + 1)
          let y = rng.bounded(output.image.height - height + 1)
          Crop(x, y, width, height)
        }
        Rotate90(turns) => Rotate90((turns % 4 + 4) % 4)
        other => other
      }
      output = apply_transform(output, operation, min_visibility)
      operations.push(operation)
    }
  }
  {
    sample: output,
    replay: {
      version: 1,
      input_width: sample.image.width,
      input_height: sample.image.height,
      input_channels: sample.image.channels,
      min_visibility,
      operations,
    },
  }
}

///|
/// Reapply resolved operations after validating the replay version and original shape.
/// A replay may be used with a different sample having the same image dimensions/channels.
/// A replay may contain at most 256 resolved operations.
pub fn replay(sample : Sample, record : Replay) -> Sample raise AugError {
  operation_count(record.operations.length())
  validate(sample)
  guard record.version == 1 else { raise Invalid("unsupported replay version") }
  guard sample.image.width == record.input_width &&
    sample.image.height == record.input_height &&
    sample.image.channels == record.input_channels else {
    raise Invalid("replay input shape does not match sample")
  }
  visibility_parameter(record.min_visibility)
  let mut output = copy_sample(sample)
  for operation in record.operations {
    output = apply_transform(output, operation, record.min_visibility)
  }
  output
}
