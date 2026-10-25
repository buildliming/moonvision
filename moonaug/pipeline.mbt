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
/// High-level steps expand into at most 256 resolved operations in total.
/// The complete geometry plan is validated before any output image allocation.
pub fn augment(
  sample : Sample,
  pipeline : Array[Step],
  seed : Int,
  sample_id : String,
  min_visibility? : Double = 0.0,
) -> Augmented raise AugError {
  validate(sample)
  let record = resolve_pipeline(
    sample.image,
    pipeline,
    seed,
    sample_id,
    min_visibility,
  )
  { sample: execute_record(sample, record), replay: record, }
}

///|
/// Reapply resolved operations after validating the replay version and original shape.
/// A replay may be used with a different sample having the same image dimensions/channels.
/// A replay may contain at most 256 resolved operations.
/// Every intermediate shape is preflighted before executing the first operation.
pub fn replay(sample : Sample, record : Replay) -> Sample raise AugError {
  validate(sample)
  check_replay_shape(sample, record)
  execute_record(sample, record)
}
