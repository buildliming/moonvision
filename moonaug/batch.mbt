///|
fn validate_batch(items : Array[BatchItem]) -> Unit raise AugError {
  guard items.length() > 0 && items.length() <= MAX_BATCH_ITEMS else {
    raise Invalid("batches must contain 1 to 64 samples")
  }
  let ids : Map[String, Bool] = Map([])
  let mut pixels = 0
  let mut annotations = 0
  for item in items {
    guard item.sample_id.length() > 0 else {
      raise Invalid("batch sample IDs must be non-empty")
    }
    guard !ids.contains(item.sample_id) else {
      raise Invalid("batch sample IDs must be unique")
    }
    ids[item.sample_id] = true
    validate_extended(item.input)
    let count = item.input.sample.image.width * item.input.sample.image.height
    guard count <= MAX_PIXELS - pixels else {
      raise Invalid("batch input exceeds 16777216 total pixels")
    }
    pixels = pixels + count
    let count = item.input.sample.boxes.length() + item.input.keypoints.length()
    guard count <= MAX_BOXES - annotations else {
      raise Invalid("batch exceeds 100000 total annotations")
    }
    annotations = annotations + count
  }
}

///|
fn check_batch_output(records : Array[Replay]) -> Unit raise AugError {
  let mut pixels = 0
  for record in records {
    let (width, height) = validate_record(record)
    let count = width * height
    guard count <= MAX_PIXELS - pixels else {
      raise Invalid("batch output exceeds 16777216 total pixels")
    }
    pixels = pixels + count
  }
}

///|
fn plan_batch(
  items : Array[BatchItem],
  pipeline : Array[Step],
  seed : Int,
  min_visibility : Double,
) -> Array[Replay] raise AugError {
  validate_batch(items)
  let records = []
  for item in items {
    records.push(
      resolve_pipeline(
        item.input.sample.image,
        pipeline,
        seed,
        item.sample_id,
        min_visibility,
      ),
    )
  }
  check_batch_output(records)
  // Preflight keypoint arithmetic for the entire batch before allocating the first output image.
  for i in 0..<items.length() {
    ignore(transformed_keypoints(items[i].input.keypoints, records[i]))
  }
  records
}

///|
fn align_batch_replays(
  items : Array[BatchItem],
  records : Array[BatchReplay],
) -> Array[Replay] raise AugError {
  validate_batch(items)
  guard records.length() == items.length() else {
    raise Invalid("batch replay count must match sample count")
  }
  let by_id : Map[String, Replay] = Map([])
  for record in records {
    guard record.sample_id.length() > 0 && !by_id.contains(record.sample_id) else {
      raise Invalid("batch replay IDs must be non-empty and unique")
    }
    ignore(validate_record(record.replay))
    by_id[record.sample_id] = snapshot_replay(record.replay)
  }
  let aligned = []
  for item in items {
    guard by_id.get(item.sample_id) is Some(record) else {
      raise Invalid("missing replay for batch sample ID: " + item.sample_id)
    }
    check_replay_shape(item.input.sample, record)
    aligned.push(record)
  }
  check_batch_output(aligned)
  for i in 0..<items.length() {
    ignore(transformed_keypoints(items[i].input.keypoints, aligned[i]))
  }
  aligned
}

///|
fn execute_batch(
  items : Array[BatchItem],
  records : Array[Replay],
) -> Array[BatchResult] raise AugError {
  let results = []
  for i in 0..<items.length() {
    results.push({
      sample_id: items[i].sample_id,
      output: execute_extended(items[i].input, records[i]),
    })
  }
  results
}

///|
/// Augment 1–64 uniquely identified samples. Each ID gets its own seed-keyed stream,
/// so reordering or splitting a batch does not change an individual sample's result.
/// All samples/plans are checked before execution. Total input and output are each
/// limited to 16777216 pixels; boxes plus keypoints total at most 100000 per batch.
pub fn augment_batch(
  items : Array[BatchItem],
  pipeline : Array[Step],
  seed : Int,
  min_visibility? : Double = 0.0,
) -> Array[BatchResult] raise AugError {
  execute_batch(items, plan_batch(items, pipeline, seed, min_visibility))
}

///|
/// Replay a batch by stable ID, accepting records in any order. Missing/duplicate IDs,
/// incompatible shapes, and oversized batches fail before the first output image allocation.
pub fn replay_batch(
  items : Array[BatchItem],
  records : Array[BatchReplay],
) -> Array[BatchResult] raise AugError {
  execute_batch(items, align_batch_replays(items, records))
}
