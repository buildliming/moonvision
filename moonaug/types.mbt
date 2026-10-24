///|
/// A tightly packed, row-major byte image. Supported channels: gray (1), RGB (3), RGBA (4).
/// Buffers are owned by the caller; augmentation never modifies its input buffers.
pub(all) struct Image {
  width : Int
  height : Int
  channels : Int
  data : Array[Byte]
} derive(Debug, Eq)

///|
/// An axis-aligned bounding box in continuous half-open pixel coordinates.
/// Labels remain attached when boxes are transformed or filtered.
pub(all) struct BBox {
  x_min : Double
  y_min : Double
  x_max : Double
  y_max : Double
  label : String
} derive(Debug, Eq, ToJson)

///|
/// An image with optional row-major integer class mask and labeled boxes.
/// Mask values are non-negative class IDs; mask dimensions equal the image dimensions.
pub(all) struct Sample {
  image : Image
  mask : Array[Int]?
  boxes : Array[BBox]
} derive(Debug, Eq)

///|
/// Supported transforms. Sizes are (width, height); Crop is (x, y, width, height).
/// Rotate90 counts clockwise quarter turns. Brightness multiplies RGB/gray;
/// Contrast scales around 128. Neither color operation changes RGBA alpha.
/// Color factors must be finite and in [0, 16]. Images are limited to 16777216 pixels.
/// RandomCrop is only permitted in an augmentation pipeline, never in a replay.
pub(all) enum Transform {
  Resize(Int, Int)
  Crop(Int, Int, Int, Int)
  RandomCrop(Int, Int)
  HFlip
  VFlip
  Rotate90(Int)
  Brightness(Double)
  Contrast(Double)
} derive(Debug, Eq)

///|
/// Apply a transform with probability in [0, 1].
pub(all) struct Step {
  transform : Transform
  probability : Double
} derive(Debug, Eq)

///|
/// Version 1 replay: resolved operations, original shape, and box visibility policy.
/// Replay validates the exact original shape before executing any operation.
pub(all) struct Replay {
  version : Int
  input_width : Int
  input_height : Int
  input_channels : Int
  min_visibility : Double
  operations : Array[Transform]
} derive(Debug, Eq)

///|
/// Augmented sample and a deterministic, serializable record of the operations used.
pub(all) struct Augmented {
  sample : Sample
  replay : Replay
} derive(Debug, Eq)

///|
/// Invalid samples, parameters, and replay records raise a catchable error.
pub(all) suberror AugError {
  Invalid(String)
} derive(Debug, Eq)

///|
const MAX_PIXELS : Int = 16777216

///|
const MAX_OPERATIONS : Int = 256

///|
const MAX_BOXES : Int = 100000

///|
const MAX_JSON_CHARS : Int = 16777216

///|
fn operation_count(count : Int) -> Unit raise AugError {
  guard count <= MAX_OPERATIONS else {
    raise Invalid("pipelines and replay records are limited to 256 operations")
  }
}

///|
fn box_count(count : Int) -> Unit raise AugError {
  guard count <= MAX_BOXES else {
    raise Invalid("samples are limited to 100000 bounding boxes")
  }
}

///|
fn finite(value : Double) -> Bool {
  value == value &&
  value <= 1.7976931348623157e308 &&
  value >= -1.7976931348623157e308
}

///|
fn dimensions(width : Int, height : Int) -> Unit raise AugError {
  guard width > 0 && height > 0 else {
    raise Invalid("image dimensions must be positive")
  }
  guard width <= MAX_PIXELS && height <= MAX_PIXELS / width else {
    raise Invalid("images are limited to 16777216 pixels")
  }
}

///|
fn visibility_parameter(value : Double) -> Unit raise AugError {
  guard finite(value) && value >= 0.0 && value <= 1.0 else {
    raise Invalid("min_visibility must be finite and in [0, 1]")
  }
}

///|
/// Validate dimensions, buffer lengths, supported channels, masks, and box coordinates.
/// Boxes must be positive-area and entirely inside the input image.
/// Samples contain at most 100000 bounding boxes.
pub fn validate(sample : Sample) -> Unit raise AugError {
  box_count(sample.boxes.length())
  let image = sample.image
  dimensions(image.width, image.height)
  guard image.channels == 1 || image.channels == 3 || image.channels == 4 else {
    raise Invalid("image channels must be 1, 3, or 4")
  }
  guard image.data.length() == image.width * image.height * image.channels else {
    raise Invalid(
      "image buffer length does not match width * height * channels",
    )
  }
  if sample.mask is Some(mask) {
    guard mask.length() == image.width * image.height else {
      raise Invalid("mask buffer length does not match image dimensions")
    }
    for label in mask {
      guard label >= 0 else {
        raise Invalid("mask class IDs must be non-negative")
      }
    }
  }
  for box in sample.boxes {
    guard finite(box.x_min) &&
      finite(box.y_min) &&
      finite(box.x_max) &&
      finite(box.y_max) else {
      raise Invalid("bounding box coordinates must be finite")
    }
    guard box.x_min >= 0.0 &&
      box.y_min >= 0.0 &&
      box.x_max <= image.width.to_double() &&
      box.y_max <= image.height.to_double() &&
      box.x_min < box.x_max &&
      box.y_min < box.y_max else {
      raise Invalid(
        "bounding boxes must have positive area and lie inside the image",
      )
    }
  }
}

///|
fn validate_transform(transform : Transform) -> Unit raise AugError {
  match transform {
    Resize(w, h) | RandomCrop(w, h) => dimensions(w, h)
    Crop(x, y, w, h) => {
      dimensions(w, h)
      guard x >= 0 && y >= 0 else {
        raise Invalid("crop origin must be non-negative")
      }
    }
    Brightness(factor) | Contrast(factor) => {
      guard finite(factor) && factor >= 0.0 && factor <= 16.0 else {
        raise Invalid("color factor must be finite and in [0, 16]")
      }
    }
    HFlip | VFlip | Rotate90(_) => ()
  }
}

///|
fn copy_sample(sample : Sample) -> Sample {
  {
    image: { ..sample.image, data: sample.image.data.copy(), },
    mask: sample.mask.map(fn(mask) { mask.copy() }),
    boxes: sample.boxes.copy(),
  }
}
