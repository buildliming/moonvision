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
/// Constant border sizes and per-channel byte fill. Fill length must equal the image channels.
/// Mask padding uses the non-negative class ID mask_value.
pub(all) struct Padding {
  left : Int
  top : Int
  right : Int
  bottom : Int
  fill : Array[Byte]
  mask_value : Int
} derive(Debug, Eq)

///|
/// Fit inside a target rectangle while preserving aspect ratio, then pad centrally.
/// Odd padding puts the extra pixel on the right/bottom. scale_up=false never enlarges.
pub(all) struct LetterboxOptions {
  width : Int
  height : Int
  fill : Array[Byte]
  mask_value : Int
  scale_up : Bool
} derive(Debug, Eq)

///|
/// Sample a crop area fraction uniformly and aspect ratio log-uniformly, then resize.
/// Ten attempts are followed by a deterministic centered aspect-ratio fallback.
pub(all) struct ResizedCropOptions {
  width : Int
  height : Int
  scale_min : Double
  scale_max : Double
  ratio_min : Double
  ratio_max : Double
} derive(Debug, Eq)

///|
/// A labeled point in pixel-edge coordinates: pixel (x,y)'s center is (x+.5,y+.5).
/// Visible points lie in the closed image rectangle. Invisible points keep their index
/// and may lie outside it, but all coordinates must be finite. Visibility never recovers.
pub(all) struct Keypoint {
  x : Double
  y : Double
  label : String
  visible : Bool
} derive(Debug, Eq, ToJson)

///|
/// Add keypoints without changing the existing Sample initialization contract.
pub(all) struct ExtendedSample {
  sample : Sample
  keypoints : Array[Keypoint]
} derive(Debug, Eq)

///|
/// Annotated augmentation result with index-preserving keypoints.
pub(all) struct ExtendedAugmented {
  sample : ExtendedSample
  replay : Replay
} derive(Debug, Eq)

///|
/// One batch input with a stable non-empty dataset identity.
pub(all) struct BatchItem {
  sample_id : String
  input : ExtendedSample
} derive(Debug, Eq)

///|
/// Batch result retaining the input identity and corresponding replay.
pub(all) struct BatchResult {
  sample_id : String
  output : ExtendedAugmented
} derive(Debug, Eq)

///|
/// A replay keyed by stable identity; replay_batch matches by identity rather than order.
pub(all) struct BatchReplay {
  sample_id : String
  replay : Replay
} derive(Debug, Eq)

///|
/// Supported transforms. Sizes are (width, height); Crop is (x, y, width, height).
/// Rotate90 counts clockwise quarter turns. Brightness multiplies RGB/gray;
/// Contrast scales around 128. Neither color operation changes RGBA alpha.
/// Color factors must be finite and in [0, 16]. Images are limited to 16777216 pixels.
/// RandomCrop, Letterbox, and RandomResizedCrop are pipeline-only; replay stores primitives.
pub(all) enum Transform {
  Resize(Int, Int)
  Crop(Int, Int, Int, Int)
  RandomCrop(Int, Int)
  HFlip
  VFlip
  Rotate90(Int)
  Brightness(Double)
  Contrast(Double)
  Pad(Padding)
  Letterbox(LetterboxOptions)
  RandomResizedCrop(ResizedCropOptions)
} derive(Debug, Eq)

///|
/// Apply a transform with probability in [0, 1].
pub(all) struct Step {
  transform : Transform
  probability : Double
} derive(Debug, Eq)

///|
/// Resolved operations, original shape, and box visibility policy.
/// Version 1 supports the original primitives; version 2 additionally supports Pad.
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
const MAX_KEYPOINTS : Int = 100000

///|
const MAX_BATCH_ITEMS : Int = 64

///|
const MAX_JSON_VALUES : Int = 1048576

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
    Pad(padding) => validate_padding(padding)
    Letterbox(options) => {
      dimensions(options.width, options.height)
      validate_fill(options.fill, options.mask_value)
    }
    RandomResizedCrop(options) => {
      dimensions(options.width, options.height)
      guard finite(options.scale_min) &&
        finite(options.scale_max) &&
        options.scale_min > 0.0 &&
        options.scale_min <= options.scale_max &&
        options.scale_max <= 1.0 else {
        raise Invalid("crop scale range must satisfy 0 < min <= max <= 1")
      }
      guard finite(options.ratio_min) &&
        finite(options.ratio_max) &&
        options.ratio_min > 0.0 &&
        options.ratio_min <= options.ratio_max else {
        raise Invalid(
          "crop aspect ratios must be finite and satisfy 0 < min <= max",
        )
      }
    }
  }
}

///|
fn validate_fill(fill : Array[Byte], mask_value : Int) -> Unit raise AugError {
  guard fill.length() == 1 || fill.length() == 3 || fill.length() == 4 else {
    raise Invalid("padding fill must have 1, 3, or 4 channel values")
  }
  guard mask_value >= 0 else {
    raise Invalid("mask padding class ID must be non-negative")
  }
}

///|
fn validate_padding(padding : Padding) -> Unit raise AugError {
  guard padding.left >= 0 &&
    padding.top >= 0 &&
    padding.right >= 0 &&
    padding.bottom >= 0 else {
    raise Invalid("padding sizes must be non-negative")
  }
  validate_fill(padding.fill, padding.mask_value)
}

///|
fn padded_shape(
  width : Int,
  height : Int,
  padding : Padding,
) -> (Int, Int) raise AugError {
  validate_padding(padding)
  guard padding.left <= MAX_PIXELS - width &&
    padding.right <= MAX_PIXELS - width - padding.left &&
    padding.top <= MAX_PIXELS - height &&
    padding.bottom <= MAX_PIXELS - height - padding.top else {
    raise Invalid("padding exceeds the image dimension limit")
  }
  let out_width = width + padding.left + padding.right
  let out_height = height + padding.top + padding.bottom
  dimensions(out_width, out_height)
  (out_width, out_height)
}

///|
fn copy_sample(sample : Sample) -> Sample {
  {
    image: { ..sample.image, data: sample.image.data.copy(), },
    mask: sample.mask.map(fn(mask) { mask.copy() }),
    boxes: sample.boxes.copy(),
  }
}
