///|
/// Invalid image/tensor layouts, unsupported channel counts, or batch selections.
/// Conversion translates upstream validation failures into this catchable error.
pub(all) suberror InteropError {
  Invalid(String)
}

// Check the shared resource envelope before allocating or copying pixel values.

///|
fn compatible_size(shape : Array[Int]) -> Int raise InteropError {
  guard shape.length() == 4 else {
    raise Invalid("expected NCHW rank 4 tensor")
  }
  guard shape[1] == 1 || shape[1] == 3 || shape[1] == 4 else {
    raise Invalid("image conversion supports 1, 3, or 4 channels")
  }
  let mut size = 1
  for dim in shape {
    guard dim > 0 && dim <= 4096 else {
      raise Invalid("each tensor dimension must be between 1 and 4096")
    }
    guard size <= 1048576 / dim else {
      raise Invalid("image conversion exceeds the 1048576-value tensor limit")
    }
    size = size * dim
  }
  size
}

///|
/// Convert a packed gray/RGB/RGBA image into a batch-one NCHW Double tensor.
/// Bytes are normalized by 255, including alpha; no color-space conversion or
/// alpha premultiplication occurs. Input data is never modified or retained.
/// Validates MoonAug's image contract and MoonWarp's 4096-dimension/1048576-value
/// limits before allocating the tensor value buffer. Raises InteropError on failure.
pub fn image_to_tensor(
  image : @moonaug.Image,
) -> @moonwarp.Tensor raise InteropError {
  @moonaug.validate({ image, mask: None, boxes: [], }) catch {
    @moonaug.Invalid(message) => raise Invalid(message)
  }
  let shape = [1, image.channels, image.height, image.width]
  let size = compatible_size(shape)
  let values = Array::make(size, 0.0)
  let plane = image.width * image.height
  for pixel in 0..<plane {
    for channel in 0..<image.channels {
      values[channel * plane + pixel] = image.data[pixel * image.channels +
        channel]
        .to_int()
        .to_double() /
        255.0
    }
  }
  @moonwarp.Tensor::new(shape, values) catch {
    @moonwarp.Invalid(message) => raise Invalid(message)
  }
}

///|
/// Select one batch item from an NCHW tensor and return a packed byte image.
/// Supports channels 1/3/4. Values clamp to [0,1], multiply by 255, and round
/// half up. Alpha is handled as an ordinary channel. The tensor remains unchanged.
/// Validates the shared resource limits and batch index before copying values or
/// allocating the image buffer. Raises InteropError on failure.
pub fn tensor_to_image(
  tensor : @moonwarp.Tensor,
  batch_index? : Int = 0,
) -> @moonaug.Image raise InteropError {
  let shape = tensor.shape()
  ignore(compatible_size(shape))
  guard batch_index >= 0 && batch_index < shape[0] else {
    raise Invalid("batch_index is outside the tensor batch")
  }
  let channels = shape[1]
  let height = shape[2]
  let width = shape[3]
  let plane = width * height
  let values = tensor.data()
  let data = Array::make(plane * channels, b'\x00')
  let offset = batch_index * plane * channels
  for pixel in 0..<plane {
    for channel in 0..<channels {
      let value = values[offset + channel * plane + pixel].max(0.0).min(1.0)
      data[pixel * channels + channel] = @math.floor(value * 255.0 + 0.5)
        .to_int()
        .to_byte()
    }
  }
  { width, height, channels, data, }
}
