///|
fn convert_error(error : @moonwarp.WarpError) -> InteropError {
  match error {
    @moonwarp.Invalid(message) => Invalid(message)
  }
}

///|
/// Pack equal-sized gray/RGB/RGBA images into a normalized NCHW batch.
/// Images must have identical dimensions/channels; letterbox heterogeneous images
/// first. Validation and the aggregate 1048576-value limit precede allocation.
pub fn images_to_tensor(
  images : Array[@moonaug.Image],
) -> @moonwarp.Tensor raise InteropError {
  guard !images.is_empty() && images.length() <= 64 else {
    raise Invalid("image batches must contain between 1 and 64 images")
  }
  let first = images[0]
  let shape = [images.length(), first.channels, first.height, first.width]
  let size = compatible_size(shape)
  for image in images {
    @moonaug.validate({ image, mask: None, boxes: [], }) catch {
      @moonaug.Invalid(message) => raise Invalid(message)
    }
    guard image.width == first.width &&
      image.height == first.height &&
      image.channels == first.channels else {
      raise Invalid("batch images must have equal width, height and channels")
    }
  }
  let values = Array::make(size, 0.0)
  let plane = first.width * first.height
  for n in 0..<images.length() {
    for c in 0..<first.channels {
      for p in 0..<plane {
        values[(n * first.channels + c) * plane + p] = images[n].data[p *
          first.channels +
          c]
          .to_int()
          .to_double() /
          255.0
      }
    }
  }
  @moonwarp.Tensor::new(shape, values) catch {
    error => raise convert_error(error)
  }
}

///|
/// Unpack an NCHW batch into independently owned byte images. Values clamp to
/// [0,1] and round half up, as tensor_to_image. At most 64 images are returned.
pub fn tensor_to_images(
  tensor : @moonwarp.Tensor,
) -> Array[@moonaug.Image] raise InteropError {
  let shape = tensor.shape()
  ignore(compatible_size(shape))
  guard shape[0] <= 64 else {
    raise Invalid("image batches are limited to 64 images")
  }
  let values = tensor.data()
  let plane = shape[2] * shape[3]
  let images : Array[@moonaug.Image] = []
  for n in 0..<shape[0] {
    let data = Array::make(plane * shape[1], b'\x00')
    for p in 0..<plane {
      for c in 0..<shape[1] {
        let value = values[(n * shape[1] + c) * plane + p].max(0.0).min(1.0)
        data[p * shape[1] + c] = @math.floor(value * 255.0 + 0.5)
          .to_int()
          .to_byte()
      }
    }
    images.push({ width: shape[3], height: shape[2], channels: shape[1], data, })
  }
  images
}

///|
fn channel_parameters(
  channels : Int,
  mean : Array[Double],
  std : Array[Double],
) -> Unit raise InteropError {
  guard mean.length() == channels && std.length() == channels else {
    raise Invalid("mean and std must contain exactly one value per channel")
  }
  for c in 0..<channels {
    let m = mean[c]
    let s = std[c]
    guard m == m &&
      m >= -1.7976931348623157e308 &&
      m <= 1.7976931348623157e308 &&
      s == s &&
      s > 0.0 &&
      s <= 1.7976931348623157e308 else {
      raise Invalid(
        "means must be finite and standard deviations finite and positive",
      )
    }
  }
}

///|
fn normalize_values(
  tensor : @moonwarp.Tensor,
  mean : Array[Double],
  std : Array[Double],
  inverse : Bool,
) -> @moonwarp.Tensor raise InteropError {
  let shape = tensor.shape()
  channel_parameters(shape[1], mean, std)
  let values = tensor.data()
  let plane = shape[2] * shape[3]
  for n in 0..<shape[0] {
    for c in 0..<shape[1] {
      let offset = (n * shape[1] + c) * plane
      for p in 0..<plane {
        let value = values[offset + p]
        values[offset + p] = if inverse {
          let product = value * std[c]
          if product >= -1.7976931348623157e308 &&
            product <= 1.7976931348623157e308 {
            product + mean[c]
          } else {
            // The product may overflow while cancellation with a finite mean
            // leaves a representable sum. Halve before the product and sum.
            (value * (std[c] * 0.5) + mean[c] * 0.5) * 2.0
          }
        } else {
          let difference = value - mean[c]
          if difference >= -1.7976931348623157e308 &&
            difference <= 1.7976931348623157e308 {
            difference / std[c]
          } else {
            // Opposite signs can overflow the subtraction even when division
            // by a large standard deviation brings the result into range.
            value / std[c] - mean[c] / std[c]
          }
        }
      }
    }
  }
  @moonwarp.Tensor::new(shape, values) catch {
    error => raise convert_error(error)
  }
}

///|
/// Apply (value-mean[channel])/std[channel] to every NCHW batch item.
/// No clamping occurs. RGBA requires four explicit parameters, including alpha.
/// Rejects nonfinite results instead of silently generating invalid model input.
pub fn normalize(
  tensor : @moonwarp.Tensor,
  mean : Array[Double],
  std : Array[Double],
) -> @moonwarp.Tensor raise InteropError {
  normalize_values(tensor, mean, std, false)
}

///|
/// Reverse channel normalization with value*std[channel]+mean[channel].
/// A subsequent tensor_to_image performs explicit clamping and byte quantization.
pub fn denormalize(
  tensor : @moonwarp.Tensor,
  mean : Array[Double],
  std : Array[Double],
) -> @moonwarp.Tensor raise InteropError {
  normalize_values(tensor, mean, std, true)
}

///|
/// Convert an integer HW category mask to a batch-one, one-channel NCHW tensor.
/// Category IDs stay exact; they are never divided by 255. Use nearest sampling.
pub fn mask_to_tensor(
  mask : Array[Int],
  width : Int,
  height : Int,
) -> @moonwarp.Tensor raise InteropError {
  let shape = [1, 1, height, width]
  let size = compatible_size(shape)
  guard mask.length() == size else {
    raise Invalid("mask dimensions do not match its values")
  }
  for label in mask {
    guard label >= 0 else {
      raise Invalid("mask category IDs must be non-negative")
    }
  }
  let values = mask.map(fn(label) { label.to_double() })
  @moonwarp.Tensor::new(shape, values) catch {
    error => raise convert_error(error)
  }
}

///|
/// Recover one batch of integer category IDs without rounding fractional labels.
/// This rejects bilinearly mixed labels and nonintegral or out-of-int32 values.
pub fn tensor_to_mask(
  tensor : @moonwarp.Tensor,
  batch_index? : Int = 0,
) -> Array[Int] raise InteropError {
  let shape = tensor.shape()
  ignore(compatible_size(shape))
  guard shape[1] == 1 else {
    raise Invalid("mask tensors must have one channel")
  }
  guard batch_index >= 0 && batch_index < shape[0] else {
    raise Invalid("mask batch index is out of bounds")
  }
  let values = tensor.data()
  let plane = shape[2] * shape[3]
  for p in 0..<plane {
    let value = values[batch_index * plane + p]
    guard value >= 0.0 && value <= 2147483647.0 && @math.floor(value) == value else {
      raise Invalid(
        "mask tensor must contain exact non-negative int32 categories; use nearest sampling",
      )
    }
  }
  let result = Array::makei(plane, p => values[batch_index * plane + p].to_int())
  result
}
