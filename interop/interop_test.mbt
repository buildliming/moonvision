///|
fn expect_interop_error(
  action : () -> Unit raise @interop.InteropError,
) -> Unit raise {
  let rejected = try {
    action()
    false
  } catch {
    @interop.Invalid(_) => true
  }
  assert_true(rejected)
}

///|
test "HWC RGB image converts into normalized NCHW planes" {
  let image : @moonaug.Image = {
    width: 2,
    height: 2,
    channels: 3,
    data: [0, 1, 2, 30, 40, 50, 100, 110, 120, 253, 254, 255],
  }
  let tensor = @interop.image_to_tensor(image)
  assert_eq(tensor.shape(), [1, 3, 2, 2])
  let expected = [0, 30, 100, 253, 1, 40, 110, 254, 2, 50, 120, 255]
  let values = tensor.data()
  for i in 0..<expected.length() {
    assert_eq(values[i], expected[i].to_double() / 255.0)
  }
}

///|
test "gray RGB and RGBA round trip every byte value exactly" {
  for channels in [1, 3, 4] {
    let image : @moonaug.Image = {
      width: 256,
      height: 1,
      channels,
      data: Array::makei(256 * channels, i => (i % 256).to_byte()),
    }
    let original = image.data.copy()
    let tensor = @interop.image_to_tensor(image)
    let output = @interop.tensor_to_image(tensor)
    assert_eq(output, image)
    output.data[0] = 255
    assert_eq(image.data, original)
    assert_eq(tensor.data()[0], 0.0)
  }
}

///|
test "batch selection clamps out of range samples and rounds halves up" {
  let tensor = @moonwarp.Tensor::new([2, 1, 1, 2], [-1.0, 2.0, 0.5, 0.1])
  assert_eq(@interop.tensor_to_image(tensor).data, [0, 255])
  assert_eq(@interop.tensor_to_image(tensor, batch_index=1).data, [128, 26])
  assert_eq(tensor.data(), [-1.0, 2.0, 0.5, 0.1])
  expect_interop_error(() => {
    ignore(@interop.tensor_to_image(tensor, batch_index=-1))
  })
  expect_interop_error(() => {
    ignore(@interop.tensor_to_image(tensor, batch_index=2))
  })
}

///|
test "channel planes become packed RGBA with the selected batch" {
  let tensor = @moonwarp.Tensor::new([2, 4, 1, 2], [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.25, 0.0, 1.0,
  ])
  let image = @interop.tensor_to_image(tensor, batch_index=1)
  assert_eq(image.data, [255, 0, 128, 0, 0, 255, 64, 255])
}

///|
test "unsupported channels and malformed images raise InteropError" {
  let two_channel = @moonwarp.Tensor::new([1, 2, 1, 1], [0.0, 1.0])
  expect_interop_error(() => ignore(@interop.tensor_to_image(two_channel)))
  let invalid : @moonaug.Image = {
    width: 1,
    height: 1,
    channels: 2,
    data: [0, 0],
  }
  expect_interop_error(() => ignore(@interop.image_to_tensor(invalid)))
  expect_interop_error(() => {
    ignore(@interop.image_to_tensor({ ..invalid, channels: 3, }))
  })
  expect_interop_error(() => {
    ignore(@interop.image_to_tensor({ ..invalid, width: 0, }))
  })
}

///|
test "conversion checks resource and dimension bounds before allocating tensor values" {
  let oversized : @moonaug.Image = {
    width: 1025,
    height: 1024,
    channels: 1,
    data: Array::make(1025 * 1024, b'\x00'),
  }
  expect_interop_error(() => ignore(@interop.image_to_tensor(oversized)))
  let too_wide : @moonaug.Image = {
    width: 4097,
    height: 1,
    channels: 1,
    data: Array::make(4097, b'\x00'),
  }
  expect_interop_error(() => ignore(@interop.image_to_tensor(too_wide)))
  let boundary : @moonaug.Image = {
    width: 1024,
    height: 1024,
    channels: 1,
    data: Array::make(1048576, b'\x7F'),
  }
  let tensor = @interop.image_to_tensor(boundary)
  assert_eq(tensor.shape(), [1, 1, 1024, 1024])
  let output = @interop.tensor_to_image(tensor)
  assert_eq(output, boundary)
}

///|
test "typed augmentation tensor warp image flow composes without mutating input" {
  let image : @moonaug.Image = {
    width: 3,
    height: 2,
    channels: 1,
    data: [10, 20, 30, 40, 50, 60],
  }
  let augmented = @moonaug.augment(
    { image, mask: None, boxes: [], },
    [{ transform: @moonaug.HFlip, probability: 1.0, }],
    42,
    "pipeline",
  )
  let tensor = @interop.image_to_tensor(augmented.sample.image)
  let warped = @moonwarp.warp_affine(
    tensor,
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    2,
    3,
  )
  let restored = @interop.tensor_to_image(warped)
  assert_eq(restored, image)
  assert_eq(augmented.sample.image.data, [30, 20, 10, 60, 50, 40])
  assert_eq(image.data, [10, 20, 30, 40, 50, 60])
}
