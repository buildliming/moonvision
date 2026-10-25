///|
/// Run one typed pipeline connecting augmentation, tensor conversion and warping.
fn run_pipeline() -> Unit raise {
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
    "typed-example",
  )
  let tensor = @interop.image_to_tensor(augmented.sample.image)
  // Reflect output coordinates horizontally into the augmented input.
  let reflected = @moonwarp.warp_affine(
    tensor,
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    image.height,
    image.width,
  )
  let output = @interop.tensor_to_image(reflected)
  assert_eq(output, image)
  assert_eq(augmented.sample.image.data, [30, 20, 10, 60, 50, 40])
  println(
    "MoonAug -> NCHW tensor -> MoonWarp -> Image: 3x2 pixels restored exactly",
  )
}

///|
fn main {
  run_pipeline() catch {
    error => abort("Pipeline failed: " + error.to_string())
  }
}
