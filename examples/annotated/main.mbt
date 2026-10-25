///|
fn annotated_pipeline() -> Unit raise {
  let input : @aug.ExtendedSample = {
    sample: {
      image: {
        width: 3,
        height: 2,
        channels: 1,
        data: [10, 20, 30, 40, 50, 60],
      },
      mask: Some([0, 1, 1, 0, 2, 2]),
      boxes: [
        { x_min: 0.0, y_min: 0.0, x_max: 3.0, y_max: 2.0, label: "whole", },
      ],
    },
    keypoints: [{ x: 1.5, y: 1.0, label: "center", visible: true, }],
  }
  let pipeline : Array[@aug.Step] = [
    {
      transform: @aug.Letterbox({
        width: 4,
        height: 4,
        fill: [114],
        mask_value: 9,
        scale_up: true,
      }),
      probability: 1.0,
    },
  ]
  let augmented = @aug.augment_with_keypoints(
    input, pipeline, 42, "annotated-example",
  )
  assert_eq(augmented.replay.version, 2)
  let repeated = @aug.replay_with_keypoints(input, augmented.replay)
  assert_eq(repeated, augmented.sample)
  let mirrored = @interop.warp_extended(
    augmented.sample,
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    4,
    4,
    mask_fill=9,
  )
  @aug.validate_extended(mirrored)
  assert_eq(mirrored.keypoints[0].label, "center")
  assert_true(mirrored.keypoints[0].visible)
  let batch = @interop.images_to_tensor([
    mirrored.sample.image,
    mirrored.sample.image,
  ])
  let model_input = @interop.normalize(batch, [0.5], [0.25])
  let restored = @interop.tensor_to_images(
    @interop.denormalize(model_input, [0.5], [0.25]),
  )
  assert_eq(restored, [mirrored.sample.image, mirrored.sample.image])
  println("Letterbox + Replay v2 + synchronized affine annotations: 4x4")
  println("NCHW batch normalization round-trip: [2,1,4,4], byte-exact")
}

///|
fn main {
  annotated_pipeline() catch {
    error => abort("Annotated example failed: " + error.to_string())
  }
}
