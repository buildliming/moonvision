# MoonAug

Annotation-aware, reproducible image augmentation in pure MoonBit. This package is part of `shunge/moonvision`; it has no third-party dependencies and no FFI.

## Quick start

Import `shunge/moonvision/moonaug` in your package configuration:

```moonbit
let sample : @moonaug.Sample = {
  image: {
    width: 2,
    height: 1,
    channels: 1,
    data: [b'\x0A', b'\x14'],
  },
  mask: Some([1, 2]),
  boxes: [{
    x_min: 0.0, y_min: 0.0,
    x_max: 1.0, y_max: 1.0,
    label: "object",
  }],
}
let result = @moonaug.augment(
  sample,
  [{ transform: HFlip, probability: 1.0 }],
  42,
  "image-001",
)
let identical = @moonaug.replay(sample, result.replay)
assert_eq(identical, result.sample)
```

The calling function must allow `raise AugError` or catch the error. No input-validation path calls `abort`.

## Data and coordinates

- `Image.data` is a tightly packed, row-major `Array[Byte]`; supported channels are gray (1), RGB (3), and RGBA (4). Dimensions must be positive, and pixel count is at most 16,777,216.
- `Sample.mask` is `None` or a flat row-major `Array[Int]` of non-negative class IDs matching the image size. A mask is a class map, not an RGB visualization. Nearest-neighbor geometry preserves source class IDs; explicit padding may introduce the caller's `mask_value` class.
- `BBox` uses continuous **half-open** `x_min, y_min, x_max, y_max`, with origin at the top left. Input boxes must have positive area and lie inside the image. `label` is preserved alongside each retained box.
- All operations return owned buffers and leave caller input untouched, including empty pipelines.
- Samples are limited to 100,000 boxes. Pipelines contain at most 256 steps, including skipped steps, and at most 256 resolved operations after expansion. These limits apply to both the typed API and JSON bridge. The complete geometry plan and every intermediate shape are checked before executing the first operation.
- Extremely small boxes can collapse under floating-point geometry. Such zero-width/height results are removed together with their labels; all retained output boxes satisfy `validate`. Visibility uses the product of retained width/height fractions to avoid area underflow for tiny valid boxes.

## Operations

| Transform | Parameters and behavior |
|---|---|
| `Resize(width, height)` | Image: half-pixel bilinear sampling with clamped borders and round-half-up bytes. Mask: exact-nearest `floor((out + 0.5) * in / out_size)`. Box corners scale continuously. |
| `Crop(x, y, width, height)` | Crop rectangle must fit inside the current image. All spatial targets receive the same crop. |
| `RandomCrop(width, height)` | Samples a valid integer crop origin; resolves to `Crop` in replay. |
| `HFlip`, `VFlip` | Exact image/mask permutation and continuous box transformation. |
| `Rotate90(turns)` | Clockwise quarter turns; negative and large values normalize modulo four. Odd turns swap width and height. |
| `Brightness(factor)` | Multiply gray/RGB values; clamp to 0–255 and round half up. |
| `Contrast(factor)` | `(value - 128) * factor + 128`; clamp and round half up. |
| `Pad(Padding)` | Constant left/top/right/bottom borders. Per-channel `fill` length must exactly equal image channels; `mask_value` is a non-negative class ID. Boxes translate by left/top. |
| `Letterbox(LetterboxOptions)` | Fit inside target width/height, optionally avoiding upscaling, then pad centrally. Resolves to `Resize` + `Pad`. |
| `RandomResizedCrop(ResizedCropOptions)` | Sample area fraction and aspect ratio, crop all annotations together, then resize to fixed width/height. Resolves to `Crop` + `Resize`. |

Color factors must be finite and in `[0, 16]`. RGBA alpha remains unchanged by color transforms. Spatial resizing interpolates all four RGBA channels independently in their stored byte space; it does not perform alpha premultiplication or linear-light conversion.

Mask resize evaluates the integer pixel-center numerator before dividing by the output size. Exact ties select the upper source index, including awkward size ratios such as 2→49; mask class boundaries remain independent of floating-point ratio rounding.

Every `Step` has an independent application probability in `[0, 1]`. Operation parameter ranges are validated even for zero-probability steps; shape-dependent fit checks apply when a step executes. A crop retains a box if its nonzero clipped area divided by its pre-crop area is at least `min_visibility`. This policy is applied at each crop, not relative to the original box before the entire pipeline.

`Padding` fields are `left`, `top`, `right`, `bottom`, `fill: Array[Byte]`, and `mask_value`. A single-value fill does **not** broadcast to RGB/RGBA. Replay snapshots the fill array; changing the original pipeline's fill later does not alter an already returned replay.

`LetterboxOptions` fields are `width`, `height`, `fill`, `mask_value`, and `scale_up`. The common scale is `min(target_width / width, target_height / height)`, capped at 1 when `scale_up=false`. Resized dimensions round half up, clamp to the target, and stay at least one pixel. Odd remaining padding puts the extra pixel on the right/bottom. Boxes/keypoints use the **actual rounded dimensions** for scaling, rather than the nominal common scale.

`ResizedCropOptions` fields are `width`, `height`, `scale_min`, `scale_max`, `ratio_min`, and `ratio_max`. Scales satisfy `0 < min <= max <= 1`; aspect ratios are positive, finite, and ordered. Each of ten attempts samples area uniformly and aspect ratio log-uniformly, rounds candidate dimensions half up, and accepts a crop that fits. A valid crop origin uses bounded modulo sampling with the pinned integer generator; this preserves existing seeded results and is not a guarantee of a strictly uniform distribution. If no attempt fits, a centered crop is chosen within the configured aspect-ratio range where possible, with dimensions at least one pixel, then resized. Applied probability is sampled before crop parameters; skipped steps do not consume parameter draws.

## Reproducibility and replay

`augment(sample, pipeline, seed, sample_id, min_visibility=0.0)` keys its random stream by the signed 32-bit seed and Unicode sample ID. No wall clock or global random state is used. The random stream is pinned by tests, so input ordering does not affect an individual sample's result. Different IDs or seeds can produce the same result; they do not guarantee distinct transformations.

`Replay` records the exact original width, height, channels, the visibility threshold, and resolved applied operations. Skipped steps are omitted. Version 1 retains all original primitives and remains readable; version 2 additionally permits `Pad`. Results containing padding use version 2; records containing only original primitives keep version 1. High-level `RandomCrop`, `Letterbox`, and `RandomResizedCrop` never appear in a valid replay. `replay` validates the original shape, every operation, and every intermediate shape before allocating the first output image. It may replay onto a different sample with the same shape.

Use `replay_to_json(record).stringify()` to persist a record and `replay_from_json(@json.parse(text))` to load it. Replay contains operation values and dimensions, not source pixels or input identity. Serialize your sample ID and dataset identity separately if the application needs that provenance.

## Keypoints without changing Sample

Existing `Sample { image, mask, boxes }` initialization remains valid. Add points using `ExtendedSample { sample, keypoints }`, then call `augment_with_keypoints` or `replay_with_keypoints`. `ExtendedAugmented` contains `sample: ExtendedSample` and `replay`. `validate_extended` checks the full input without copying image buffers.

Each `Keypoint` has finite `x`, `y`, a string `label`, and boolean `visible`. Coordinates use image **edges**: pixel `(x,y)` has center `(x+0.5,y+0.5)`. Visible points lie in the closed rectangle `[0,width] × [0,height]`. Consequently a horizontal flip maps `x` to `width-x`, resize multiplies coordinates by the actual size ratio, and clockwise rotation maps `(x,y)` to `(height-y,x)`.

Cropping marks outside points `visible=false`, preserving their array index, label, and transformed coordinates. An already invisible point never becomes visible again, even if later padding brings its coordinates inside the image. Invisible coordinates may be outside the image but must stay finite. No semantic left/right label swapping is performed for skeletons; applications that need it should apply their label permutation explicitly.

## Stable batches

`augment_batch(items, pipeline, seed, min_visibility=0.0)` accepts `BatchItem { sample_id, input: ExtendedSample }` and returns `BatchResult { sample_id, output: ExtendedAugmented }`. IDs must be non-empty and unique within the batch. Each ID receives the same stream as an individual `augment_with_keypoints` call, so reordering or partitioning a batch does not change an individual result.

`replay_batch(items, records)` accepts `BatchReplay { sample_id, replay }` records in any order and matches them to input IDs. Missing, extra, duplicate, or incompatible records are rejected before execution.

Batches contain 1–64 samples. Total input and output pixels are each limited to 16,777,216, with at most 100,000 boxes plus keypoints across the batch. Every sample, replay plan, and keypoint arithmetic result is checked before allocating the first output image. Returned images/masks/point arrays and replay padding fills are independent of input buffers.

## JSON bridge

`process_json(String) -> String` is the browser/CLI boundary. It always returns a JSON success or error object.
JSON requests are limited to 16,777,216 UTF-16 code units before parsing. A separate JSON resource budget limits decoded images, intermediate images, and returned images to 1,048,576 channel values. For a batch, decoded inputs and returned outputs each share that total budget. These checks occur before decoded byte allocation or output execution. The typed API keeps the larger pixel limits above.

```json
{
  "action": "augment",
  "sample": {
    "image": { "width": 2, "height": 1, "channels": 1, "data": [10, 20] },
    "mask": [1, 2],
    "boxes": [{ "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": "object" }]
  },
  "pipeline": [{ "op": "hflip", "p": 1 }],
  "seed": 42,
  "sample_id": "image-001",
  "min_visibility": 0.25
}
```

Operation names are `resize`, `crop`, `random_crop`, `hflip`, `vflip`, `rotate90`, `brightness`, `contrast`. Sizes use `width`/`height`; crop adds `x`/`y`; rotation uses `turns`; color operations use `factor`. `p` defaults to 1. `seed`, `sample_id`, and `min_visibility` default to 0, `""`, and 0 respectively. Missing mask and boxes mean `null` and `[]`.

Additional operation schemas are:

```json
{"op":"pad","left":1,"top":2,"right":1,"bottom":2,"fill":[114,114,114],"mask_value":255}
{"op":"letterbox","width":640,"height":640,"fill":[114,114,114],"mask_value":255,"scale_up":true}
{"op":"random_resized_crop","width":224,"height":224,"scale_min":0.08,"scale_max":1,"ratio_min":0.75,"ratio_max":1.3333333333}
```

`scale_up` defaults to true; other shown fields are required. Fill bytes must be integers from 0 to 255. Mask fills may introduce an explicit new class, e.g. the dataset's ignore label.

Success is `{ "ok": true, "sample": {...}, "replay": {...} }`. Replay requests use `{ "action": "replay", "sample": {...}, "replay": {...} }` and return the same shape. Errors use `{ "ok": false, "error": "message" }`. Numbers for bytes, dimensions, seed, and mask labels must be integers in their declared range; numeric strings are rejected.

For keypoints, add a **top-level** `keypoints: [{"x":0.5,"y":1.5,"label":"left_eye","visible":true}]` field to either action. The response includes top-level `keypoints` only when the request included it; the existing `sample` object schema is unchanged.

Batch actions use `action:"augment_batch"`, `items:[{sample_id,sample,keypoints?}]`, and the shared `pipeline`, `seed`, and `min_visibility`. `action:"replay_batch"` uses the same input items plus `records:[{sample_id,replay}]`. Both return `{ok:true,items:[{sample_id,sample,keypoints,replay}]}`. Batch responses always include a keypoints array, even when empty.

Public `sample_from_json(Json)` validates the single-sample schema and JSON decoded-value budget. `sample_to_json(Sample)` serializes an already valid sample without imposing a response budget, for callers that need their own transport; call `validate` before serializing untrusted typed input.

## Checks

```text
moon check moonaug --deny-warn
moon test moonaug --target wasm-gc --deny-warn
moon test moonaug --target wasm --deny-warn
moon test moonaug --target js --deny-warn
```

Tests cover synchronized targets, crop filtering and boundary equality, exact rotations and flips, resize sampling, color semantics, replay validation, JSON round trips, invalid input, buffer ownership, keypoint visibility history, ID-keyed batches, preflight budgets, tiny boxes, and the pinned random stream. `reference_geometry_test.mbt` and `reference_extended_test.mbt` contain independent NumPy/PyTorch fixtures, including an exhaustive small-image crop-window oracle; generation instructions are in the repository's `verification` directory.

The library operates on decoded pixels. Image codecs, arbitrary affine/perspective transforms, GPU acceleration, and training frameworks are outside this version's scope. The repository's interop package connects these samples to MoonWarp when a learned or arbitrary affine image warp is needed.
