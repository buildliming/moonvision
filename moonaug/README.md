# MoonAug

Annotation-aware, reproducible image augmentation in pure MoonBit. This package is part of `buildliming/moonvision`; it has no third-party dependencies and no FFI.

## Quick start

Import `buildliming/moonvision/moonaug` in your package configuration:

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
- `Sample.mask` is `None` or a flat row-major `Array[Int]` of non-negative class IDs matching the image size. A mask is a class map, not an RGB visualization.
- `BBox` uses continuous **half-open** `x_min, y_min, x_max, y_max`, with origin at the top left. Input boxes must have positive area and lie inside the image. `label` is preserved alongside each retained box.
- All operations return owned buffers and leave caller input untouched, including empty pipelines.
- Samples are limited to 100,000 boxes. Pipelines and replay records are limited to 256 operations, including skipped steps. These limits apply to both the typed API and JSON bridge, and are checked before executing the operations.

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

Color factors must be finite and in `[0, 16]`. RGBA alpha remains unchanged by color transforms. Spatial resizing interpolates all four RGBA channels independently in their stored byte space; it does not perform alpha premultiplication or linear-light conversion.

Every `Step` has an independent application probability in `[0, 1]`. Operation parameter ranges are validated even for zero-probability steps; shape-dependent fit checks apply when a step executes. A crop retains a box if its nonzero clipped area divided by its pre-crop area is at least `min_visibility`. This policy is applied at each crop, not relative to the original box before the entire pipeline.

## Reproducibility and replay

`augment(sample, pipeline, seed, sample_id, min_visibility=0.0)` keys its random stream by the signed 32-bit seed and Unicode sample ID. No wall clock or global random state is used. The random stream is pinned by tests, so input ordering does not affect an individual sample's result. Different IDs or seeds can produce the same result; they do not guarantee distinct transformations.

`Replay` version 1 records the exact original width, height, channels, the visibility threshold, and resolved applied operations. Skipped steps are omitted. `replay` validates the original shape and rejects unknown versions or unresolved random crops. It may replay onto a different sample with the same shape.

Use `replay_to_json(record).stringify()` to persist a record and `replay_from_json(@json.parse(text))` to load it. Replay contains operation values and dimensions, not source pixels or input identity. Serialize your sample ID and dataset identity separately if the application needs that provenance.

## JSON bridge

`process_json(String) -> String` is the browser/CLI boundary. It always returns a JSON success or error object.
JSON requests are limited to 16,777,216 UTF-16 code units before parsing. The typed API accepts larger decoded buffers up to the pixel limit; JSON's text-size limit may be reached first for large byte arrays.

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

Success is `{ "ok": true, "sample": {...}, "replay": {...} }`. Replay requests use `{ "action": "replay", "sample": {...}, "replay": {...} }` and return the same shape. Errors use `{ "ok": false, "error": "message" }`. Numbers for bytes, dimensions, seed, and mask labels must be integers in their declared range; numeric strings are rejected.

## Checks

```text
moon check moonaug --deny-warn
moon test moonaug --target wasm-gc --deny-warn
moon test moonaug --target wasm --deny-warn
moon test moonaug --target js --deny-warn
```

Tests cover synchronized targets, crop filtering and boundary equality, exact rotations and flips, resize sampling, color semantics, replay validation, JSON round trips, invalid input, buffer ownership, and the pinned random stream. `reference_geometry_test.mbt` contains independently generated NumPy/PyTorch reference fixtures; generation instructions are in the repository's `verification` directory.

The library operates on decoded pixels. Image codecs, keypoints, arbitrary affine/perspective transforms, GPU acceleration, and training frameworks are outside this version's scope.
