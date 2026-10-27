# Independent numerical and geometry references

The committed fixtures and black-box MoonBit tests are generated independently
of the MoonBit implementation. Normal `moon test` does not need Python.

```powershell
python -m pip install -r verification/requirements.txt
python verification/generate_references.py
moon test
python verification/generate_references.py --check
python verification/generate_references.py --check-numerical
```

The reference uses CPU PyTorch float64, NumPy, seed `20261025`, one PyTorch
thread, and deterministic algorithms. Exact dependency versions are embedded in
both JSON files. The installed MoonBit `moonfmt` formats generated test sources
without touching implementation files. The generator prints SHA-256 hashes and fails on non-finite
values or failed finite-difference audits. `--check` compares the complete
generated files, including version metadata; run it with the pinned environment.
The committed reference environment is Python 3.13.9, NumPy 2.3.5, and
PyTorch 2.12.0+cpu. Regenerate intentionally when changing this environment.

`--check-numerical` is intended for CI across Windows/Linux and different CPUs.
It recomputes all independent references and finite-difference audits, compares
metadata/structure/shapes/integer masks strictly, and applies the documented
floating-point tolerances. Committed JSON is then used to regenerate and exactly
check the MoonBit test source, binding the reviewed fixtures to the runnable tests.
Only expected resize/photometric byte values allow a one-level difference;
permutations, categorical masks, and unchanged alpha remain exact.

## What is independently checked

* `grid_sample`: forward, input gradient and grid gradient against PyTorch
  autograd; two batches, two channels, non-square inputs, different output sizes,
  both `align_corners` values, exact normalized boundaries, partly/fully outside
  zero padding, and singleton input axes.
* `affine_grid` / composed affine warp: generated grid, forward, input gradient,
  theta gradient and standalone affine-grid backward; multiple batches,
  both align conventions, and singleton output axes.
* Central finite differences independently audit smooth PyTorch reference
  gradients for inputs, grids, and theta. Exact boundary samples instead check
  PyTorch's chosen derivative convention, since derivatives may be discontinuous.
* Augmentation: NumPy flip/rotation/slicing, PyTorch half-pixel bilinear resize
  without antialiasing, categorical nearest pixel-center masks, and continuous
  bbox corner transforms with clipping and visibility filtering.
* Combined transforms, all clockwise quarter-turns (including negative turns),
  up/down resize, RGBA alpha preservation, deterministic random pipelines,
  zero-probability operations, and serialized replay are exercised.

MoonWarp tolerance is absolute `1e-9`. Geometric bbox tolerance is `1e-10`.
Image resize/photometric values allow one uint8 level for rounding order;
categorical masks and replay pixel arrays are checked exactly. These fixtures
validate the supported CPU bilinear/zero-padding subset; they make no GPU,
performance, antialiasing, higher-order-gradient, or complete PyTorch claim.

`warp_reference.json` and `augmentation_reference.json` are reviewable inputs
and expected outputs. `moonwarp/reference_pytorch_test.mbt` and
`moonaug/reference_geometry_test.mbt` exercise only public APIs.

## Extended APIs

The original acceptance fixtures above stay separate from the additional
sampler, augmentation, and interop references:

```powershell
python verification/generate_extended_references.py
python verification/generate_extended_references.py --check-numerical
python verification/check_private_tensor_fields.py
```

`extended_warp_reference.json` covers bilinear/nearest × zeros/border/reflection
× both corner conventions, with forward/input/grid/theta gradients, exact
nearest ties, reflection cycles, singleton axes, and independent NumPy batch
matrix conversion/composition/inversion. Smooth grid/theta gradients are
audited using central finite differences; knot/tie cases compare the PyTorch
derivative convention directly.

Intentional knot/tie cases use binary-exact normalized coordinates and compatible
image dimensions. Fraction arithmetic verifies the intended pixel positions;
separate, reassociated and fused unnormalization must all agree exactly. General
smooth and affine-chain cases retain ordinary dimensions and require a measured
branch margin of at least `1e-4` input pixels. Numerical checking also requires
explicit reference grids to be bit-identical, so cross-platform rounding cannot
silently switch the derivative branch while passing a coordinate tolerance.

`extended_augmentation_reference.json` covers per-channel padding and categorical
fills, odd centered letterboxing with/without upscaling, tiny positive boxes,
and keypoint visibility through crop/pad/resize. RandomResizedCrop is checked
against all 60 possible crop windows of the small input image. The runtime's
resolved crop chooses an independent NumPy/PyTorch reference table entry;
the oracle does not implement MoonBit's random generator. Separate tests check
determinism, fallback, replay, and sample-ID invariance when batching/reordering.

`extended_interop_reference.json` checks bilinear image pixels and nearest
categorical masks against PyTorch, and boxes/keypoints against an independently
inverted homogeneous edge-coordinate matrix. It also checks multi-batch NCHW
packing, per-channel normalization, and exact int32 mask IDs.

Both generation scripts support exact `--check` in the recorded environment
and numerical `--check-numerical` across operating systems with identical
dependency versions. Numerical mode recomputes references, enforces numerical
bounds and exact structure/categories, then regenerates the checked-in tests
from the checked-in JSON to prove their correspondence.

The private-field probe uses an isolated temporary module under `_build`, first
compiles a public-API control, then requires all four external Tensor/Grid array
field-access probes to fail. It never leaves intentionally invalid source in a
normal repository package and does not exercise unbounded allocations.

## Real pixels on every backend

`python verification/generate_real_image_references.py --check-numerical` verifies
the offline manifest/file/decoded-pixel hashes, reduces all ten photo fixtures
to 10×8 with Pillow BOX, and independently recomputes PyTorch references.
The generated `moonwarp/reference_real_images_test.mbt` embeds those actual
pixels and expected forward/input/theta gradients, so normal `moon test` runs
them on native, JS, Wasm, and Wasm GC without a runtime image codec or Python.
Each photo checks bilinear/reflection plus exact nearest/border using identical
explicit grids. This small backend regression complements the separate full
photo bridge runner; it does not replace its image sizes or codec checks.

## Degraded photos and complex boundaries

`python verification/test_stress_images.py` reuses the real-photo bridge and
independent augmentation oracle on 100 deterministic degraded/degenerate inputs.
It covers Gaussian/defocus/motion blur, noise, low contrast, occlusion, flat
colors, singleton axes, and single pixels, including grayscale/RGB/RGBA photos.
Every case checks all twelve mode/padding/corner combinations and their input
and grid gradients against CPU PyTorch float64 using identical coordinates.
It also checks combined random augmentation/replay, exact adjacent maximal
int32 mask IDs, channel-wise normalized batch round trips, identity annotations,
and process recovery after malformed nested JSON fields. Selected cases check
single/multiscale registration output/theta/loss consistency and monotonicity;
ambiguous images are not required to recover a unique ground-truth affine.
Source hashes are checked against the offline manifest; generated-pixel hashes
and dependency versions are recorded in `artifacts/stress-images-report.json`.
This runner is part of CI and
`tools/verify.py --real-images`; details are in `docs/real-images.md`.
