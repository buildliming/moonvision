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
