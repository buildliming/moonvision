# Changelog

## 0.2.0 — 2026-10-25

- Fix tiny-box area underflow/degenerate geometry and protect Tensor/Grid internal buffers from external mutation.
- Add Pad, Letterbox, RandomResizedCrop, stable keypoints, ID-keyed batches and validated Replay v2; retain v1 compatibility and old random streams.
- Add nearest/bilinear sampling with zeros/border/reflection padding and complete first-order gradients, robust affine matrix utilities and antialiased multiscale registration.
- Connect full image/mask/box/keypoint warps, image batches, exact integer masks and channel normalization through typed and JSON APIs.
- Add attributed offline photographs, independent full-image validation, cross-backend real-pixel references, extended numerical oracles and negative compilation probes.
- Extend the browser gallery and typed examples; document numerical discontinuities, limits and 0.2 migration.

## 0.1.0 — 2026-10-25

- MoonAug: synchronized images, categorical masks and labeled boxes; deterministic probabilistic pipelines; versioned JSON replay.
- MoonWarp: affine grids, bilinear zero-padded sampling, analytic input/grid/theta gradients and local affine registration.
- Interoperability between byte HWC images and normalized NCHW tensors.
- Independent NumPy/PyTorch fixtures, gradient audits, malformed-input regression tests and four-backend verification.
- Node JSON CLI, typed examples, browser playground, reproduction instructions and CI.
