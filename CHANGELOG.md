# Changelog

## 0.2.2 — 2026-10-28

- Avoid evaluating zero-weight neighboring differences in bilinear grid gradients, so unused extreme pixels cannot invalidate a finite boundary gradient.
- Add an importable Node client and bound CLI input before parsing; return structured file/runtime errors and respect output backpressure while preserving JSON-lines recovery.
- Add an offline real-photo batch example with checked source hashes, ID-matched replay, independently verified normalized NCHW tensors, byte-exact image round trips, and saved provenance and previews.
- Run CLI regressions and the real-photo workflow in verification/CI; clarify that the compatible random crop origin generator uses bounded modulo sampling.

## 0.2.1 — 2026-10-27

- Fix categorical resize ties (for example 2 to 49 pixels) by computing exact pixel-center indices, preserving int32 mask labels on both axes.
- Retain representable subnormal MSE values and avoid overflowing inactive bilinear gradient differences on clipped axes or zero upstream gradients.
- Stabilize channel normalization/denormalization when an intermediate overflows but the final result is finite; keep rejecting genuinely nonfinite results.
- Preserve fully visible boxes at `min_visibility=1` when affine resize roundoff moves a boundary just outside the image, with tolerance limited by each box's size so real clipping of tiny boxes is retained.
- Add 11 MoonBit regression tests and a deterministic independent stress runner covering 100 degraded real-photo/shape cases, all sampling modes and gradients, replay, int32 masks, normalized batches, malformed-input recovery, and local registration consistency. Run it in CI and with `tools/verify.py --real-images`.
- Compatibility: replay schemas and random streams are unchanged; old replays at affected resize ratios now produce corrected categorical pixels.

## 0.2.0 — 2026-10-25

- Publish on Mooncakes as `shunge/moonvision`; keep the GitHub repository at `buildliming/moonvision`.
- Audit the release archive in CI, retaining source, fixtures and attribution while excluding local delivery files.
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
