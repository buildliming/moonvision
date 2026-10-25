"""Embed small, attributed offline photographs in cross-backend MoonBit tests.

All manifest file/pixel hashes are checked before Pillow reduces each decoded
fixture to 10x8 pixels. PyTorch float64 independently supplies bilinear/reflection
forward, input/theta gradients, and direct-grid nearest/border references.
No codec, Python, filesystem image access, or network is needed by moon test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F

from generate_extended_references import canonical
from generate_references import DTYPE, SEED, assert_numerically_equal, finite_difference, flattened, mb_array, reference_affine_grid, tensor


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "web" / "samples"
THUMBNAIL = (10, 8)
THETA = [.873, -.197, .143, .127, .911, -.083]


def verified_thumbnails():
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1, "unsupported image-manifest version"
    identifiers = set()
    for fixture in manifest["fixtures"]:
        assert fixture["mode"] in ("L", "RGB", "RGBA"), "unsupported photo mode"
        assert fixture["id"] not in identifiers, "duplicate image fixture ID"
        identifiers.add(fixture["id"])
        assert fixture["source_id"] in manifest["sources"], "unattributed image fixture"
        path = (SAMPLES / fixture["file"]).resolve()
        assert path.is_relative_to(SAMPLES.resolve()), "image path escapes sample directory"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"], f'file hash mismatch: {fixture["id"]}'
        with Image.open(path) as decoded:
            decoded.load()
            assert decoded.mode == fixture["mode"] and decoded.size == (fixture["width"], fixture["height"]), f'image header mismatch: {fixture["id"]}'
            assert len(decoded.getbands()) == fixture["channels"], "channel count mismatch"
            assert hashlib.sha256(decoded.tobytes()).hexdigest() == fixture["pixels_sha256"], f'pixel hash mismatch: {fixture["id"]}'
            reduced = decoded.resize(THUMBNAIL, resample=Image.Resampling.BOX)
            values = np.array(reduced, dtype=np.uint8)
            if values.ndim == 2:
                values = values[:, :, None]
        yield manifest, fixture, values


def reference_document():
    cases = []
    manifest = None
    for index, (manifest, fixture, pixels) in enumerate(verified_thumbnails()):
        height, width, channels = pixels.shape
        shape = (1, channels, height, width)
        values = pixels.transpose(2, 0, 1)[None].copy()/255.
        image = tensor(values, shape, grad=True)
        theta = tensor(THETA, (1, 2, 3), grad=True)
        align = index % 2 == 1
        output_shape = (1, channels, 5, 6)
        grid = reference_affine_grid(theta, output_shape, align)
        output = F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=align)
        upstream = torch.linspace(-.37, .83, output.numel(), dtype=DTYPE).reshape(output_shape)
        input_grad, theta_grad = torch.autograd.grad(output, (image, theta), upstream)
        # The same explicit grid is supplied to both implementations for nearest
        # sampling. This does not conflate affine-grid last-bit drift at a
        # discontinuous half-pixel decision with a nearest-sampler error.
        direct_grid = grid.detach().clone().requires_grad_()
        nearest_input = image.detach().clone().requires_grad_()
        nearest = F.grid_sample(nearest_input, direct_grid, mode="nearest", padding_mode="border", align_corners=align)
        nearest_input_grad, nearest_grid_grad = torch.autograd.grad(nearest, (nearest_input, direct_grid), upstream)
        assert torch.count_nonzero(nearest_grid_grad).item() == 0
        def theta_loss(coefficients):
            ref_grid = reference_affine_grid(tensor(coefficients, (1, 2, 3)), output_shape, align)
            result = F.grid_sample(image.detach(), ref_grid, mode="bilinear", padding_mode="reflection", align_corners=align)
            return (result*upstream).sum().item()
        numerical = finite_difference(theta_loss, np.asarray(THETA).reshape(1, 2, 3))
        error = float(np.max(np.abs(numerical.reshape(-1)-theta_grad.detach().numpy().reshape(-1))))
        assert error < 2e-8, f'nonsmooth or inaccurate theta-gradient reference: {fixture["id"]}, {error}'
        cases.append({
            "id": fixture["id"], "source_id": fixture["source_id"], "file": fixture["file"],
            "source_file_sha256": fixture["sha256"], "source_pixels_sha256": fixture["pixels_sha256"],
            "thumbnail_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "mode": fixture["mode"], "thumbnail_width": width, "thumbnail_height": height,
            "thumbnail_channels": channels, "thumbnail_data": pixels.reshape(-1).tolist(),
            "input_shape": list(shape), "input": flattened(image), "theta": THETA,
            "align_corners": align, "output_shape": list(output_shape), "grad_output": flattened(upstream),
            "reflection_output": flattened(output), "reflection_input_grad": flattened(input_grad),
            "reflection_theta_grad": flattened(theta_grad),
            "direct_grid_shape": list(direct_grid.shape), "direct_grid": flattened(direct_grid),
            "nearest_output": flattened(nearest), "nearest_input_grad": flattened(nearest_input_grad),
            "finite_difference": {"epsilon": 1e-6, "theta_grad": numerical.reshape(-1).tolist(), "max_error": error},
        })
    assert manifest is not None and cases, "no offline image fixtures"
    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "metadata": {
            "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "pillow": PIL.__version__, "device": "cpu", "dtype": "float64",
            "manifest_content_sha256": hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest(),
            "source_collection": manifest["source_collection"], "fixture_count": len(cases),
            "original_photo_count": len({case["source_id"] for case in cases}),
            "thumbnail": "Pillow BOX resize to 10x8; source mode/channels preserved",
            "oracle": "torch float64 bilinear/reflection affine chain and direct-grid nearest/border; theta central differences",
            "absolute_tolerance": 1e-9, "finite_difference_tolerance": 2e-8,
            "nearest_forward_policy": "exact values from identical explicitly supplied grid; no relaxed nearest pixel tolerance",
            "scope": "Small decoded real-photo pixels embedded in typed tests for every backend; full-resolution JSON/codec coverage is separate.",
        },
        "cases": cases,
    }


def emit_tests(document):
    lines = ["// Generated from hash-verified offline photographs; no runtime image codec.",
             "// See web/samples/ATTRIBUTION.md and verification/real_image_backend_reference.json.", "",
             "fn real_photo_close(actual : Array[Double], expected : Array[Double]) -> Unit raise {",
             "  assert_eq(actual.length(), expected.length())",
             "  for i = 0; i < actual.length(); i = i + 1 { let delta = actual[i]-expected[i]; assert_true(delta >= -1.0e-9 && delta <= 1.0e-9) }", "}"]
    for case in document["cases"]:
        align = str(case["align_corners"]).lower()
        lines += [f'// Offline fixture {case["file"]}, SHA-256 {case["source_file_sha256"]}.',
                  f'test "real photo cross-backend: {case["id"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let theta = {mb_array(case["theta"])}',
                  f'  let output = @moonwarp.warp_affine(input, theta, 5, 6, mode=@moonwarp.Bilinear, padding=@moonwarp.Reflection, align_corners={align})',
                  f'  assert_eq(output.shape(), {mb_array(case["output_shape"], "int")})',
                  f'  real_photo_close(output.data(), {mb_array(case["reflection_output"])})',
                  f'  let upstream = @moonwarp.Tensor::new({mb_array(case["output_shape"], "int")}, {mb_array(case["grad_output"])})',
                  f'  let gradient = @moonwarp.warp_affine_backward(input, theta, upstream, mode=@moonwarp.Bilinear, padding=@moonwarp.Reflection, align_corners={align})',
                  f'  real_photo_close(gradient.input_grad.data(), {mb_array(case["reflection_input_grad"])})',
                  f'  real_photo_close(gradient.theta_grad, {mb_array(case["reflection_theta_grad"])})',
                  f'  let direct_grid = @moonwarp.Grid::new({mb_array(case["direct_grid_shape"], "int")}, {mb_array(case["direct_grid"])})',
                  f'  let nearest = @moonwarp.grid_sample(input, direct_grid, mode=@moonwarp.Nearest, padding=@moonwarp.Border, align_corners={align})',
                  f'  assert_eq(nearest.data(), {mb_array(case["nearest_output"])})',
                  f'  let nearest_grad = @moonwarp.grid_sample_backward(input, direct_grid, upstream, mode=@moonwarp.Nearest, padding=@moonwarp.Border, align_corners={align})',
                  f'  real_photo_close(nearest_grad.input_grad.data(), {mb_array(case["nearest_input_grad"])})',
                  '  for value in nearest_grad.grid_grad.data() { assert_eq(value, 0.0) }',
                  f'  assert_eq(input.data(), {mb_array(case["input"])})', '}']
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="require exact generated bytes in the recorded environment")
    group.add_argument("--check-numerical", action="store_true", help="compare regenerated numbers with strict metadata/structure and bind tests to stored fixtures")
    args = parser.parse_args()
    fresh = reference_document()
    fixture_path = ROOT / "verification" / "real_image_backend_reference.json"
    test_path = ROOT / "moonwarp" / "reference_real_images_test.mbt"
    fixture_text = json.dumps(fresh, indent=2, allow_nan=False) + "\n"
    selected = fresh
    if args.check_numerical:
        selected = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert_numerically_equal(fresh, selected, "warp")
    elif args.check:
        if fixture_path.read_text(encoding="utf-8") != fixture_text:
            raise SystemExit("Real-photo backend reference fixture differs")
    else:
        fixture_path.write_text(fixture_text, encoding="utf-8", newline="\n")
    source = canonical(emit_tests(selected))
    if args.check or args.check_numerical:
        if test_path.read_text(encoding="utf-8") != source:
            raise SystemExit("Real-photo backend reference tests differ from committed JSON")
    else:
        test_path.write_text(source, encoding="utf-8", newline="\n")
    verb = "checked" if args.check or args.check_numerical else "generated"
    print(f'{verb} {len(fresh["cases"])} real-photo backend cases from {fresh["metadata"]["original_photo_count"]} original photographs; all file/pixel hashes and theta finite differences passed')


if __name__ == "__main__":
    main()
