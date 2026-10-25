"""Independent PyTorch/NumPy references for MoonVision's extended public APIs.

Does not overwrite the original acceptance fixtures. --check requires exact
bytes; --check-numerical reruns the oracle with bounded platform drift and
checks that the committed MoonBit tests encode the committed JSON fixtures.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate_references import (
    DTYPE, SEED, assert_numerically_equal, finite_difference, flattened,
    mb_array, reference_affine_grid, tensor,
)


ROOT = Path(__file__).resolve().parents[1]


def sampling_case(name, image_values, image_shape, grid_values, grid_shape, mode, padding, align, audit=False):
    image = tensor(image_values, image_shape, grad=True)
    grid = tensor(grid_values, grid_shape, grad=True)
    output = F.grid_sample(image, grid, mode=mode, padding_mode=padding, align_corners=align)
    upstream = torch.linspace(-0.63, 1.17, output.numel(), dtype=DTYPE).reshape(output.shape)
    input_grad, grid_grad = torch.autograd.grad(output, (image, grid), upstream)
    case = {
        "name": name, "mode": mode, "padding": padding, "align_corners": align,
        "input_shape": list(image_shape), "input": flattened(image),
        "grid_shape": list(grid_shape), "grid": flattened(grid),
        "output_shape": list(output.shape), "output": flattened(output),
        "grad_output": flattened(upstream), "input_grad": flattened(input_grad),
        "grid_grad": flattened(grid_grad),
    }
    if audit:
        def grid_loss(values):
            result = F.grid_sample(image.detach(), tensor(values, grid_shape), mode=mode, padding_mode=padding, align_corners=align)
            return (result * upstream).sum().item()
        numerical = finite_difference(grid_loss, np.asarray(grid_values, dtype=np.float64).reshape(grid_shape))
        error = float(np.max(np.abs(numerical.reshape(-1) - np.asarray(case["grid_grad"]))))
        assert error < 2e-8, (name, error)
        case["finite_difference"] = {"epsilon": 1e-6, "grid_grad": numerical.reshape(-1).tolist(), "max_error": error}
    return case


def affine_case(name, values, shape, theta_values, output_shape, mode, padding, align):
    image = tensor(values, shape, grad=True)
    theta = tensor(theta_values, (shape[0], 2, 3), grad=True)
    grid = reference_affine_grid(theta, output_shape, align)
    output = F.grid_sample(image, grid, mode=mode, padding_mode=padding, align_corners=align)
    upstream = torch.linspace(.91, -.57, output.numel(), dtype=DTYPE).reshape(output.shape)
    input_grad, theta_grad = torch.autograd.grad(output, (image, theta), upstream)
    case = {
        "name": name, "mode": mode, "padding": padding, "align_corners": align,
        "input_shape": list(shape), "input": flattened(image), "theta": flattened(theta),
        "output_shape": list(output_shape), "output": flattened(output),
        "grad_output": flattened(upstream), "input_grad": flattened(input_grad),
        "theta_grad": flattened(theta_grad),
    }
    if mode == "bilinear":
        def theta_loss(values):
            ref_grid = reference_affine_grid(tensor(values, (shape[0], 2, 3)), output_shape, align)
            result = F.grid_sample(image.detach(), ref_grid, mode=mode, padding_mode=padding, align_corners=align)
            return (result * upstream).sum().item()
        numerical = finite_difference(theta_loss, np.asarray(theta_values).reshape(shape[0], 2, 3))
        error = float(np.max(np.abs(numerical.reshape(-1) - np.asarray(case["theta_grad"]))))
        assert error < 2e-8, (name, error)
        case["finite_difference"] = {"epsilon": 1e-6, "theta_grad": numerical.reshape(-1).tolist(), "max_error": error}
    return case


def pyramid_cases():
    """Use grouped 2D convolution, independently of MoonWarp's two-pass blur."""
    random = np.random.default_rng(SEED + 2)
    cases = []
    for shape in [(2, 2, 7, 9), (1, 2, 1, 7)]:
        values = random.uniform(-0.7, 1.3, size=shape)
        image = tensor(values, shape)
        coefficients = torch.tensor([1., 4., 6., 4., 1.], dtype=DTYPE)
        kernel = torch.outer(coefficients, coefficients).reshape(1, 1, 5, 5) / 256.
        kernel = kernel.expand(shape[1], 1, 5, 5)
        filtered = F.conv2d(F.pad(image, (2, 2, 2, 2), mode="replicate"), kernel, groups=shape[1])
        for align in (False, True):
            output = F.interpolate(filtered, size=((shape[2] + 1) // 2, (shape[3] + 1) // 2), mode="bilinear", align_corners=align)
            cases.append({
                "name": f'binomial_{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}_{align}',
                "input_shape": list(shape), "input": flattened(image),
                "align_corners": align, "output_shape": list(output.shape),
                "output": flattened(output),
            })
    return cases


def homogeneous(matrix):
    result = np.eye(3)
    result[:2] = np.asarray(matrix).reshape(2, 3)
    return result


def normalizer(height, width, align):
    # Homogeneous map from pixel-center indices (0..size-1) to normalized space.
    if align:
        return np.array([[2/(width-1), 0, -1.], [0, 2/(height-1), -1.], [0, 0, 1.]])
    return np.array([[2/width, 0, 1/width-1], [0, 2/height, 1/height-1], [0, 0, 1.]])


def matrix_cases():
    pixels = [[1.2, .13, 2.1, -.17, .93, -.8], [.59, -.31, -.2, .27, 1.17, .31]]
    inner = [[.83, .11, -.13, -.12, .91, .17], [1.09, -.19, .23, .14, .77, -.31]]
    cases = []
    for align in (False, True):
        ni, no = normalizer(3, 5, align), normalizer(4, 2, align)
        theta = [(ni @ homogeneous(p) @ np.linalg.inv(no))[:2].reshape(-1).tolist() for p in pixels]
        inverses = [np.linalg.inv(homogeneous(t))[:2].reshape(-1).tolist() for t in theta]
        composed = [(homogeneous(outer) @ homogeneous(inside))[:2].reshape(-1).tolist() for outer, inside in zip(theta, inner)]
        cases.append({
            "name": f"matrix_batch_{str(align).lower()}", "align_corners": align,
            "input_height": 3, "input_width": 5, "output_height": 4, "output_width": 2,
            "pixel_matrices": sum(pixels, []), "theta": sum(theta, []),
            "inverse": sum(inverses, []), "inner": sum(inner, []), "composition": sum(composed, []),
        })
    return cases


def warp_document():
    values = np.random.default_rng(SEED + 1).normal(size=(2, 2, 3, 4))
    grid = np.array([[-.81, -.73], [-.11, .23], [.67, .61], [1.23, -.23], [-1.21, 1.19], [2.17, -2.23], [-.61, .47], [.17, -.37], [.83, -.81], [-1.37, .07], [.43, 1.31], [-2.41, 2.33]])
    points = np.array([[-.5, .37], [.5, .37], [1.5, .37], [2.5, .37], [-5.2, -3.4], [8.7, 4.2], [.2, -.5], [.2, .5], [.2, 1.5], [0., 0.], [2., 1.], [-1., -1.], [3., 2.], [-.5, -.5], [2.5, 1.5]])
    theta = [.87, .13, .37, -.17, .79, -.29, .91, -.23, -.47, .18, .83, .31]
    samples, affines = [], []
    for mode in ("bilinear", "nearest"):
        for padding in ("zeros", "border", "reflection"):
            for align in (False, True):
                suffix = f"{mode}_{padding}_{str(align).lower()}"
                samples.append(sampling_case("multi_batch_" + suffix, values, values.shape, grid, (2, 2, 3, 2), mode, padding, align, audit=True))
                # These points deliberately include integer knots and nearest
                # half-way ties; compare PyTorch conventions, not finite differences.
                if align:
                    boundary = points * np.array([1., 2.]) - 1.
                else:
                    boundary = (points + .5) * np.array([2/3, 1.]) - 1.
                samples.append(sampling_case("knots_ties_cycles_" + suffix, [1., 3., -2., 5., 7., 4.], (1, 1, 2, 3), boundary, (1, 3, 5, 2), mode, padding, align))
                samples.append(sampling_case("singleton_" + suffix, [2., -3.], (1, 2, 1, 1), [-1000., 1000., -.4, .3, 1.2, -.7], (1, 1, 3, 2), mode, padding, align))
                affines.append(affine_case("affine_chain_" + suffix, values, values.shape, theta, (2, 2, 2, 3), mode, padding, align))
    return {
        "metadata": {
            "seed": SEED + 1, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "dtype": "float64", "oracle": "torch.nn.functional.grid_sample + affine_grid + autograd; NumPy homogeneous matrix algebra; grouped 2D convolution + interpolate for pyramids",
            "absolute_tolerance": 1e-9, "finite_difference_tolerance": 2e-8,
            "pixel_matrix_direction": "output pixel-center index -> input pixel-center index",
        },
        "sample_cases": samples, "affine_cases": affines, "matrix_cases": matrix_cases(),
        "pyramid_cases": pyramid_cases(),
    }


def options(case):
    mode = {"bilinear": "Bilinear", "nearest": "Nearest"}[case["mode"]]
    padding = {"zeros": "Zeros", "border": "Border", "reflection": "Reflection"}[case["padding"]]
    return f'align_corners={str(case["align_corners"]).lower()}, mode=@moonwarp.{mode}, padding=@moonwarp.{padding}'


def emit_warp(document):
    lines = ["// Generated from independent PyTorch and NumPy references.", "",
             "fn extended_close(actual : Array[Double], expected : Array[Double]) -> Unit raise {",
             "  assert_eq(actual.length(), expected.length())",
             "  for i = 0; i < actual.length(); i = i + 1 {",
             "    let delta = actual[i] - expected[i]",
             "    assert_true(delta >= -1.0e-9 && delta <= 1.0e-9)", "  }", "}"]
    for case in document["sample_cases"]:
        lines += [f'test "extended PyTorch: {case["name"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let grid = @moonwarp.Grid::new({mb_array(case["grid_shape"], "int")}, {mb_array(case["grid"])})',
                  f'  let output = @moonwarp.grid_sample(input, grid, {options(case)})',
                  f'  assert_eq(output.shape(), {mb_array(case["output_shape"], "int")})',
                  f'  extended_close(output.data(), {mb_array(case["output"])})',
                  f'  let upstream = @moonwarp.Tensor::new({mb_array(case["output_shape"], "int")}, {mb_array(case["grad_output"])})',
                  f'  let grads = @moonwarp.grid_sample_backward(input, grid, upstream, {options(case)})',
                  f'  extended_close(grads.input_grad.data(), {mb_array(case["input_grad"])})',
                  f'  extended_close(grads.grid_grad.data(), {mb_array(case["grid_grad"])})', "}"]
    for case in document["affine_cases"]:
        _, _, height, width = case["output_shape"]
        lines += [f'test "extended PyTorch: {case["name"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let theta = {mb_array(case["theta"])}',
                  f'  let output = @moonwarp.warp_affine(input, theta, {height}, {width}, {options(case)})',
                  f'  extended_close(output.data(), {mb_array(case["output"])})',
                  f'  let upstream = @moonwarp.Tensor::new({mb_array(case["output_shape"], "int")}, {mb_array(case["grad_output"])})',
                  f'  let grads = @moonwarp.warp_affine_backward(input, theta, upstream, {options(case)})',
                  f'  extended_close(grads.input_grad.data(), {mb_array(case["input_grad"])})',
                  f'  extended_close(grads.theta_grad, {mb_array(case["theta_grad"])})', "}"]
    for case in document["matrix_cases"]:
        dims = ", ".join(str(case[k]) for k in ("input_height", "input_width", "output_height", "output_width"))
        align = str(case["align_corners"]).lower()
        lines += [f'test "independent NumPy: {case["name"]}" {{',
                  f'  let pixels = {mb_array(case["pixel_matrices"])}',
                  f'  let theta = @moonwarp.pixel_affine_to_theta(pixels, {dims}, align_corners={align})',
                  f'  extended_close(theta, {mb_array(case["theta"])})',
                  f'  extended_close(@moonwarp.theta_to_pixel_affine(theta, {dims}, align_corners={align}), pixels)',
                  f'  extended_close(@moonwarp.invert_theta(theta), {mb_array(case["inverse"])})',
                  f'  extended_close(@moonwarp.compose_theta(theta, {mb_array(case["inner"])}), {mb_array(case["composition"])})', "}"]
    for case in document["pyramid_cases"]:
        align = str(case["align_corners"]).lower()
        lines += [f'test "independent PyTorch pyramid: {case["name"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let output = @moonwarp.pyramid_downsample(input, align_corners={align})',
                  f'  assert_eq(output.shape(), {mb_array(case["output_shape"], "int")})',
                  f'  extended_close(output.data(), {mb_array(case["output"])})', "}"]
    return "\n".join(lines) + "\n"


def canonical(source):
    formatter = shutil.which("moonfmt")
    if formatter is None:
        raise SystemExit("moonfmt is required")
    for _ in range(4):
        result = subprocess.run([formatter, "-"], input=source, capture_output=True, text=True, encoding="utf-8", check=True)
        if result.stdout == source:
            return source
        source = result.stdout
    raise SystemExit("moonfmt did not converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--check-numerical", action="store_true")
    args = parser.parse_args()
    documents = [("extended_warp_reference.json", "warp", warp_document(), ROOT / "moonwarp" / "reference_extended_test.mbt", emit_warp)]
    # Augmentation reference construction lives separately so matrix/sampling
    # references do not depend on the augmentation implementation or its PRNG.
    from extended_augmentation_reference import aug_document, emit_aug
    documents.append(("extended_augmentation_reference.json", "augmentation", aug_document(), ROOT / "moonaug" / "reference_extended_test.mbt", emit_aug))
    from extended_interop_reference import check_interop_numerically, emit_interop, interop_document
    documents.append(("extended_interop_reference.json", "interop", interop_document(), ROOT / "interop" / "reference_interop_test.mbt", emit_interop))
    for filename, kind, fresh, test_path, emitter in documents:
        fixture_path = ROOT / "verification" / filename
        selected = fresh
        fresh_text = json.dumps(fresh, indent=2, allow_nan=False) + "\n"
        if args.check_numerical:
            selected = json.loads(fixture_path.read_text(encoding="utf-8"))
            if kind == "interop":
                check_interop_numerically(fresh, selected)
            else:
                assert_numerically_equal(fresh, selected, kind)
        elif args.check:
            if fixture_path.read_text(encoding="utf-8") != fresh_text:
                raise SystemExit(f"Reference differs: {filename}")
        else:
            fixture_path.write_text(fresh_text, encoding="utf-8", newline="\n")
        source = canonical(emitter(selected))
        if args.check or args.check_numerical:
            if test_path.read_text(encoding="utf-8") != source:
                raise SystemExit(f"Generated tests differ: {test_path.relative_to(ROOT)}")
        else:
            test_path.write_text(source, encoding="utf-8", newline="\n")
        print(f'{"checked" if args.check or args.check_numerical else "generated"} {filename} and {test_path.relative_to(ROOT)}')
    print("Extended independent oracle and smooth finite-difference audits passed.")


if __name__ == "__main__":
    main()
