"""Regenerate independent CPU reference fixtures and MoonBit black-box tests.

Run from any directory: python verification/generate_references.py [--check]
No MoonBit implementation is imported: PyTorch supplies grid_sample,
affine_grid, autograd, and bilinear interpolation; NumPy supplies image
permutations. Bounding-box geometry is computed from continuous corners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261025
DTYPE = torch.float64
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def flattened(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().reshape(-1).tolist()


def tensor(values, shape, *, grad=False):
    return torch.tensor(values, dtype=DTYPE).reshape(shape).requires_grad_(grad)


def finite_difference(function, values: np.ndarray, epsilon=1e-6):
    result = np.zeros_like(values, dtype=np.float64)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result


def sample_case(name, values, input_shape, grid_values, grid_shape, align, *, audit=False):
    image = tensor(values, input_shape, grad=True)
    grid = tensor(grid_values, grid_shape, grad=True)
    result = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=align)
    weights = torch.linspace(-0.7, 1.3, result.numel(), dtype=DTYPE).reshape(result.shape)
    (result * weights).sum().backward()
    case = {
        "name": name, "align_corners": align,
        "input_shape": list(input_shape), "input": flattened(image),
        "grid_shape": list(grid_shape), "grid": flattened(grid),
        "output_shape": list(result.shape), "output": flattened(result),
        "grad_output": flattened(weights),
        "input_grad": flattened(image.grad), "grid_grad": flattened(grid.grad),
    }
    if audit:
        def image_loss(array):
            output = F.grid_sample(tensor(array, input_shape), grid.detach(), align_corners=align)
            return (output * weights).sum().item()
        def grid_loss(array):
            output = F.grid_sample(image.detach(), tensor(array, grid_shape), align_corners=align)
            return (output * weights).sum().item()
        image_fd = finite_difference(image_loss, np.asarray(case["input"]).reshape(input_shape))
        grid_fd = finite_difference(grid_loss, np.asarray(case["grid"]).reshape(grid_shape))
        image_error = np.max(np.abs(image_fd.reshape(-1) - np.asarray(case["input_grad"])))
        grid_error = np.max(np.abs(grid_fd.reshape(-1) - np.asarray(case["grid_grad"])))
        assert image_error < 2e-8 and grid_error < 2e-8, (image_error, grid_error)
        case["finite_difference"] = {
            "epsilon": 1e-6, "input_grad": image_fd.reshape(-1).tolist(),
            "grid_grad": grid_fd.reshape(-1).tolist(),
            "max_input_error": float(image_error), "max_grid_error": float(grid_error),
        }
    return case


def reference_affine_grid(theta, output_shape, align):
    # Unit axes deliberately exercise PyTorch's documented zero-base convention.
    # Suppress only that advisory; numerical and unrelated warnings remain visible.
    with warnings.catch_warnings():
        if align and 1 in output_shape[-2:]:
            warnings.filterwarnings(
                "ignore",
                message=r"Since version 1\.3\.0, affine_grid behavior has changed for unit-size grids when align_corners=True\.",
                category=UserWarning,
            )
        return F.affine_grid(theta, output_shape, align_corners=align)


def affine_case(name, values, input_shape, theta_values, output_shape, align, *, audit=False):
    image = tensor(values, input_shape, grad=True)
    theta = tensor(theta_values, (input_shape[0], 2, 3), grad=True)
    grid = reference_affine_grid(theta, output_shape, align)
    grid.retain_grad()
    output = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=align)
    weights = torch.linspace(0.9, -0.4, output.numel(), dtype=DTYPE).reshape(output_shape)
    (output * weights).sum().backward()
    case = {
        "name": name, "align_corners": align,
        "input_shape": list(input_shape), "input": flattened(image),
        "theta": flattened(theta), "grid_shape": list(grid.shape), "grid": flattened(grid),
        "output_shape": list(output_shape), "output": flattened(output),
        "grad_output": flattened(weights), "input_grad": flattened(image.grad),
        "grid_grad": flattened(grid.grad), "theta_grad": flattened(theta.grad),
    }
    if audit:
        def theta_loss(array):
            ref_grid = reference_affine_grid(tensor(array, (input_shape[0], 2, 3)), output_shape, align)
            result = F.grid_sample(image.detach(), ref_grid, align_corners=align)
            return (result * weights).sum().item()
        numerical = finite_difference(theta_loss, np.asarray(theta_values).reshape(input_shape[0], 2, 3))
        error = np.max(np.abs(numerical.reshape(-1) - np.asarray(case["theta_grad"])))
        assert error < 2e-8, error
        case["finite_difference"] = {"epsilon": 1e-6, "theta_grad": numerical.reshape(-1).tolist(), "max_error": float(error)}
    return case


def warp_references():
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(2, 2, 3, 4))
    grid = np.array([
        [-0.81, -0.73], [-0.11, 0.23], [0.67, 0.61],
        [1.23, -0.23], [-1.21, 1.19], [2.1, -2.2],
        [-0.61, 0.47], [0.17, -0.37], [0.83, -0.81],
        [-1.37, 0.07], [0.43, 1.31], [-2.4, 2.3],
    ])
    samples = [sample_case(f"multi_batch_channel_{str(align).lower()}", values, values.shape, grid, (2, 2, 3, 2), align, audit=True) for align in (False, True)]
    boundary_grid = [-1., -1., 1., 1., -3., 0., 3., 0., 0., -3., 0., 3., 0., 0., -1.5, -1.5, -1.5, 0., 0., -1.5, 1.5, 0., 0., 1.5]
    samples.extend(sample_case(f"boundary_zero_padding_{str(align).lower()}", [1., 3., 5., 7.], (1, 1, 2, 2), boundary_grid, (1, 3, 4, 2), align) for align in (False, True))
    samples.extend(sample_case(f"singleton_input_{str(align).lower()}", [2., -3.], (1, 2, 1, 1), [-0.4, 0.3, 1.2, -0.7, 2.5, 2.1], (1, 1, 3, 2), align) for align in (False, True))
    theta = [.63, .11, .13, -.17, .59, -.09, .71, -.13, -.19, .08, .67, .17]
    affines = [affine_case(f"affine_multi_batch_{str(align).lower()}", values, values.shape, theta, (2, 2, 2, 3), align, audit=True) for align in (False, True)]
    affines.extend(affine_case(f"singleton_output_axis_{str(align).lower()}", [1., 2., 4., -1., 3., 5.], (1, 1, 2, 3), [.8, .2, .11, -.1, .7, -.09], (1, 1, 1, 2), align) for align in (False, True))
    return {"sample_cases": samples, "affine_cases": affines}


def transform_box(box, matrix, width, height, visibility):
    x0, y0, x1, y1 = box[:4]
    points = np.array([[x0, y0, 1.], [x1, y0, 1.], [x0, y1, 1.], [x1, y1, 1.]]) @ matrix.T
    low, high = points.min(axis=0), points.max(axis=0)
    full_area = np.prod(high - low)
    clipped_low = np.maximum(low, [0., 0.])
    clipped_high = np.minimum(high, [float(width), float(height)])
    size = clipped_high - clipped_low
    if np.any(size <= 0.) or np.prod(size) / full_area < visibility:
        return None
    return [*clipped_low.tolist(), *clipped_high.tolist(), box[4]]


def augmentation_reference(image, mask, boxes, operations, visibility):
    image, mask, boxes = image.copy(), mask.copy(), [list(box) for box in boxes]
    for op in operations:
        height, width = image.shape[:2]
        kind, *args = op
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        if kind == "HFlip":
            image, mask = np.flip(image, 1), np.flip(mask, 1)
            matrix = np.array([[-1., 0., width], [0., 1., 0.]])
        elif kind == "VFlip":
            image, mask = np.flip(image, 0), np.flip(mask, 0)
            matrix = np.array([[1., 0., 0.], [0., -1., height]])
        elif kind == "Rotate90":
            turns = args[0] % 4
            image, mask = np.rot90(image, -turns), np.rot90(mask, -turns)
            matrix = [matrix, np.array([[0., -1., height], [1., 0., 0.]]), np.array([[-1., 0., width], [0., -1., height]]), np.array([[0., 1., 0.], [-1., 0., width]])][turns]
        elif kind == "Crop":
            x, y, w, h = args
            image, mask = image[y:y+h, x:x+w], mask[y:y+h, x:x+w]
            matrix = np.array([[1., 0., -x], [0., 1., -y]])
        elif kind == "Resize":
            w, h = args
            source = torch.tensor(image.copy(), dtype=DTYPE).permute(2, 0, 1).unsqueeze(0)
            resized = F.interpolate(source, size=(h, w), mode="bilinear", align_corners=False, antialias=False)
            image = np.clip(np.floor(resized.squeeze(0).permute(1, 2, 0).numpy() + .5), 0, 255).astype(np.uint8)
            yy = np.minimum(np.floor((np.arange(h) + .5) * height / h).astype(int), height-1)
            xx = np.minimum(np.floor((np.arange(w) + .5) * width / w).astype(int), width-1)
            mask = mask[np.ix_(yy, xx)]
            matrix = np.array([[w/width, 0., 0.], [0., h/height, 0.]])
        elif kind in ("Brightness", "Contrast"):
            channels = min(image.shape[2], 3)
            result = image.copy()
            colors = result[:, :, :channels].astype(float)
            colors = colors * args[0] if kind == "Brightness" else (colors - 128.) * args[0] + 128.
            result[:, :, :channels] = np.clip(np.floor(colors+.5), 0, 255).astype(np.uint8)
            image = result
        else:
            raise ValueError(kind)
        height, width = image.shape[:2]
        boxes = [new for box in boxes if (new := transform_box(box, matrix, width, height, visibility)) is not None]
    return {"width": image.shape[1], "height": image.shape[0], "channels": image.shape[2], "data": image.reshape(-1).tolist(), "mask": mask.reshape(-1).tolist(), "boxes": boxes}


def aug_references():
    image = ((np.arange(3*4*3).reshape(3, 4, 3) * 7 + 5) % 256).astype(np.uint8)
    mask = np.array([[0, 0, 1, 1], [2, 3, 3, 1], [2, 2, 4, 4]], dtype=np.int64)
    boxes = [[.25, .5, 3.6, 2.6, "main"], [0., 0., 1., 1., "corner"], [3., 2., 4., 3., "edge"]]
    specs = [
        ("horizontal_flip", [["HFlip"]], 0.),
        ("vertical_flip", [["VFlip"]], 0.),
        ("clockwise_non_square", [["Rotate90", 1]], 0.),
        ("half_turn", [["Rotate90", 2]], 0.),
        ("counterclockwise_non_square", [["Rotate90", -1]], 0.),
        ("crop_visibility", [["Crop", 1, 0, 2, 3]], .55),
        ("resize_halfpixel", [["Resize", 7, 5]], 0.),
        ("resize_downsample", [["Resize", 2, 2]], 0.),
        ("combined_annotation_geometry", [["HFlip"], ["Crop", 1, 0, 3, 2], ["Rotate90", 1], ["Resize", 5, 4]], .15),
    ]
    cases = []
    for name, operations, visibility in specs:
        cases.append({"name": name, "input": {"width": 4, "height": 3, "channels": 3, "data": image.reshape(-1).tolist(), "mask": mask.reshape(-1).tolist(), "boxes": boxes}, "operations": operations, "min_visibility": visibility, "expected": augmentation_reference(image, mask, boxes, operations, visibility)})
    rgba = np.concatenate([image, np.array([[0, 1, 127, 255], [12, 45, 78, 90], [255, 224, 200, 100]], dtype=np.uint8)[:, :, None]], 2)
    operations = [["Brightness", 1.2], ["Contrast", .7]]
    cases.append({"name": "rgba_photometric_preserves_alpha", "input": {"width": 4, "height": 3, "channels": 4, "data": rgba.reshape(-1).tolist(), "mask": mask.reshape(-1).tolist(), "boxes": boxes}, "operations": operations, "min_visibility": 0., "expected": augmentation_reference(rgba, mask, boxes, operations, 0.)})
    return {"cases": cases}


def mb_double(value):
    value = float(value)
    assert math.isfinite(value)
    return repr(value)


def mb_array(values, kind="double"):
    formatter = {"double": mb_double, "int": str, "byte": lambda value: str(value).replace("b", "")}[kind]
    return "[" + ", ".join(formatter(v) for v in values) + "]"


def emit_warp_tests(reference):
    lines = [
        "// Generated by verification/generate_references.py. Do not edit by hand.",
        "// PyTorch CPU float64 reference; includes independently audited finite differences.",
        "fn reference_close(got : Array[Double], expected : Array[Double]) -> Unit raise {",
        "  assert_eq(got.length(), expected.length())",
        "  for i = 0; i < got.length(); i = i + 1 {",
        "    let delta = got[i] - expected[i]",
        "    assert_true(delta >= -1.0e-9 && delta <= 1.0e-9)",
        "  }", "}",
    ]
    for case in reference["sample_cases"]:
        align = str(case["align_corners"]).lower()
        lines += [f'test "PyTorch reference: {case["name"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let grid = @moonwarp.Grid::new({mb_array(case["grid_shape"], "int")}, {mb_array(case["grid"])})',
                  f'  let output = @moonwarp.grid_sample(input, grid, align_corners={align})',
                  f'  assert_eq(output.shape(), {mb_array(case["output_shape"], "int")})',
                  f'  reference_close(output.data(), {mb_array(case["output"])})',
                  f'  let upstream = @moonwarp.Tensor::new({mb_array(case["output_shape"], "int")}, {mb_array(case["grad_output"])})',
                  f'  let grads = @moonwarp.grid_sample_backward(input, grid, upstream, align_corners={align})',
                  f'  reference_close(grads.input_grad.data(), {mb_array(case["input_grad"])})',
                  f'  reference_close(grads.grid_grad.data(), {mb_array(case["grid_grad"])})', "}"]
    for case in reference["affine_cases"]:
        align = str(case["align_corners"]).lower()
        n, _, h, w = case["output_shape"]
        lines += [f'test "PyTorch reference: {case["name"]}" {{',
                  f'  let input = @moonwarp.Tensor::new({mb_array(case["input_shape"], "int")}, {mb_array(case["input"])})',
                  f'  let theta = {mb_array(case["theta"])}',
                  f'  let grid = @moonwarp.affine_grid(theta, {n}, {h}, {w}, align_corners={align})',
                  f'  assert_eq(grid.shape(), {mb_array(case["grid_shape"], "int")})',
                  f'  reference_close(grid.data(), {mb_array(case["grid"])})',
                  f'  let output = @moonwarp.warp_affine(input, theta, {h}, {w}, align_corners={align})',
                  f'  reference_close(output.data(), {mb_array(case["output"])})',
                  f'  let upstream = @moonwarp.Tensor::new({mb_array(case["output_shape"], "int")}, {mb_array(case["grad_output"])})',
                  f'  let grads = @moonwarp.warp_affine_backward(input, theta, upstream, align_corners={align})',
                  f'  reference_close(grads.input_grad.data(), {mb_array(case["input_grad"])})',
                  f'  reference_close(grads.theta_grad, {mb_array(case["theta_grad"])})',
                  f'  let grid_grad = @moonwarp.Grid::new({mb_array(case["grid_shape"], "int")}, {mb_array(case["grid_grad"])})',
                  f'  reference_close(@moonwarp.affine_grid_backward(grid_grad, align_corners={align}), {mb_array(case["theta_grad"])})', "}"]
    return "\n".join(lines) + "\n"


def mb_box(box):
    x0, y0, x1, y1, label = box
    return f'@moonaug.BBox::{{ x_min: {mb_double(x0)}, y_min: {mb_double(y0)}, x_max: {mb_double(x1)}, y_max: {mb_double(y1)}, label: {json.dumps(label)} }}'


def mb_sample(sample):
    boxes = "[" + ", ".join(mb_box(box) for box in sample["boxes"]) + "]"
    return f'@moonaug.Sample::{{ image: @moonaug.Image::{{ width: {sample["width"]}, height: {sample["height"]}, channels: {sample["channels"]}, data: {mb_array(sample["data"], "byte")} }}, mask: Some({mb_array(sample["mask"], "int")}), boxes: {boxes} }}'


def emit_aug_tests(reference):
    lines = ["// Generated by verification/generate_references.py. Do not edit by hand.",
             "// NumPy pixel permutations / PyTorch half-pixel resize / continuous box corners.",
             "fn reference_sample(got : @moonaug.Sample, expected : @moonaug.Sample, pixel_tolerance : Int) -> Unit raise {",
             "  assert_eq(got.image.width, expected.image.width)",
             "  assert_eq(got.image.height, expected.image.height)",
             "  assert_eq(got.image.channels, expected.image.channels)",
             "  assert_eq(got.image.data.length(), expected.image.data.length())",
             "  for i = 0; i < got.image.data.length(); i = i + 1 {",
             "    let delta = got.image.data[i].to_int() - expected.image.data[i].to_int()",
             "    assert_true(delta >= -pixel_tolerance && delta <= pixel_tolerance)",
             "    if got.image.channels == 4 && i % 4 == 3 { assert_eq(got.image.data[i], expected.image.data[i]) }", "  }",
             "  assert_eq(got.mask, expected.mask)",
             "  assert_eq(got.boxes.length(), expected.boxes.length())",
             "  for i = 0; i < got.boxes.length(); i = i + 1 {",
             "    let a = got.boxes[i]", "    let b = expected.boxes[i]",
             "    assert_eq(a.label, b.label)",
             "    let delta = [a.x_min-b.x_min, a.y_min-b.y_min, a.x_max-b.x_max, a.y_max-b.y_max]",
             "    for d in delta { assert_true(d >= -1.0e-10 && d <= 1.0e-10) }", "  }", "}"]
    for case in reference["cases"]:
        steps = []
        tolerance = int(any(op[0] in ("Resize", "Brightness", "Contrast") for op in case["operations"]))
        for kind, *args in case["operations"]:
            argstr = ", ".join(mb_double(arg) if kind in ("Brightness", "Contrast") else str(arg) for arg in args)
            op = f'@moonaug.{kind}({argstr})' if args else f'@moonaug.{kind}'
            steps.append(f'@moonaug.Step::{{ transform: {op}, probability: 1.0 }}')
        lines += [f'test "independent geometry reference: {case["name"]}" {{',
                  f'  let input = {mb_sample(case["input"])}',
                  f'  let expected = {mb_sample(case["expected"])}',
                  f'  let actual = @moonaug.augment(input, [{", ".join(steps)}], {SEED}, "reference", min_visibility={mb_double(case["min_visibility"])})',
                  f'  reference_sample(actual.sample, expected, {tolerance})',
                  f'  assert_eq(input, {mb_sample(case["input"])})',
                  "  let saved = @moonaug.replay_to_json(actual.replay)",
                  "  let restored = @moonaug.replay_from_json(saved)",
                  "  assert_eq(@moonaug.replay(input, restored), actual.sample)", "}"]
    # Randomness is tested through contracts, not through a copied PRNG algorithm.
    sample = mb_sample(reference["cases"][0]["input"])
    lines += ['test "random pipeline: deterministic by seed and sample ID, replay exact" {',
              f'  let input = {sample}',
              '  let pipeline = [@moonaug.Step::{ transform: @moonaug.RandomCrop(3, 2), probability: 1.0 }, @moonaug.Step::{ transform: @moonaug.HFlip, probability: 0.5 }, @moonaug.Step::{ transform: @moonaug.VFlip, probability: 0.0 }]',
              f'  let a = @moonaug.augment(input, pipeline, {SEED}, "random-case")',
              f'  let b = @moonaug.augment(input, pipeline, {SEED}, "random-case")',
              '  assert_eq(a.sample.image.data, b.sample.image.data)',
              '  assert_eq(@moonaug.replay_to_json(a.replay), @moonaug.replay_to_json(b.replay))',
              '  let restored = @moonaug.replay_from_json(@moonaug.replay_to_json(a.replay))',
              '  let again = @moonaug.replay(input, restored)',
              '  assert_eq(again.image.data, a.sample.image.data)',
              '  assert_eq(again, a.sample)', '}',
              'test "zero-probability steps preserve the independent input reference" {',
              f'  let input = {sample}',
              '  let pipeline = [@moonaug.Step::{ transform: @moonaug.RandomCrop(1, 1), probability: 0.0 }, @moonaug.Step::{ transform: @moonaug.HFlip, probability: 0.0 }, @moonaug.Step::{ transform: @moonaug.Rotate90(1), probability: 0.0 }]',
              f'  let actual = @moonaug.augment(input, pipeline, {SEED}, "skip-case")',
              f'  assert_eq(actual.sample, {sample})',
              '  assert_eq(actual.replay.operations.length(), 0)', '}']
    return "\n".join(lines) + "\n"


def assert_numerically_equal(regenerated, committed, kind, path=(), document=None):
    """Compare platform-independent structure and bounded floating-point drift."""
    if document is None:
        document = regenerated
    location = ".".join(map(str, path)) or "root"
    if isinstance(regenerated, dict):
        if not isinstance(committed, dict) or regenerated.keys() != committed.keys():
            raise SystemExit(f"Reference structure differs at {location}")
        if path == ("metadata",):
            if regenerated != committed:
                raise SystemExit("Reference dependency versions or metadata differ; use the recorded environment or explicitly regenerate.")
            return
        for key in regenerated:
            assert_numerically_equal(regenerated[key], committed[key], kind, (*path, key), document)
        return
    if isinstance(regenerated, list):
        if not isinstance(committed, list) or len(regenerated) != len(committed):
            raise SystemExit(f"Reference array shape differs at {location}")
        for index, (new, old) in enumerate(zip(regenerated, committed)):
            assert_numerically_equal(new, old, kind, (*path, index), document)
        return
    if type(regenerated) is int:
        tolerance = 0
        if kind == "augmentation" and len(path) == 5 and path[0] == "cases" and path[2:4] == ("expected", "data"):
            case = document["cases"][path[1]]
            operations = {operation[0] for operation in case["operations"]}
            if operations & {"Resize", "Brightness", "Contrast"}:
                tolerance = 1
            if case["expected"]["channels"] == 4 and path[4] % 4 == 3 and "Resize" not in operations:
                tolerance = 0
        if type(committed) is not int or abs(regenerated - committed) > tolerance:
            raise SystemExit(f"Reference integer or categorical value differs at {location}")
        return
    if type(regenerated) is float:
        tolerance = 2e-8 if "finite_difference" in path else 1e-10 if kind == "augmentation" and "boxes" in path else 1e-9
        if type(committed) not in (int, float) or not math.isfinite(committed) or not math.isfinite(regenerated):
            raise SystemExit(f"Reference contains an invalid numeric value at {location}")
        if abs(regenerated - committed) > tolerance:
            raise SystemExit(f"Reference number differs at {location}: regenerated={regenerated!r}, committed={committed!r}, tolerance={tolerance}")
        return
    if type(regenerated) is not type(committed) or regenerated != committed:
        raise SystemExit(f"Reference value differs at {location}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Verify exact committed bytes, including floating-point literals; use the same OS/environment")
    modes.add_argument("--check-numerical", action="store_true", help="Cross-OS check: exact metadata/structure, bounded numerical differences, generated tests bound to committed JSON")
    args = parser.parse_args()
    warp, aug = warp_references(), aug_references()
    metadata = {
        "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "device": "cpu", "dtype": "float64",
        "warp_reference": "torch.nn.functional.grid_sample / affine_grid + torch.autograd",
        "augmentation_reference": "NumPy permutations, continuous-corner affine box geometry, torch.interpolate bilinear align_corners=False",
        "coordinate_contract": "image HWC uint8; mask HW categorical int; boxes continuous half-open xyxy; grid NHW2 normalized output-to-input; tensor NCHW",
        "pixel_tolerance": 1, "double_tolerance": 1e-9,
        "limitations": "No antialias filtering; zero padding only; PyTorch gradients at non-smooth boundary points are convention checks, finite differences only at smooth samples.",
    }
    formatter = shutil.which("moonfmt")
    if formatter is None:
        raise SystemExit("moonfmt is required to generate canonical MoonBit reference tests; install the MoonBit toolchain.")
    def canonical_moonbit(source):
        # moonfmt's first pass can insert ///|, then a second pass separates that
        # marker from file-header comments. Reach a fixed point before comparing
        # artifacts, so a later project-wide `moon fmt` cannot change the bytes.
        for _ in range(4):
            result = subprocess.run([formatter, "-"], input=source, capture_output=True, text=True, encoding="utf-8", check=True)
            if result.stdout == source:
                return source
            source = result.stdout
        raise SystemExit("moonfmt did not produce stable reference source after four passes.")
    warp_document = {"metadata": metadata, **warp}
    aug_document = {"metadata": metadata, **aug}
    if args.check_numerical:
        # Check independent fresh computations numerically, then regenerate tests
        # from COMMITTED JSON. This avoids platform-dependent literal changes
        # while proving the checked-in tests still encode the checked-in oracle.
        stored_warp = json.loads((ROOT / "verification" / "warp_reference.json").read_text(encoding="utf-8"))
        stored_aug = json.loads((ROOT / "verification" / "augmentation_reference.json").read_text(encoding="utf-8"))
        assert_numerically_equal(warp_document, stored_warp, "warp")
        assert_numerically_equal(aug_document, stored_aug, "augmentation")
        warp_tests, aug_tests = stored_warp, stored_aug
    else:
        warp_tests, aug_tests = warp, aug
    artifacts = {
        ROOT / "verification" / "warp_reference.json": json.dumps(warp_document, indent=2, allow_nan=False) + "\n",
        ROOT / "verification" / "augmentation_reference.json": json.dumps(aug_document, indent=2, allow_nan=False) + "\n",
        ROOT / "moonwarp" / "reference_pytorch_test.mbt": canonical_moonbit(emit_warp_tests(warp_tests)),
        ROOT / "moonaug" / "reference_geometry_test.mbt": canonical_moonbit(emit_aug_tests(aug_tests)),
    }
    for path, content in artifacts.items():
        if args.check_numerical and path.suffix == ".json":
            # Already checked every structural and numerical field above.
            content = path.read_text(encoding="utf-8")
        elif args.check or args.check_numerical:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Reference artifact differs: {path.relative_to(ROOT)}; regenerate with pinned dependencies.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        mode = "checked numerically" if args.check_numerical else "checked" if args.check else "generated"
        print(f'{mode} {path.relative_to(ROOT)} sha256={hashlib.sha256(content.encode()).hexdigest()}')
    print(f'Finite-difference audits passed. {len(warp["sample_cases"])} sampler + {len(warp["affine_cases"])} affine + {len(aug["cases"])} augmentation reference cases.')


if __name__ == "__main__":
    main()
