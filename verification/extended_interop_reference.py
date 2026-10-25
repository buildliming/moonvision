"""Cross-representation oracle: PyTorch rasters and NumPy forward edge geometry."""

from __future__ import annotations

import copy
import platform

import numpy as np
import torch
import torch.nn.functional as F

from extended_augmentation_reference import points_source
from generate_references import DTYPE, SEED, assert_numerically_equal, mb_array, mb_double, mb_sample, reference_affine_grid
from generate_extended_references import homogeneous, normalizer


def interop_document():
    image = ((np.arange(3*4*3).reshape(3, 4, 3)*7+5) % 256).astype(np.uint8)
    mask = np.array([[0, 1, 4, 2147483647], [2, 7, 4, 1], [2, 2, 1024, 4]], dtype=np.int64)
    boxes = [[.25, .5, 3.6, 2.6, "main"], [0., 0., 1., 1., "corner"], [3., 2., 4., 3., "edge"]]
    points = [[0., 0., "origin", True], [4., 3., "far-edge", True], [1., 2., "hidden", False], [3.7, .2, "inside", True]]
    source = {"width": 4, "height": 3, "channels": 3, "data": image.reshape(-1).tolist(), "mask": mask.reshape(-1).tolist(), "boxes": boxes}
    image_tensor = torch.tensor(image.copy(), dtype=DTYPE).permute(2, 0, 1).unsqueeze(0)/255.
    mask_tensor = torch.tensor(mask.copy(), dtype=DTYPE).reshape(1, 1, 3, 4)
    cases = []
    for align in (False, True):
        for name, theta, height, width in [
            ("identity_different_output_size", [1., 0., 0., 0., 1., 0.], 5, 6),
            ("reflection", [-1., 0., 0., 0., 1., 0.], 3, 4),
            ("rotation_scale_translation", [.82, -.23, .27, .17, .91, -.19], 4, 5),
            ("partly_outside_mask_fill", [.9, .1, 1.2, -.1, 1., .4], 5, 3),
        ]:
            grid = reference_affine_grid(torch.tensor(theta, dtype=DTYPE).reshape(1, 2, 3), (1, 3, height, width), align)
            raster = F.grid_sample(image_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=align)
            image_out = np.floor(raster.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)*255.+.5).astype(np.uint8)
            mask_out = F.grid_sample(mask_tensor, grid, mode="nearest", padding_mode="zeros", align_corners=align).numpy().reshape(height, width)
            coverage = F.grid_sample(torch.ones_like(mask_tensor), grid, mode="nearest", padding_mode="zeros", align_corners=align).numpy().reshape(height, width)
            mask_out = np.where(coverage > .5, mask_out, 987654).astype(np.int64)
            # Convert the backward normalized map to a FORWARD edge-coordinate
            # map with homogeneous basis changes. No MoonBit matrix helper runs.
            pixel_back = np.linalg.inv(normalizer(3, 4, align)) @ homogeneous(theta) @ normalizer(height, width, align)
            to_center = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])
            to_edge = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
            forward = to_edge @ np.linalg.inv(pixel_back) @ to_center
            # Exact geometric invariants avoid using a numerically perturbed
            # matrix to decide the Boolean visibility of a boundary point.
            if name == "reflection":
                forward = np.array([[-1., 0., 4.], [0., 1., 0.], [0., 0., 1.]])
            elif name == "identity_different_output_size" and not align:
                forward = np.diag([width/4, height/3, 1.])
            mapped_boxes = []
            for x0, y0, x1, y1, label in boxes:
                corners = np.array([[x0, y0, 1.], [x1, y0, 1.], [x0, y1, 1.], [x1, y1, 1.]]) @ forward.T
                low, high = corners[:, :2].min(0), corners[:, :2].max(0)
                clipped_low, clipped_high = np.maximum(low, [0., 0.]), np.minimum(high, [width, height])
                full_size, retained_size = high-low, clipped_high-clipped_low
                if np.all(retained_size > 0) and np.prod(retained_size/full_size) >= .2:
                    mapped_boxes.append([*clipped_low.tolist(), *clipped_high.tolist(), label])
            mapped_points = []
            edge_tolerance = 1e-12 * max(width, height)
            for x, y, label, visible in points:
                mapped = forward @ [x, y, 1.]
                if visible:
                    # Closed-boundary geometry: inversion roundoff just outside
                    # an exact edge must not hide a previously visible point.
                    for axis, limit in enumerate((width, height)):
                        if -edge_tolerance <= mapped[axis] <= 0.:
                            mapped[axis] = 0.
                        elif limit <= mapped[axis] <= limit + edge_tolerance:
                            mapped[axis] = float(limit)
                mapped_points.append([float(mapped[0]), float(mapped[1]), label, bool(visible and 0 <= mapped[0] <= width and 0 <= mapped[1] <= height)])
            cases.append({
                "name": f"{name}_{str(align).lower()}", "input": source, "points": points, "theta": theta,
                "output_height": height, "output_width": width, "align_corners": align,
                "min_visibility": .2, "mask_fill": 987654,
                "expected": {"width": width, "height": height, "channels": 3, "data": image_out.reshape(-1).tolist(), "mask": mask_out.reshape(-1).tolist(), "boxes": mapped_boxes},
                "expected_points": mapped_points,
            })
    second = np.flip(image, (0, 1)).copy()
    packed = np.stack([image, second]).transpose(0, 3, 1, 2)/255.
    mean, std = np.array([.2, .4, .6]), np.array([.3, .5, .7])
    normalized = (packed-mean[None, :, None, None])/std[None, :, None, None]
    return {
        "metadata": {"seed": SEED+1, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "oracle": "torch grid_sample bilinear image/nearest mask; NumPy homogeneous inverse edge matrix and NCHW normalization", "dtype": "float64", "device": "cpu", "pixel_tolerance": 1, "double_tolerance": 1e-9, "keypoint_boundary_tolerance": "1e-12 * max(output width, output height), previously visible points only"},
        "cases": cases,
        "batch": {"shape": list(packed.shape), "images": [source, {**source, "data": second.reshape(-1).tolist()}], "mean": mean.tolist(), "std": std.tolist(), "packed": packed.reshape(-1).tolist(), "normalized": normalized.reshape(-1).tolist()},
    }


def check_interop_numerically(fresh, stored):
    checked = copy.deepcopy(stored)
    if len(fresh["cases"]) != len(stored["cases"]):
        raise SystemExit("Interop reference case count changed")
    for new, old in zip(fresh["cases"], checked["cases"]):
        new_pixels, old_pixels = new["expected"]["data"], old["expected"]["data"]
        if len(new_pixels) != len(old_pixels) or any(type(value) is not int for value in old_pixels):
            raise SystemExit("Interop reference raster structure changed")
        if any(abs(a-b) > 1 for a, b in zip(new_pixels, old_pixels)):
            raise SystemExit("Interop reference raster differs by more than one uint8 value")
        old["expected"]["data"] = new_pixels
    assert_numerically_equal(fresh, checked, "warp")


def emit_interop(document):
    lines = ["// Generated from independent PyTorch rasters and NumPy edge/batch geometry.", "",
             "fn interop_reference_close(actual : Array[Double], expected : Array[Double]) -> Unit raise {",
             "  assert_eq(actual.length(), expected.length())",
             "  for i = 0; i < actual.length(); i = i + 1 { let d = actual[i]-expected[i]; assert_true(d >= -1.0e-9 && d <= 1.0e-9) }", "}",
             "fn interop_reference_sample(actual : @moonaug.Sample, expected : @moonaug.Sample) -> Unit raise {",
             "  assert_eq((actual.image.width, actual.image.height, actual.image.channels), (expected.image.width, expected.image.height, expected.image.channels))",
             "  assert_eq(actual.image.data.length(), expected.image.data.length())",
             "  for i = 0; i < actual.image.data.length(); i = i + 1 { let d = actual.image.data[i].to_int()-expected.image.data[i].to_int(); assert_true(d >= -1 && d <= 1) }",
             "  assert_eq(actual.mask, expected.mask)", "  assert_eq(actual.boxes.length(), expected.boxes.length())",
             "  for i = 0; i < actual.boxes.length(); i = i + 1 {",
             "    let a = actual.boxes[i]", "    let b = expected.boxes[i]",
             "    assert_true(a.x_min < a.x_max && a.y_min < a.y_max)", "    assert_eq(a.label, b.label)",
             "    interop_reference_close([a.x_min,a.y_min,a.x_max,a.y_max], [b.x_min,b.y_min,b.x_max,b.y_max])", "  }", "}"]
    for case in document["cases"]:
        input_source = f'@moonaug.ExtendedSample::{{ sample: {mb_sample(case["input"])}, keypoints: {points_source(case["points"])} }}'
        options = f'align_corners={str(case["align_corners"]).lower()}, min_visibility={mb_double(case["min_visibility"])}, mask_fill={case["mask_fill"]}'
        lines += [f'test "independent interop warp: {case["name"]}" {{',
                  f'  let input = {input_source}',
                  f'  let theta = {mb_array(case["theta"])}',
                  f'  let result = @interop.warp_extended(input, theta, {case["output_height"]}, {case["output_width"]}, {options})',
                  f'  interop_reference_sample(result.sample, {mb_sample(case["expected"])})',
                  f'  let expected_points = {points_source(case["expected_points"])}',
                  '  assert_eq(result.keypoints.length(), expected_points.length())',
                  '  for i = 0; i < result.keypoints.length(); i = i + 1 {',
                  '    let a = result.keypoints[i]', '    let b = expected_points[i]',
                  '    assert_eq((a.label, a.visible), (b.label, b.visible))',
                  '    interop_reference_close([a.x, a.y], [b.x, b.y])', '  }',
                  f'  assert_eq(@interop.warp_sample(input.sample, theta, {case["output_height"]}, {case["output_width"]}, {options}), result.sample)',
                  f'  assert_eq(input, {input_source})', '}']
    batch = document["batch"]
    # Strip annotations: images_to_tensor takes image values only by design.
    images = "[" + ", ".join(f'({mb_sample(image)}).image' for image in batch["images"]) + "]"
    lines += ['test "independent NumPy NCHW batch normalization and exact image round trip" {',
              f'  let images = {images}',
              '  let packed = @interop.images_to_tensor(images)',
              f'  assert_eq(packed.shape(), {mb_array(batch["shape"], "int")})',
              f'  interop_reference_close(packed.data(), {mb_array(batch["packed"])})',
              f'  let mean = {mb_array(batch["mean"])}', f'  let std = {mb_array(batch["std"])}',
              '  let normalized = @interop.normalize(packed, mean, std)',
              f'  interop_reference_close(normalized.data(), {mb_array(batch["normalized"])})',
              '  interop_reference_close(@interop.denormalize(normalized, mean, std).data(), packed.data())',
              '  assert_eq(@interop.tensor_to_images(packed), images)',
              f'  interop_reference_close(packed.data(), {mb_array(batch["packed"])})', '}']
    mask = document["cases"][0]["input"]["mask"]
    lines += ['test "categorical mask conversion preserves large IDs and batch offsets exactly" {',
              f'  let mask = {mb_array(mask, "int")}',
              '  let packed = @interop.mask_to_tensor(mask, 4, 3)',
              '  assert_eq(packed.shape(), [1, 1, 3, 4])',
              f'  interop_reference_close(packed.data(), {mb_array(mask)})',
              '  assert_eq(@interop.tensor_to_mask(packed), mask)',
              f'  let batch = @moonwarp.Tensor::new([2, 1, 3, 4], {mb_array(list(reversed(mask))+mask)})',
              '  assert_eq(@interop.tensor_to_mask(batch, batch_index=1), mask)', '}']
    return "\n".join(lines) + "\n"
