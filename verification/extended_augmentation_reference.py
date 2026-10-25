"""NumPy/PyTorch augmentation geometry oracle, independent of MoonBit/its PRNG."""

from __future__ import annotations

import copy
import json
import platform

import numpy as np
import torch
import torch.nn.functional as F

from generate_references import DTYPE, SEED, assert_numerically_equal, mb_array, mb_double, mb_sample


def resize(image, mask, width, height):
    source = torch.tensor(image.copy(), dtype=DTYPE).permute(2, 0, 1).unsqueeze(0)
    result = F.interpolate(source, size=(height, width), mode="bilinear", align_corners=False, antialias=False)
    result = np.floor(result.squeeze(0).permute(1, 2, 0).numpy() + .5).clip(0, 255).astype(np.uint8)
    yy = np.minimum(np.floor((np.arange(height) + .5) * image.shape[0] / height).astype(int), image.shape[0]-1)
    xx = np.minimum(np.floor((np.arange(width) + .5) * image.shape[1] / width).astype(int), image.shape[1]-1)
    return result, mask[np.ix_(yy, xx)]


def transform_boxes(boxes, matrix, width, height, visibility):
    output = []
    for x0, y0, x1, y1, label in boxes:
        corners = np.array([[x0, y0, 1.], [x1, y0, 1.], [x0, y1, 1.], [x1, y1, 1.]]) @ matrix.T
        low, high = corners.min(0), corners.max(0)
        clipped_low, clipped_high = np.maximum(low, [0., 0.]), np.minimum(high, [width, height])
        full_size, clipped_size = high-low, clipped_high-clipped_low
        if np.any(clipped_size <= 0.):
            continue
        # Axis fractions avoid underflow from multiplying two tiny positive sides.
        retained = np.prod(clipped_size/full_size)
        if retained >= visibility:
            output.append([*clipped_low.tolist(), *clipped_high.tolist(), label])
    return output


def apply_operations(source, operations, visibility=0.):
    width, height, channels = source["width"], source["height"], source["channels"]
    image = np.asarray(source["data"], dtype=np.uint8).reshape(height, width, channels)
    mask = np.asarray(source["mask"], dtype=np.int64).reshape(height, width)
    boxes = source["boxes"]
    for name, *args in operations:
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        old_height, old_width = image.shape[:2]
        if name == "Pad":
            left, top, right, bottom, fill, mask_value = args
            height, width = old_height+top+bottom, old_width+left+right
            result = np.broadcast_to(np.asarray(fill, dtype=np.uint8), (height, width, channels)).copy()
            result[top:top+old_height, left:left+old_width] = image
            result_mask = np.full((height, width), mask_value, dtype=np.int64)
            result_mask[top:top+old_height, left:left+old_width] = mask
            image, mask = result, result_mask
            matrix[:, 2] = [left, top]
        elif name == "Crop":
            x, y, width, height = args
            image, mask = image[y:y+height, x:x+width], mask[y:y+height, x:x+width]
            matrix[:, 2] = [-x, -y]
        elif name == "Resize":
            width, height = args
            image, mask = resize(image, mask, width, height)
            matrix = np.array([[width/old_width, 0., 0.], [0., height/old_height, 0.]])
        else:
            raise ValueError(name)
        height, width = image.shape[:2]
        boxes = transform_boxes(boxes, matrix, width, height, visibility)
    return {"width": width, "height": height, "channels": channels, "data": image.reshape(-1).tolist(), "mask": mask.reshape(-1).tolist(), "boxes": boxes}


def letterbox_operations(source, width, height, fill, mask_value, scale_up):
    scale = min(width/source["width"], height/source["height"])
    if not scale_up:
        scale = min(scale, 1.)
    resized_width = max(1, int(np.floor(source["width"] * scale + .5)))
    resized_height = max(1, int(np.floor(source["height"] * scale + .5)))
    dx, dy = width-resized_width, height-resized_height
    return [["Resize", resized_width, resized_height], ["Pad", dx//2, dy//2, dx-dx//2, dy-dy//2, fill, mask_value]]


def aug_document():
    image = ((np.arange(3*4*3).reshape(3, 4, 3) * 7 + 5) % 256).astype(np.uint8)
    source = {
        "width": 4, "height": 3, "channels": 3,
        "data": image.reshape(-1).tolist(),
        "mask": [0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4],
        "boxes": [[.25, .5, 3.6, 2.6, "main"], [0., 0., 1., 1., "corner"], [3., 2., 4., 3., "edge"]],
    }
    cases = []
    for name, parameters in [
        ("asymmetric_per_channel_pad", [1, 2, 2, 0, [7, 29, 131], 1000]),
        ("zero_extent_pad", [0, 0, 0, 0, [0, 0, 0], 9]),
        ("right_bottom_pad", [0, 0, 1, 2, [255, 11, 0], 2147483647]),
    ]:
        operations = [["Pad", *parameters]]
        cases.append({"name": name, "input": source, "pipeline": operations, "operations": operations, "expected": apply_operations(source, operations), "min_visibility": 0.})
    for width, height, scale_up in [(7, 7, True), (3, 7, True), (9, 7, True), (9, 9, False), (1, 7, True)]:
        parameters = [width, height, [23, 47, 91], 1024, scale_up]
        operations = letterbox_operations(source, *parameters)
        cases.append({"name": f"letterbox_{width}x{height}_{scale_up}", "input": source, "pipeline": [["Letterbox", *parameters]], "operations": operations, "expected": apply_operations(source, operations), "min_visibility": 0.})
    tiny = {"width": 2, "height": 1, "channels": 1, "data": [10, 20], "mask": [1, 2], "boxes": [[0., 0., 1e-200, 1e-200, "tiny"]]}
    operations = [["Crop", 0, 0, 2, 1]]
    cases.append({"name": "positive_box_area_must_not_underflow", "input": tiny, "pipeline": operations, "operations": operations, "expected": tiny, "min_visibility": 0.})
    # Exhaustive possible 4x3 crop windows form a PRNG-independent oracle table.
    # Runtime RandomResizedCrop chooses a window; only geometry is looked up.
    for width in range(1, 5):
        for height in range(1, 4):
            for x in range(5-width):
                for y in range(4-height):
                    operations = [["Crop", x, y, width, height], ["Resize", 2, 2]]
                    cases.append({"name": f"rrc_window_{x}_{y}_{width}_{height}", "input": source, "pipeline": operations, "operations": operations, "expected": apply_operations(source, operations), "min_visibility": 0., "rrc_window": [x, y, width, height]})
    point_input = [[0., 0., "origin", True], [4., 3., "edge", True], [1., 2., "already-hidden", False], [3.7, .2, "inside", True], [2., 1., "center", True], [.5, 2.5, "cropped-out", True]]
    point_pipeline = [["Crop", 1, 0, 3, 2], ["Pad", 1, 2, 0, 1, [7, 29, 131], 1000], ["Letterbox", 7, 7, [23, 47, 91], 1024, True]]
    point_operations = point_pipeline[:2] + [["Resize", 6, 7], ["Pad", 0, 0, 1, 0, [23, 47, 91], 1024]]
    expected_points = []
    for x, y, label, visible in point_input:
        # Explicit intermediate rectangles make the visibility history independent
        # of the final image bounds: padding must not resurrect cropped-out points.
        point = np.array([x, y, 1.])
        for matrix, width, height in [
            (np.array([[1., 0., -1.], [0., 1., 0.]]), 3, 2),
            (np.array([[1., 0., 1.], [0., 1., 2.]]), 4, 5),
            (np.array([[1.5, 0., 0.], [0., 1.4, 0.]]), 6, 7),
            (np.array([[1., 0., 0.], [0., 1., 0.]]), 7, 7),
        ]:
            xy = matrix @ point
            visible = bool(visible and 0 <= xy[0] <= width and 0 <= xy[1] <= height)
            point = np.array([xy[0], xy[1], 1.])
        expected_points.append([float(point[0]), float(point[1]), label, visible])
    return {
        "metadata": {"seed": SEED + 1, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "dtype": "float64", "oracle": "NumPy padding/slicing/corner geometry; torch.interpolate bilinear align_corners=False; exhaustive crop-window table; no MoonBit PRNG copied", "pixel_tolerance": 1, "box_tolerance": 1e-10},
        "cases": cases,
        "keypoint_cases": [{"name": "crop_pad_letterbox_preserves_visibility_history", "input": source, "points": point_input, "pipeline": point_pipeline, "operations": point_operations, "expected": apply_operations(source, point_operations), "expected_points": expected_points}],
    }


def operation_source(operation):
    name, *args = operation
    if name == "Pad":
        left, top, right, bottom, fill, mask_value = args
        return f'@moonaug.Pad(@moonaug.Padding::{{ left: {left}, top: {top}, right: {right}, bottom: {bottom}, fill: {mb_array(fill, "byte")}, mask_value: {mask_value} }})'
    if name == "Letterbox":
        width, height, fill, mask_value, scale_up = args
        return f'@moonaug.Letterbox(@moonaug.LetterboxOptions::{{ width: {width}, height: {height}, fill: {mb_array(fill, "byte")}, mask_value: {mask_value}, scale_up: {str(scale_up).lower()} }})'
    return f'@moonaug.{name}({", ".join(map(str, args))})'


def points_source(points):
    return "[" + ", ".join(f'@moonaug.Keypoint::{{ x: {mb_double(x)}, y: {mb_double(y)}, label: {json.dumps(label)}, visible: {str(visible).lower()} }}' for x, y, label, visible in points) + "]"


def quantized_pixel_mask(case):
    """Track only bytes that can acquire cross-CPU uint8 rounding differences."""
    source = case["input"]
    mask = np.zeros((source["height"], source["width"], source["channels"]), dtype=bool)
    quantizations = 0
    for name, *args in case["operations"]:
        height, width, channels = mask.shape
        if name == "Resize":
            w, h = args
            if (w, h) != (width, height):
                mask = np.ones((h, w, channels), dtype=bool)
                quantizations += 1
        elif name == "Pad":
            left, top, right, bottom, _, _ = args
            padded = np.zeros((height+top+bottom, width+left+right, channels), dtype=bool)
            padded[top:top+height, left:left+width] = mask
            mask = padded
        elif name == "Crop":
            x, y, w, h = args
            mask = mask[y:y+h, x:x+w]
        elif name == "HFlip":
            mask = np.flip(mask, 1)
        elif name == "VFlip":
            mask = np.flip(mask, 0)
        elif name == "Rotate90":
            mask = np.rot90(mask, -args[0])
        elif name in ("Brightness", "Contrast"):
            mask = mask.copy()
            mask[:, :, :min(channels, 3)] = True
            quantizations += 1
        else:
            raise SystemExit(f"Unreviewed resolved operation in numerical raster check: {name}")
    if quantizations > 1:
        raise SystemExit("Multiple uint8 quantization stages require explicit intermediate references; do not accumulate a blanket pixel tolerance")
    expected = case["expected"]
    if mask.shape != (expected["height"], expected["width"], expected["channels"]):
        raise SystemExit("Numerical raster provenance shape differs from expected image")
    return mask.reshape(-1)


def check_aug_numerically(fresh, stored):
    """Apply declared byte tolerance to both raster-bearing case families only."""
    checked = copy.deepcopy(stored)
    for family in ("cases", "keypoint_cases"):
        if len(fresh[family]) != len(checked[family]):
            raise SystemExit(f"Augmentation reference case count changed: {family}")
        for index, (new, old) in enumerate(zip(fresh[family], checked[family])):
            allowed = quantized_pixel_mask(new)
            new_pixels, old_pixels = new["expected"]["data"], old["expected"]["data"]
            if len(new_pixels) != len(old_pixels) or len(new_pixels) != len(allowed):
                raise SystemExit(f"Augmentation raster shape differs at {family}.{index}")
            for pixel, (a, b, rounded) in enumerate(zip(new_pixels, old_pixels, allowed)):
                if type(a) is not int or type(b) is not int or not (0 <= a <= 255 and 0 <= b <= 255) or abs(a-b) > int(rounded):
                    raise SystemExit(f"Augmentation uint8 value differs at {family}.{index}.expected.data.{pixel}")
            # Only the already-validated raster bytes are normalized for the
            # common recursive comparator. Masks, labels, shapes, input bytes,
            # fill values, point visibility and all metadata remain untouched.
            old["expected"]["data"] = new_pixels
    assert_numerically_equal(fresh, checked, "augmentation")


def emit_aug(document):
    fixed = [case for case in document["cases"] if "rrc_window" not in case]
    windows = [case for case in document["cases"] if "rrc_window" in case]
    lines = ["// Generated from independent NumPy/PyTorch geometry references.", "",
             "fn extended_sample_close(actual : @moonaug.Sample, expected : @moonaug.Sample, tolerance : Int) -> Unit raise {",
             "  assert_eq((actual.image.width, actual.image.height, actual.image.channels), (expected.image.width, expected.image.height, expected.image.channels))",
             "  assert_eq(actual.image.data.length(), expected.image.data.length())",
             "  for i = 0; i < actual.image.data.length(); i = i + 1 {",
             "    let delta = actual.image.data[i].to_int() - expected.image.data[i].to_int()",
             "    assert_true(delta >= -tolerance && delta <= tolerance)", "  }",
             "  assert_eq(actual.mask, expected.mask)",
             "  assert_eq(actual.boxes.length(), expected.boxes.length())",
             "  for i = 0; i < actual.boxes.length(); i = i + 1 {",
             "    let a = actual.boxes[i]", "    let b = expected.boxes[i]",
             "    assert_true(a.x_min < a.x_max && a.y_min < a.y_max)",
             "    assert_eq(a.label, b.label)",
             "    for delta in [a.x_min-b.x_min, a.y_min-b.y_min, a.x_max-b.x_max, a.y_max-b.y_max] { assert_true(delta >= -1.0e-10 && delta <= 1.0e-10) }", "  }", "}"]
    for case in fixed:
        steps = ", ".join(f'@moonaug.Step::{{ transform: {operation_source(op)}, probability: 1.0 }}' for op in case["pipeline"])
        tolerance = int(any(op[0] == "Resize" for op in case["operations"]))
        lines += [f'test "extended independent geometry: {case["name"]}" {{',
                  f'  let input = {mb_sample(case["input"])}',
                  f'  let result = @moonaug.augment(input, [{steps}], {SEED+1}, "extended-reference")',
                  f'  extended_sample_close(result.sample, {mb_sample(case["expected"])}, {tolerance})',
                  f'  assert_eq(input, {mb_sample(case["input"])})',
                  '  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))',
                  '  assert_eq(@moonaug.replay(input, record), result.sample)', '}']
    lines += ["fn extended_crop_expected(x : Int, y : Int, width : Int, height : Int) -> @moonaug.Sample raise {", "  match (x, y, width, height) {"]
    for case in windows:
        key = ", ".join(map(str, case["rrc_window"]))
        lines.append(f'    ({key}) => {mb_sample(case["expected"])}')
    lines += ['    _ => fail("resolved random crop lies outside the independent oracle table")', '  }', '}']
    lines += ['test "random resized crop uses independent exhaustive window oracle and exact replay" {',
              f'  let input = {mb_sample(windows[0]["input"])}',
              '  let transform = @moonaug.RandomResizedCrop(@moonaug.ResizedCropOptions::{ width: 2, height: 2, scale_min: 0.3, scale_max: 0.95, ratio_min: 0.5, ratio_max: 2.0 })',
              '  for seed = 0; seed < 24; seed = seed + 1 {',
              '    let result = @moonaug.augment(input, [{ transform, probability: 1.0 }], seed, "independent-rrc")',
              '    assert_eq(result, @moonaug.augment(input, [{ transform, probability: 1.0 }], seed, "independent-rrc"))',
              '    guard result.replay.operations is [@moonaug.Crop(x, y, width, height), @moonaug.Resize(2, 2)] else { fail("random resized crop must resolve to Crop followed by Resize") }',
              '    extended_sample_close(result.sample, extended_crop_expected(x, y, width, height), 1)',
              '    assert_eq(@moonaug.replay(input, result.replay), result.sample)',
              '  }', '}']
    lines += ['test "random resized crop impossible aspect ratio uses documented center fallback" {',
              f'  let input = {mb_sample(windows[0]["input"])}',
              '  let transform = @moonaug.RandomResizedCrop(@moonaug.ResizedCropOptions::{ width: 2, height: 2, scale_min: 1.0, scale_max: 1.0, ratio_min: 0.2, ratio_max: 0.2 })',
              f'  let result = @moonaug.augment(input, [@moonaug.Step::{{ transform, probability: 1.0 }}], {SEED+1}, "fallback")',
              '  assert_eq(result.replay.operations, [@moonaug.Crop(1, 0, 1, 3), @moonaug.Resize(2, 2)])',
              '  extended_sample_close(result.sample, extended_crop_expected(1, 0, 1, 3), 1)', '}']
    lines += ["fn extended_points_close(actual : Array[@moonaug.Keypoint], expected : Array[@moonaug.Keypoint]) -> Unit raise {",
              "  assert_eq(actual.length(), expected.length())",
              "  for i = 0; i < actual.length(); i = i + 1 {",
              "    assert_eq((actual[i].label, actual[i].visible), (expected[i].label, expected[i].visible))",
              "    let dx = actual[i].x - expected[i].x", "    let dy = actual[i].y - expected[i].y",
              "    assert_true(dx >= -1.0e-10 && dx <= 1.0e-10 && dy >= -1.0e-10 && dy <= 1.0e-10)", "  }", "}"]
    for case in document["keypoint_cases"]:
        input_source = f'@moonaug.ExtendedSample::{{ sample: {mb_sample(case["input"])}, keypoints: {points_source(case["points"])} }}'
        steps = ", ".join(f'@moonaug.Step::{{ transform: {operation_source(op)}, probability: 1.0 }}' for op in case["pipeline"])
        lines += [f'test "independent keypoint geometry: {case["name"]}" {{',
                  f'  let input = {input_source}',
                  f'  let result = @moonaug.augment_with_keypoints(input, [{steps}], {SEED+1}, "points")',
                  f'  extended_sample_close(result.sample.sample, {mb_sample(case["expected"])}, 1)',
                  f'  extended_points_close(result.sample.keypoints, {points_source(case["expected_points"])})',
                  '  assert_eq(result.replay.version, 2)',
                  '  let restored = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))',
                  '  assert_eq(@moonaug.replay_with_keypoints(input, restored), result.sample)',
                  f'  assert_eq(input, {input_source})', '}']
    # No expected random stream is invented: batching is required to equal the
    # public single-item API for each ID and remain invariant to input ordering.
    first = document["keypoint_cases"][0]
    input_source = f'@moonaug.ExtendedSample::{{ sample: {mb_sample(first["input"])}, keypoints: {points_source(first["points"])} }}'
    letterbox = operation_source(["Letterbox", 5, 4, [23, 47, 91], 1024, True])
    lines += ['test "batch random geometry is keyed by ID, replay records align by ID" {',
              f'  let input = {input_source}',
              '  let items = [@moonaug.BatchItem::{ sample_id: "alpha", input }, @moonaug.BatchItem::{ sample_id: "beta", input }]',
              f'  let pipeline = [@moonaug.Step::{{ transform: @moonaug.RandomCrop(3, 2), probability: 1.0 }}, @moonaug.Step::{{ transform: @moonaug.HFlip, probability: 0.45 }}, @moonaug.Step::{{ transform: {letterbox}, probability: 1.0 }}]',
              f'  let outputs = @moonaug.augment_batch(items, pipeline, {SEED+1})',
              '  assert_eq(outputs[0].sample_id, "alpha")', '  assert_eq(outputs[1].sample_id, "beta")',
              f'  assert_eq(outputs[0].output, @moonaug.augment_with_keypoints(input, pipeline, {SEED+1}, "alpha"))',
              f'  assert_eq(outputs[1].output, @moonaug.augment_with_keypoints(input, pipeline, {SEED+1}, "beta"))',
              f'  let reversed = @moonaug.augment_batch([items[1], items[0]], pipeline, {SEED+1})',
              '  assert_eq(reversed, [outputs[1], outputs[0]])',
              '  let records = [@moonaug.BatchReplay::{ sample_id: "alpha", replay: outputs[0].output.replay }, @moonaug.BatchReplay::{ sample_id: "beta", replay: outputs[1].output.replay }]',
              '  assert_eq(@moonaug.replay_batch([items[1], items[0]], records), reversed)', '}']
    return "\n".join(lines) + "\n"
