"""Exercise compiled MoonBit on attributed, offline photographs of diverse kinds.

The implementation under test is the compiled JavaScript JSON bridge. NumPy and
PyTorch independently compute expected pixels, categorical masks, geometry and
gradients. Region annotations are explicit geometric probes, not semantic labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "web" / "samples"
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class Bridge:
    def __init__(self, mode):
        self.process = subprocess.Popen(["node", str(ROOT / "tools/moonvision.mjs"), mode], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.lines = queue.Queue()
        self.calls = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def request(self, request, *, success=True):
        self.process.stdin.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=45)
        except queue.Empty:
            self.process.kill()
            raise AssertionError("compiled bridge timed out") from None
        if line is None:
            raise AssertionError("compiled bridge exited: " + self.process.stderr.read()[-2000:])
        result = json.loads(line)
        assert result.get("ok") is success, result
        self.calls += 1
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        error = self.process.stderr.read()
        assert self.process.returncode == 0, error


def chw(values):
    return torch.tensor(values.transpose(2, 0, 1)[None].copy(), dtype=torch.float64)


def tensor_json(tensor):
    return {"shape": list(tensor.shape), "data": tensor.detach().reshape(-1).tolist()}


def hwc(values, *, quantize=False):
    result = values.detach().numpy()[0].transpose(1, 2, 0)
    return np.floor(np.clip(result, 0, 255) + .5).astype(np.uint8) if quantize else result


def sample_for(image):
    height, width, channels = image.shape
    yy, xx = np.mgrid[:height, :width]
    mask = np.zeros((height, width), np.int32)
    mask[(xx >= width * .1) & (xx < width * .62) & (yy >= height * .12) & (yy < height * .76)] = 3
    mask[((xx / width - .75) ** 2 + (yy / height - .6) ** 2) < .13 ** 2] = 7
    boxes = [
        {"x_min": width * .1, "y_min": height * .12, "x_max": width * .62, "y_max": height * .76, "label": "probe_a"},
        {"x_min": width * .62, "y_min": height * .47, "x_max": width * .88, "y_max": height * .73, "label": "probe_b"},
    ]
    points = [{"x": width * .36, "y": height * .44, "label": "center", "visible": True}, {"x": width, "y": 0.0, "label": "edge", "visible": True}, {"x": width * .5, "y": height * .5, "label": "previously_hidden", "visible": False}]
    return {"image": {"width": width, "height": height, "channels": channels, "data": image.reshape(-1).tolist()}, "mask": mask.reshape(-1).tolist(), "boxes": boxes}, points


def transformed_targets(boxes, points, matrix, width, height, visibility):
    output_boxes = []
    for box in boxes:
        corners = np.array([[box["x_min"], box["y_min"], 1], [box["x_min"], box["y_max"], 1], [box["x_max"], box["y_min"], 1], [box["x_max"], box["y_max"], 1]], dtype=np.float64) @ matrix.T
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        clipped_lower = np.clip(lower, [0, 0], [width, height])
        clipped_upper = np.clip(upper, [0, 0], [width, height])
        span = upper - lower
        visible_span = clipped_upper - clipped_lower
        if np.all(visible_span > 0) and np.all(span > 0) and np.prod(visible_span / span) >= visibility:
            output_boxes.append({"x_min": float(clipped_lower[0]), "y_min": float(clipped_lower[1]), "x_max": float(clipped_upper[0]), "y_max": float(clipped_upper[1]), "label": box["label"]})
    output_points = []
    for point in points:
        x, y = matrix @ [point["x"], point["y"], 1]
        # BLAS can round a mathematically exact closed edge a few ulps outside.
        # Preserve that edge convention without clamping genuinely outside probes.
        tolerance = 16 * np.finfo(np.float64).eps * max(width, height)
        if point["visible"]:
            if -tolerance <= x < 0: x = 0.0
            if width < x <= width + tolerance: x = float(width)
            if -tolerance <= y < 0: y = 0.0
            if height < y <= height + tolerance: y = float(height)
        output_points.append({"x": float(x), "y": float(y), "label": point["label"], "visible": bool(point["visible"] and 0 <= x <= width and 0 <= y <= height)})
    return output_boxes, output_points


def expected_augmentation(sample, points, replay):
    source = sample["image"]
    image = np.array(source["data"], dtype=np.uint8).reshape(source["height"], source["width"], source["channels"])
    mask = np.array(sample["mask"], dtype=np.int32).reshape(image.shape[:2])
    boxes, points = copy.deepcopy(sample["boxes"]), copy.deepcopy(points)
    for operation in replay["operations"]:
        name = operation["op"]
        height, width, channels = image.shape
        matrix = None
        if name == "hflip":
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            matrix = np.array([[-1, 0, width], [0, 1, 0]], dtype=float)
        elif name == "vflip":
            image, mask = image[::-1].copy(), mask[::-1].copy()
            matrix = np.array([[1, 0, 0], [0, -1, height]], dtype=float)
        elif name == "rotate90":
            for _ in range(operation["turns"] % 4):
                h, w = image.shape[:2]
                image, mask = np.rot90(image, -1).copy(), np.rot90(mask, -1).copy()
                boxes, points = transformed_targets(boxes, points, np.array([[0, -1, h], [1, 0, 0]], dtype=float), h, w, 0)
        elif name == "crop":
            x, y, w, h = (operation[key] for key in ["x", "y", "width", "height"])
            image, mask = image[y:y+h, x:x+w].copy(), mask[y:y+h, x:x+w].copy()
            matrix = np.array([[1, 0, -x], [0, 1, -y]], dtype=float)
        elif name == "resize":
            w, h = operation["width"], operation["height"]
            image = hwc(F.interpolate(chw(image), size=(h, w), mode="bilinear", align_corners=False, antialias=False), quantize=True)
            ys = np.minimum(np.floor((np.arange(h) + .5) * height / h).astype(int), height - 1)
            xs = np.minimum(np.floor((np.arange(w) + .5) * width / w).astype(int), width - 1)
            mask = mask[ys[:, None], xs[None, :]].copy()
            matrix = np.array([[w / width, 0, 0], [0, h / height, 0]], dtype=float)
        elif name == "pad":
            left, top, right, bottom = (operation[key] for key in ["left", "top", "right", "bottom"])
            pixels = np.empty((height + top + bottom, width + left + right, channels), dtype=np.uint8)
            pixels[:] = operation["fill"]
            pixels[top:top+height, left:left+width] = image
            image = pixels
            mask = np.pad(mask, ((top, bottom), (left, right)), constant_values=operation["mask_value"])
            matrix = np.array([[1, 0, left], [0, 1, top]], dtype=float)
        elif name in ("brightness", "contrast"):
            floating = image.astype(float)
            colors = floating[:, :, :3] if channels == 4 else floating
            colors[:] = colors * operation["factor"] if name == "brightness" else (colors - 128) * operation["factor"] + 128
            image = np.floor(np.clip(floating, 0, 255) + .5).astype(np.uint8)
        else:
            raise AssertionError("replay contains unresolved or unsupported operation " + name)
        if matrix is not None:
            h, w = image.shape[:2]
            boxes, points = transformed_targets(boxes, points, matrix, w, h, replay["min_visibility"] if name == "crop" else 0)
    return image, mask, boxes, points


def assert_targets(actual_boxes, actual_points, boxes, points):
    assert len(actual_boxes) == len(boxes)
    for actual, expected in zip(actual_boxes, boxes):
        assert actual["label"] == expected["label"]
        np.testing.assert_allclose([actual[k] for k in ["x_min", "y_min", "x_max", "y_max"]], [expected[k] for k in ["x_min", "y_min", "x_max", "y_max"]], rtol=0, atol=1e-9)
    assert len(actual_points) == len(points)
    for actual, expected in zip(actual_points, points):
        assert actual["label"] == expected["label"] and actual["visible"] == expected["visible"], (actual, expected)
        np.testing.assert_allclose([actual["x"], actual["y"]], [expected["x"], expected["y"]], rtol=0, atol=1e-9)


def assert_augmentation(result, expected):
    image, mask, boxes, points = expected
    actual = np.array(result["sample"]["image"]["data"], dtype=np.int16).reshape(image.shape)
    pixel_error = int(np.abs(actual - image.astype(np.int16)).max(initial=0))
    assert pixel_error <= 1, pixel_error
    assert result["sample"]["image"]["width"] == image.shape[1]
    assert result["sample"]["image"]["height"] == image.shape[0]
    np.testing.assert_array_equal(result["sample"]["mask"], mask.reshape(-1))
    assert_targets(result["sample"]["boxes"], result["keypoints"], boxes, points)
    return pixel_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/real-images-report.json")
    parser.add_argument("--preview", type=Path, default=ROOT / "artifacts/real-images-results.png")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    bridges = {name: Bridge(name) for name in ("augment", "warp", "interop")}
    reports, previews = [], []
    try:
        for fixture in manifest["fixtures"]:
            path = SAMPLES / fixture["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
            with Image.open(path) as decoded:
                image = np.array(decoded, dtype=np.uint8)
            assert hashlib.sha256(image.tobytes()).hexdigest() == fixture["pixels_sha256"]
            if image.ndim == 2:
                image = image[:, :, None]
            h, w, c = image.shape
            assert [w, h, c] == [fixture["width"], fixture["height"], fixture["channels"]]
            sample, points = sample_for(image)
            pipelines = [
                [{"op": "hflip"}, {"op": "rotate90", "turns": 1}, {"op": "resize", "width": 113, "height": 79}, {"op": "brightness", "factor": .9}, {"op": "contrast", "factor": 1.1}],
                [{"op": "letterbox", "width": 211, "height": 157, "fill": [114] * c if c != 4 else [114, 114, 114, 255], "mask_value": 99, "scale_up": False}],
                [{"op": "random_resized_crop", "width": 128, "height": 96, "scale_min": .5, "scale_max": 1.0, "ratio_min": .75, "ratio_max": 1.3}, {"op": "vflip", "p": .5}],
            ]
            maximum_pixel_error = 0
            for index, pipeline in enumerate(pipelines):
                request = {"action": "augment", "sample": sample, "keypoints": points, "pipeline": pipeline, "seed": 42, "sample_id": fixture["id"], "min_visibility": .25}
                result = bridges["augment"].request(request)
                maximum_pixel_error = max(maximum_pixel_error, assert_augmentation(result, expected_augmentation(sample, points, result["replay"])))
                replayed = bridges["augment"].request({"action": "replay", "sample": sample, "keypoints": points, "replay": result["replay"]})
                assert replayed == result, "replay must reproduce all pixels and annotations exactly"
                if index == 1:
                    preview_augmented = np.array(result["sample"]["image"]["data"], np.uint8).reshape(157, 211, c)
                if index == 2:
                    assert bridges["augment"].request(request) == result, "fixed key must be deterministic"
            input_tensor = chw(image) / 255.0
            shape = list(input_tensor.shape)
            theta = torch.tensor([[[.973, .037, .121], [-.049, 1.017, -.083]]], dtype=torch.float64)
            align = bool(len(reports) % 2)
            grid = F.affine_grid(theta, shape, align_corners=align)
            maximum_warp_error = 0.0
            nearest_tie_positions = 0
            moon_grid = None
            for mode, padding in [("bilinear", "zeros"), ("bilinear", "border"), ("bilinear", "reflection"), ("nearest", "zeros")]:
                expected = F.grid_sample(input_tensor, grid, mode=mode, padding_mode=padding, align_corners=align)
                result = bridges["warp"].request({"action": "sample", "input": tensor_json(input_tensor), "theta": theta.reshape(-1).tolist(), "out_height": h, "out_width": w, "align_corners": align, "mode": mode, "padding": padding})
                actual = np.array(result["output"]["data"])
                sampled_grid = torch.tensor(result["grid"]["data"], dtype=torch.float64).reshape(1, h, w, 2)
                np.testing.assert_allclose(sampled_grid.numpy(), grid.numpy(), rtol=0, atol=1e-14)
                moon_grid = sampled_grid
                if mode == "nearest":
                    # Nearest is discontinuous at half pixels: ~1ulp differences
                    # between affine-grid implementations can change one pixel.
                    # Check exact sampling on identical supplied coordinates,
                    # then exact theta-route pixels away from those tie points.
                    direct = bridges["warp"].request({"action": "sample", "input": tensor_json(input_tensor), "grid": {"shape": list(grid.shape), "data": grid.reshape(-1).tolist()}, "align_corners": align, "mode": mode, "padding": padding})
                    np.testing.assert_array_equal(direct["output"]["data"], expected.numpy().reshape(-1))
                    same_grid_expected = F.grid_sample(input_tensor, sampled_grid, mode=mode, padding_mode=padding, align_corners=align)
                    np.testing.assert_array_equal(actual, same_grid_expected.numpy().reshape(-1))
                    coords = grid.numpy()[0]
                    px = (coords[:, :, 0] + 1) * (w - 1) / 2 if align else ((coords[:, :, 0] + 1) * w - 1) / 2
                    py = (coords[:, :, 1] + 1) * (h - 1) / 2 if align else ((coords[:, :, 1] + 1) * h - 1) / 2
                    non_ties = (np.abs(px - np.floor(px) - .5) > 1e-10) & (np.abs(py - np.floor(py) - .5) > 1e-10)
                    nearest_tie_positions = int(np.count_nonzero(~non_ties))
                    np.testing.assert_array_equal(actual.reshape(shape)[0][:, non_ties], expected.numpy()[0][:, non_ties])
                    expected = same_grid_expected
                error = float(np.max(np.abs(actual - expected.numpy().reshape(-1))))
                assert error < 1e-11, (fixture["id"], mode, padding, error)
                maximum_warp_error = max(maximum_warp_error, error)
            x = input_tensor.clone().requires_grad_()
            actual_grid = moon_grid.clone().requires_grad_()
            weights = torch.linspace(-.2, .9, x.numel(), dtype=torch.float64).reshape(shape) / x.numel()
            predicted = F.grid_sample(x, actual_grid, padding_mode="reflection", align_corners=align)
            (predicted * weights).sum().backward()
            t = theta.clone().requires_grad_()
            affine_reference = F.affine_grid(t, shape, align_corners=align)
            affine_reference.backward(actual_grid.grad)
            result = bridges["warp"].request({"action": "backward", "input": tensor_json(input_tensor), "theta": theta.reshape(-1).tolist(), "grad_output": tensor_json(weights), "align_corners": align, "mode": "bilinear", "padding": "reflection"})
            gradient_error = max(float(np.max(np.abs(np.array(result["input_grad"]["data"]) - x.grad.numpy().reshape(-1)))), float(np.max(np.abs(np.array(result["theta_grad"]) - t.grad.numpy().reshape(-1)))))
            assert gradient_error < 1e-10
            # A theta perturbation crossing a pixel knot has no unique central
            # derivative. Find a deterministic nearby affine with a verified
            # margin; this is a separate strict chain-rule/FD audit.
            random = np.random.default_rng(20261025 + len(reports))
            for _ in range(100):
                safe_theta = theta + torch.tensor(random.uniform(-.001, .001, size=(1, 2, 3)), dtype=torch.float64)
                safe_grid = F.affine_grid(safe_theta, shape, align_corners=align)
                coords = safe_grid.numpy()[0]
                px = (coords[:, :, 0] + 1) * (w - 1) / 2 if align else ((coords[:, :, 0] + 1) * w - 1) / 2
                py = (coords[:, :, 1] + 1) * (h - 1) / 2 if align else ((coords[:, :, 1] + 1) * h - 1) / 2
                margin = min(float(np.min(np.abs(px - np.rint(px)))), float(np.min(np.abs(py - np.rint(py)))))
                if margin > 1e-5:
                    break
            else:
                raise AssertionError("could not construct a smooth affine finite-difference probe")
            safe_t = safe_theta.clone().requires_grad_()
            safe_prediction = F.grid_sample(input_tensor, F.affine_grid(safe_t, shape, align_corners=align), padding_mode="reflection", align_corners=align)
            (safe_prediction * weights).sum().backward()
            safe_gradient = bridges["warp"].request({"action": "backward", "input": tensor_json(input_tensor), "theta": safe_theta.reshape(-1).tolist(), "grad_output": tensor_json(weights), "align_corners": align, "mode": "bilinear", "padding": "reflection"})
            np.testing.assert_allclose(safe_gradient["theta_grad"], safe_t.grad.numpy().reshape(-1), rtol=0, atol=1e-10)
            finite_differences = []
            for k in range(6):
                deltas = torch.zeros_like(safe_theta)
                deltas.reshape(-1)[k] = 1e-8
                losses = []
                for perturbed in (safe_theta + deltas, safe_theta - deltas):
                    value = F.grid_sample(input_tensor, F.affine_grid(perturbed, shape, align_corners=align), padding_mode="reflection", align_corners=align)
                    losses.append(float((value * weights).sum()))
                finite_differences.append((losses[0] - losses[1]) / 2e-8)
            finite_difference_error = float(np.max(np.abs(np.array(finite_differences) - safe_t.grad.numpy().reshape(-1))))
            assert finite_difference_error < 5e-6, (fixture["id"], finite_difference_error)
            warp = bridges["interop"].request({"action": "warp_sample", "sample": sample, "keypoints": points, "theta": theta.reshape(-1).tolist(), "out_height": h, "out_width": w, "align_corners": align, "mask_fill": 99, "min_visibility": .25})
            warped_image = F.grid_sample(input_tensor, grid, align_corners=align)
            expected_bytes = np.floor(np.clip(hwc(warped_image) * 255, 0, 255) + .5).astype(np.uint8)
            actual_bytes = np.array(warp["sample"]["image"]["data"], np.int16).reshape(image.shape)
            assert np.max(np.abs(actual_bytes - expected_bytes.astype(np.int16))) <= 1
            mask = torch.tensor(sample["mask"], dtype=torch.float64).reshape(1, 1, h, w)
            expected_mask = F.grid_sample(mask, moon_grid, mode="nearest", align_corners=align).numpy().reshape(-1).astype(np.int64)
            coverage = F.grid_sample(torch.ones_like(mask), moon_grid, mode="nearest", align_corners=align).numpy().reshape(-1)
            expected_mask[coverage == 0] = 99
            np.testing.assert_array_equal(warp["sample"]["mask"], expected_mask)
            def normalizer(height, width):
                sx, sy = (2 / (width - 1), 2 / (height - 1)) if align else (2 / width, 2 / height)
                return np.array([[sx, 0, -1 if align else 1 / width - 1], [0, sy, -1 if align else 1 / height - 1], [0, 0, 1]])
            backward = np.linalg.inv(normalizer(h, w)) @ np.vstack([theta.numpy()[0], [0, 0, 1]]) @ normalizer(h, w)
            to_center = np.array([[1, 0, -.5], [0, 1, -.5], [0, 0, 1]])
            to_edge = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]])
            forward = (to_edge @ np.linalg.inv(backward) @ to_center)[:2]
            boxes, mapped_points = transformed_targets(sample["boxes"], points, forward, w, h, .25)
            assert_targets(warp["sample"]["boxes"], warp["keypoints"], boxes, mapped_points)
            small = image[:128, :128]
            second = small[:, ::-1].copy()
            images = [{"width": item.shape[1], "height": item.shape[0], "channels": c, "data": item.reshape(-1).tolist()} for item in (small, second)]
            mean, std = [.4] * c, [.2] * c
            packed = bridges["interop"].request({"action": "pack", "images": images, "mean": mean, "std": std})
            normalized = (torch.cat([chw(small), chw(second)]) / 255.0 - .4) / .2
            np.testing.assert_allclose(packed["tensor"]["data"], normalized.numpy().reshape(-1), rtol=0, atol=1e-12)
            unpacked = bridges["interop"].request({"action": "unpack", "tensor": packed["tensor"], "mean": mean, "std": std})
            assert unpacked["images"] == images
            registration = None
            if fixture["id"] in ("astronaut", "chelsea", "clock"):
                with Image.open(path) as decoded:
                    gray = np.array(decoded.convert("L").resize((32, 32), Image.Resampling.LANCZOS))[:, :, None]
                source = chw(gray) / 255.0
                angle = .035
                affine = torch.tensor([[[math.cos(angle), -math.sin(angle), .03], [math.sin(angle), math.cos(angle), -.02]]], dtype=torch.float64)
                target = F.grid_sample(source, F.affine_grid(affine, source.shape, align_corners=False), padding_mode="border", align_corners=False)
                fitted = bridges["warp"].request({"action": "register_multiscale", "input": tensor_json(source), "target": tensor_json(target), "levels": 2, "iterations_per_level": 60, "learning_rate": .5, "padding": "border"})
                assert fitted["final_loss"] < fitted["initial_loss"]
                learned_grid = F.affine_grid(torch.tensor(fitted["theta"], dtype=torch.float64).reshape(1, 2, 3), source.shape, align_corners=False)
                expected_output = F.grid_sample(source, learned_grid, padding_mode="border", align_corners=False)
                np.testing.assert_allclose(fitted["output"]["data"], expected_output.reshape(-1).numpy(), rtol=0, atol=1e-11)
                registration = {"initial_loss": fitted["initial_loss"], "final_loss": fitted["final_loss"], "loss_reduction_percent": (1 - fitted["final_loss"] / fitted["initial_loss"]) * 100}
            reports.append({"id": fixture["id"], "category": fixture["category"], "mode": fixture["mode"], "dimensions": [w, h], "augmentation_cases": 3, "sampling_cases": 4, "nearest_tie_positions": nearest_tie_positions, "nearest_contract": "identical supplied grid exact; theta route exact away from half-pixel knots", "max_augmentation_byte_error": maximum_pixel_error, "max_sampling_absolute_error": maximum_warp_error, "max_gradient_absolute_error": gradient_error, "finite_difference_knot_margin": margin, "max_theta_finite_difference_error": finite_difference_error, "exact_replay": True, "exact_mask_categories": True, "annotations": "geometric probes", "registration": registration})
            previews.append((fixture["id"], image, preview_augmented, actual_bytes.astype(np.uint8)))
            print(f"PASS {fixture['id']:20s} {w:3d}x{h:<3d} {fixture['mode']:4s} pixels<={maximum_pixel_error} warp={maximum_warp_error:.2e} grad={gradient_error:.2e}", flush=True)
        report = {"passed": True, "real_source_photos": len(manifest["sources"]), "fixtures": len(reports), "categories": sorted({r["category"] for r in reports}), "bridge_requests": sum(bridge.calls for bridge in bridges.values()), "elapsed_seconds": round(time.perf_counter() - started, 3), "versions": {"torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__}, "cases": reports}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        montage = Image.new("RGB", (3 * 224, len(previews) * 170 + 24), "#f1f5ef")
        draw = ImageDraw.Draw(montage)
        for col, name in enumerate(["Real photo input", "MoonAug letterbox", "MoonWarp + annotations"]):
            draw.text((col * 224 + 10, 6), name, fill="#16312f")
        for row, (identifier, *items) in enumerate(previews):
            for col, pixels in enumerate(items):
                picture = Image.fromarray(pixels[:, :, 0] if pixels.shape[2] == 1 else pixels).convert("RGBA")
                picture.thumbnail((208, 138), Image.Resampling.LANCZOS)
                position = (col * 224 + (224 - picture.width) // 2, 24 + row * 170 + (142 - picture.height) // 2)
                montage.paste(picture, position, picture)
                draw.text((col * 224 + 8, 24 + row * 170 + 146), identifier, fill="#16312f")
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        montage.save(args.preview)
        print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    finally:
        for bridge in bridges.values():
            bridge.close()


if __name__ == "__main__":
    main()
