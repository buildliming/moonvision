"""Independent, deterministic checks on degraded real photos and narrow images.

Reuses the real-photo bridge and annotation oracle. Degradations are test inputs,
not new augmentation APIs or evidence of recognition/registration accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from test_real_images import (
    ROOT, SAMPLES, Bridge, assert_augmentation, chw, expected_augmentation,
    hwc, sample_for, tensor_json,
)


def degraded_images(image, seed):
    """Yield named uint8 inputs; leave RGBA alpha untouched by degradation."""
    colors = image[:, :, :3] if image.shape[2] == 4 else image
    yy, xx = np.mgrid[-3:4, -3:4]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * 1.7 ** 2))
    disk = (xx * xx + yy * yy <= 9).astype(float)
    motion = np.eye(7, dtype=float)
    for name, kernel in (("gaussian_blur", gaussian), ("defocus_blur", disk), ("motion_blur", motion)):
        channels = colors.shape[2]
        weights = torch.tensor(kernel / kernel.sum(), dtype=torch.float64)[None, None].repeat(channels, 1, 1, 1)
        blurred = F.conv2d(F.pad(chw(colors), (3, 3, 3, 3), mode="replicate"), weights, groups=channels)
        output = image.copy()
        output[:, :, :channels] = hwc(blurred, quantize=True)
        yield name, output
    random = np.random.default_rng(seed)
    for name, pixels in (
        ("low_contrast", (colors.astype(float) - 128) * .015 + 128),
        ("sensor_noise", colors.astype(float) + random.normal(0, 35, colors.shape)),
        ("constant", np.full(colors.shape, 128.0)),
    ):
        output = image.copy()
        output[:, :, :colors.shape[2]] = np.floor(np.clip(pixels, 0, 255) + .5).astype(np.uint8)
        yield name, output
    output = image.copy()
    output[3:14, 5:19, :colors.shape[2]] = 0
    yield "occlusion", output
    yield "single_row", image[:1, :2].copy()
    yield "single_column", image[:2, :1].copy()
    yield "single_pixel", image[:1, :1].copy()


def check_sampling(bridge, image):
    """Compare all mode/padding/corner combinations on identical coordinates."""
    source = chw(image) / 255
    # A second batch has different content, exercising batch/channel strides.
    source = torch.cat([source, source.flip(-1) * .7])
    random = np.random.default_rng(20261027)
    grid = torch.tensor(random.uniform(-3.2, 3.2, (2, 5, 7, 2)), dtype=torch.float64)
    grid[:, 0, :3] = torch.tensor([[-1., -1.], [0., 0.], [1., 1.]], dtype=torch.float64)
    maximum_error, maximum_gradient_error = 0.0, 0.0
    for align in (False, True):
        for mode in ("bilinear", "nearest"):
            for padding in ("zeros", "border", "reflection"):
                x = source.clone().requires_grad_()
                g = grid.clone().requires_grad_()
                expected = F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=align)
                request = {"input": tensor_json(source), "grid": tensor_json(grid), "align_corners": align, "mode": mode, "padding": padding}
                actual = bridge.request({"action": "sample", **request})
                assert actual["output"]["shape"] == list(expected.shape)
                np.testing.assert_allclose(actual["output"]["data"], expected.detach().numpy().reshape(-1), rtol=0, atol=1e-12 if mode == "bilinear" else 0)
                maximum_error = max(maximum_error, float(np.max(np.abs(np.array(actual["output"]["data"]) - expected.detach().numpy().reshape(-1)))))
                weights = torch.linspace(-.4, .8, expected.numel(), dtype=torch.float64).reshape(expected.shape) / expected.numel()
                (expected * weights).sum().backward()
                backward = bridge.request({"action": "backward", "grad_output": tensor_json(weights), **request})
                for key, reference in (("input_grad", x.grad), ("grid_grad", g.grad)):
                    np.testing.assert_allclose(backward[key]["data"], reference.numpy().reshape(-1), rtol=0, atol=1e-11)
                    maximum_gradient_error = max(maximum_gradient_error, float(np.max(np.abs(np.array(backward[key]["data"]) - reference.numpy().reshape(-1)))))
    return maximum_error, maximum_gradient_error


def check_augmentation(bridge, sample, points, identifier):
    channels = sample["image"]["channels"]
    # 2 -> 49 forces a categorical center exactly onto a source pixel boundary.
    # Random crop plus padding also exercises resolved replay and hidden points.
    request = {
        "action": "augment", "sample": sample, "keypoints": points,
        "seed": -20261027, "sample_id": identifier + "-复杂场景", "min_visibility": .25,
        "pipeline": [
            {"op": "resize", "width": 49, "height": 49},
            {"op": "random_resized_crop", "width": 19, "height": 13, "scale_min": .25, "scale_max": .95, "ratio_min": .5, "ratio_max": 2.0},
            {"op": "pad", "left": 2, "top": 1, "right": 3, "bottom": 2, "fill": [17] * channels, "mask_value": 2147483647},
            {"op": "rotate90", "turns": -1}, {"op": "hflip", "p": .5},
        ],
    }
    result = bridge.request(request)
    error = assert_augmentation(result, expected_augmentation(sample, points, result["replay"]))
    assert bridge.request(request) == result, identifier + ": deterministic execution"
    assert bridge.request({"action": "replay", "sample": sample, "keypoints": points, "replay": result["replay"]}) == result, identifier + ": serialized replay"
    return error


def check_interop(bridge, sample, points):
    image = sample["image"]
    channels = image["channels"]
    mean = [.1 + .07 * channel for channel in range(channels)]
    std = [.2 + .03 * channel for channel in range(channels)]
    packed = bridge.request({"action": "pack", "images": [image, image], "mean": mean, "std": std})
    assert packed["tensor"]["shape"] == [2, channels, image["height"], image["width"]]
    pixels = np.array(image["data"]).reshape(image["height"], image["width"], channels)
    reference = (torch.cat([chw(pixels), chw(pixels)]) / 255 - torch.tensor(mean, dtype=torch.float64)[None, :, None, None]) / torch.tensor(std, dtype=torch.float64)[None, :, None, None]
    np.testing.assert_allclose(packed["tensor"]["data"], reference.numpy().reshape(-1), rtol=0, atol=1e-12)
    unpacked = bridge.request({"action": "unpack", "tensor": packed["tensor"], "mean": mean, "std": std})
    assert unpacked["images"] == [image, image], "normalized multi-channel byte round trip"
    identity = {"action": "warp_sample", "sample": sample, "keypoints": points, "theta": [1, 0, 0, 0, 1, 0], "out_height": image["height"], "out_width": image["width"], "mask_fill": 2147483647}
    result = bridge.request(identity)
    assert result["sample"] == sample, "identity preserves pixels, int32 categories, and boxes"
    assert result["keypoints"] == points, "identity preserves keypoint visibility history"
    if min(image["height"], image["width"]) == 1:
        rejected = bridge.request({**identity, "align_corners": True}, success=False)
        assert isinstance(rejected.get("error"), str) and rejected["error"]


def check_error_recovery(bridges):
    """Malformed nested fields must be rejected without poisoning the process."""
    cases = {
        "augment": {"action": "augment", "sample": {"image": {"width": 1, "height": 1, "channels": 1, "data": [10]}}, "pipeline": []},
        "warp": {"action": "sample", "input": {"shape": [1, 1, 1, 1], "data": [1]}, "grid": {"shape": [1, 1, 1, 2], "data": [0, 0]}},
        "interop": {"action": "pack", "images": [{"width": 1, "height": 1, "channels": 1, "data": [10]}]},
    }
    for name, valid in cases.items():
        bridge = bridges[name]
        baseline = bridge.request(valid)
        nested_key = {"augment": "sample", "warp": "input", "interop": "images"}[name]
        for invalid in (None, False, "", -1, .5, {}, [None]):
            result = bridge.request({**valid, nested_key: invalid}, success=False)
            assert isinstance(result.get("error"), str) and result["error"]
            assert bridge.request(valid) == baseline, name + ": recovery after rejection"


def check_registration(bridge, image):
    """Local optimization must return consistent, non-worsening finite results."""
    source = chw(image) / 255
    theta = torch.tensor([[[.997, -.043, .071], [.043, .997, -.059]]], dtype=torch.float64)
    target = F.grid_sample(source, F.affine_grid(theta, source.shape, align_corners=False), padding_mode="border", align_corners=False)
    identity = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64)
    initial = F.grid_sample(source, F.affine_grid(identity, source.shape, align_corners=False), padding_mode="border", align_corners=False)
    initial_loss = float(((initial - target) ** 2).mean())
    losses = {}
    for action in ("register", "register_multiscale"):
        request = {"action": action, "input": tensor_json(source), "target": tensor_json(target), "padding": "border", "learning_rate": .5}
        request.update({"iterations": 12} if action == "register" else {"iterations_per_level": 12, "levels": 3})
        result = bridge.request(request)
        learned = torch.tensor(result["theta"], dtype=torch.float64).reshape(1, 2, 3)
        expected = F.grid_sample(source, F.affine_grid(learned, source.shape, align_corners=False), padding_mode="border", align_corners=False)
        np.testing.assert_allclose(result["output"]["data"], expected.numpy().reshape(-1), rtol=0, atol=1e-12)
        assert abs(result["initial_loss"] - initial_loss) < 1e-12
        assert abs(result["final_loss"] - float(((expected - target) ** 2).mean())) < 1e-12
        assert np.isfinite(result["final_loss"]) and result["final_loss"] <= result["initial_loss"]
        histories = [result["losses"]] if action == "register" else [level["losses"] for level in result["levels"]]
        for history in histories:
            assert all(np.isfinite(value) for value in history)
            assert all(later <= earlier for earlier, later in zip(history, history[1:]))
        losses[action] = {"initial_loss": result["initial_loss"], "final_loss": result["final_loss"]}
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/stress-images-report.json")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    bridges = {name: Bridge(name) for name in ("augment", "warp", "interop")}
    cases = []
    try:
        check_error_recovery(bridges)
        for fixture in manifest["fixtures"]:
            path = SAMPLES / fixture["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
            with Image.open(path) as decoded:
                decoded_pixels = np.array(decoded, dtype=np.uint8)
                assert hashlib.sha256(decoded_pixels.tobytes()).hexdigest() == fixture["pixels_sha256"]
                image = np.array(decoded.resize((23, 17), Image.Resampling.BOX), dtype=np.uint8)
            if image.ndim == 2:
                image = image[:, :, None]
            for name, pixels in degraded_images(image, 20261027):
                identifier = fixture["id"] + "/" + name
                sample, points = sample_for(pixels)
                # Adjacent maximal int32 labels catch accidental float32 masks.
                sample["mask"] = [2147483647 if value == 7 else 2147483646 if value == 3 else 0 for value in sample["mask"]]
                if min(pixels.shape[:2]) == 1:
                    # The geometric probes otherwise leave narrow masks all zero,
                    # which cannot detect a wrong class at the 2 -> 49 center tie.
                    sample["mask"] = [2147483646 + index % 2 for index in range(pixels.shape[0] * pixels.shape[1])]
                pixel_error = check_augmentation(bridges["augment"], sample, points, identifier)
                sampling_error, gradient_error = check_sampling(bridges["warp"], pixels)
                check_interop(bridges["interop"], sample, points)
                registration = check_registration(bridges["warp"], pixels) if name in ("gaussian_blur", "defocus_blur", "motion_blur", "low_contrast", "constant", "single_row", "single_column", "single_pixel") else None
                cases.append({"id": identifier, "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "shape": list(pixels.shape), "max_augmentation_byte_error": pixel_error, "max_sampling_absolute_error": sampling_error, "max_gradient_absolute_error": gradient_error, "registration": registration})
            print(f"PASS {fixture['id']}: 10 degradation/shape cases", flush=True)
        report = {"passed": True, "source_fixtures": len(manifest["fixtures"]), "cases": cases, "bridge_requests": sum(bridge.calls for bridge in bridges.values()), "elapsed_seconds": round(time.perf_counter() - started, 3), "versions": {"torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__}}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({**{key: value for key, value in report.items() if key != "cases"}, "case_count": len(cases)}, indent=2))
    finally:
        for bridge in bridges.values():
            bridge.close()


if __name__ == "__main__":
    main()
