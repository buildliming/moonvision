"""Verified offline photos -> MoonBit augmentation/replay -> normalized NCHW batch.

Run: python examples/real_pipeline.py
Requires Python with Pillow and Node.js; build once with python tools/build_web.py.
No downloads, NumPy, PyTorch or test helpers are used. Probe annotations are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "web" / "samples"
DEFAULT_SAMPLES = ["astronaut_portrait", "chelsea", "coffee"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def decoded_samples(identifiers):
    from PIL import Image

    manifest_bytes = (SAMPLES / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["version"] != 1:
        raise ValueError("unsupported fixture manifest version")
    by_id = {entry["id"]: entry for entry in manifest["fixtures"]}
    if len(by_id) != len(manifest["fixtures"]):
        raise ValueError("duplicate IDs in fixture manifest")
    if not 1 <= len(identifiers) <= 9 or len(set(identifiers)) != len(identifiers):
        raise ValueError("choose 1 to 9 unique RGB sample IDs (JSON tensor budget)")
    entries, originals, total = [], [], 0
    for identifier in identifiers:
        if identifier not in by_id:
            raise ValueError(f"unknown sample ID: {identifier}")
        fixture = by_id[identifier]
        width, height = fixture["width"], fixture["height"]
        if fixture["mode"] != "RGB" or fixture["channels"] != 3:
            raise ValueError(f"{identifier}: select an RGB fixture for this model-input example")
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError(f"{identifier}: invalid dimensions in manifest")
        total += width * height * 3
        if total > 1048576:
            raise ValueError("decoded batch exceeds the JSON image budget")
        source = (SAMPLES / fixture["file"]).resolve()
        if not source.is_relative_to(SAMPLES.resolve()):
            raise ValueError("fixture path escapes the offline samples directory")
        encoded = source.read_bytes()
        if digest(encoded) != fixture["sha256"]:
            raise ValueError(f"{identifier}: source file SHA-256 mismatch")
        with Image.open(io.BytesIO(encoded)) as image:
            if image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"{identifier}: decoded image differs from manifest")
            pixels = image.tobytes()
            originals.append(image.copy())
        if digest(pixels) != fixture["pixels_sha256"]:
            raise ValueError(f"{identifier}: decoded-pixel SHA-256 mismatch")
        attribution = manifest["sources"][fixture["source_id"]]
        entries.append({
            "sample_id": identifier,
            "image": {"width": width, "height": height, "channels": 3, "data": list(pixels)},
            "source": {"file": source.relative_to(ROOT).as_posix(), "sha256": digest(encoded),
                       "pixels_sha256": digest(pixels), "source_id": fixture["source_id"],
                       "author": attribution["author"], "license": attribution["license"], "url": attribution["source"]},
        })
    return entries, originals, digest(manifest_bytes)


def save_preview(path, entries, originals, previews):
    from PIL import Image, ImageDraw

    cell = 224
    sheet = Image.new("RGB", (cell * len(entries), 510), "#f0f4f7")
    draw = ImageDraw.Draw(sheet)
    for i, (entry, original, preview) in enumerate(zip(entries, originals, previews)):
        x = i * cell
        draw.text((x + 12, 12), entry["sample_id"], fill="#14334a")
        display = original.copy()
        display.thumbnail((192, 192), Image.Resampling.LANCZOS)
        sheet.paste(display, (x + (cell - display.width) // 2, 42 + (192 - display.height) // 2))
        draw.text((x + 12, 245), "96x96: letterbox + seeded flip", fill="#14334a")
        image = preview["sample"]["image"]
        display = Image.frombytes("RGB", (96, 96), bytes(image["data"]))
        display = display.resize((192, 192), Image.Resampling.NEAREST)
        marks = ImageDraw.Draw(display)
        for box in preview["sample"]["boxes"]:
            marks.rectangle(tuple(box[key] * 2 for key in ("x_min", "y_min", "x_max", "y_max")), outline="#ffcc00", width=2)
        for point in preview["keypoints"]:
            px, py = point["x"] * 2, point["y"] * 2
            marks.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#00ffff")
        sheet.paste(display, (x + 16, 273))
        draw.text((x + 12, 478), "Geometric probes, not ground truth", fill="#14334a")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES, metavar="ID")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "real-pipeline")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    write_json(report_path, {"status": "running"})
    try:
        entries, originals, manifest_hash = decoded_samples(args.samples)
        completed = subprocess.run(
            ["node", str(ROOT / "examples" / "real_pipeline.mjs")],
            input=json.dumps(entries, ensure_ascii=False), capture_output=True, text=True,
            encoding="utf-8", cwd=ROOT, timeout=120, check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or f"Node exited with {completed.returncode}")
        result = json.loads(completed.stdout)
        result["replays"]["manifest_sha256"] = manifest_hash
        write_json(output / "model-input.json", result["model_input"])
        write_json(output / "replays.json", result["replays"])
        save_preview(output / "preprocessing.png", entries, originals, result["previews"])
        report = {"status": "passed", "sample_ids": args.samples, "manifest_sha256": manifest_hash,
                  "source_hashes_verified": True, "shape": result["model_input"]["shape"],
                  "checks": result["checks"], "annotation_policy": result["replays"]["annotation_policy"]}
        write_json(report_path, report)
        print(f"Passed: {len(entries)} verified real photos -> NCHW {report['shape']}")
        print(f"Replay and normalized-byte roundtrip exact; max normalization error {result['checks']['normalization_max_abs_error']}")
        print(f"Artifacts: {output}")
        return 0
    except (ImportError, OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        write_json(report_path, {"status": "failed", "error": str(error)})
        print(f"Real pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
