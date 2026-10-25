"""Prepare attributed, offline real-photo fixtures from scikit-image 0.25.2 data.

This is an optional asset-maintenance command, not a test-time download.
python tools/prepare_real_images.py --source-dir /path/to/skimage/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "web" / "samples"
DOC = "https://scikit-image.org/docs/0.25.x/api/skimage.data.html"
SOURCES = {
    "astronaut": {"file": "astronaut.png", "author": "NASA", "license": "Public domain", "source": "https://flic.kr/p/r9qvLn", "category": "person", "description": "NASA portrait of astronaut Eileen Collins"},
    "chelsea": {"file": "chelsea.png", "author": "Stefan van der Walt", "license": "CC0-1.0", "source": DOC + "#skimage.data.chelsea", "category": "animal", "description": "Chelsea the cat: fur, edges and low-contrast regions"},
    "coffee": {"file": "coffee.png", "author": "Rachel Michetti", "license": "CC0-1.0", "source": DOC + "#skimage.data.coffee", "category": "objects", "description": "Coffee cup, porcelain and wood texture"},
    "rocket": {"file": "rocket.jpg", "author": "SpaceX", "license": "Public domain", "source": "https://www.flickr.com/photos/spacexphotos/16511594820/", "category": "outdoor", "description": "DSCOVR launch, dark sky and high-contrast structures"},
    "grass": {"file": "grass.png", "author": "linolafett", "license": "CC0-1.0", "source": "https://www.deviantart.com/linolafett/art/Grass-01-434853879", "category": "texture", "description": "Grayscale natural grass texture"},
    "clock": {"file": "clock_motion.png", "author": "Stefan van der Walt", "license": "Public domain", "source": DOC + "#skimage.data.clock", "category": "blurred", "description": "Motion-blurred grayscale clock photograph"},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scaled(image: Image.Image, maximum: int) -> Image.Image:
    ratio = maximum / max(image.size)
    return image.resize(tuple(max(1, round(value * ratio)) for value in image.size), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    originals = {}
    source_metadata = {}
    for name, entry in SOURCES.items():
        source = args.source_dir / entry["file"]
        with Image.open(source) as image:
            originals[name] = image.convert("L" if name in ("grass", "clock") else "RGB")
        source_metadata[name] = {**entry, "source_sha256": sha256(source.read_bytes()), "original_size": list(originals[name].size), "license_evidence": DOC + "#skimage.data." + name, "distribution_url": "https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/" + entry["file"]}
    prepared = [(name, name, scaled(originals[name], size), "aspect-preserving Lanczos thumbnail") for name, size in [("astronaut", 192), ("chelsea", 224), ("coffee", 240), ("rocket", 256), ("grass", 192), ("clock", 224)]]
    rgba = scaled(originals["chelsea"], 160).convert("RGBA")
    yy, xx = np.mgrid[:rgba.height, :rgba.width]
    alpha = np.clip((xx / max(1, rgba.width - 1) * .7 + yy / max(1, rgba.height - 1) * .3) * 255, 0, 255).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))
    prepared.append(("chelsea_alpha", "chelsea", rgba, "photo with an explicitly synthetic alpha ramp; not a segmentation ground truth"))
    portrait = originals["astronaut"]
    prepared.append(("astronaut_portrait", "astronaut", portrait.crop((128, 0, 384, 512)).resize((96, 192), Image.Resampling.LANCZOS), "central 1:2 portrait crop"))
    wide = originals["rocket"]
    middle = wide.height // 2
    prepared.append(("rocket_wide", "rocket", wide.crop((0, middle - 80, wide.width, middle + 80)).resize((256, 64), Image.Resampling.LANCZOS), "4:1 outdoor crop"))
    prepared.append(("coffee_thumbnail", "coffee", originals["coffee"].resize((17, 11), Image.Resampling.LANCZOS), "very small real-photo thumbnail"))
    fixtures = []
    for identifier, source, image, derivation in prepared:
        filename = identifier + ".png"
        path = DESTINATION / filename
        image.save(path, optimize=False, compress_level=9)
        values = np.asarray(image, dtype=np.uint8)
        fixtures.append({"id": identifier, "file": filename, "source_id": source, "category": source_metadata[source]["category"], "mode": image.mode, "width": image.width, "height": image.height, "channels": 1 if image.mode == "L" else len(image.getbands()), "derivation": derivation, "sha256": sha256(path.read_bytes()), "pixels_sha256": sha256(values.tobytes())})
    manifest = {"version": 1, "source_collection": "scikit-image 0.25.2", "preparation": "Pillow 12.3.0, deterministic resize/crop; see tools/prepare_real_images.py", "annotation_policy": "Tests add explicit geometric boxes, points and category masks; these are test probes, not hand-labeled semantic ground truths.", "sources": source_metadata, "fixtures": fixtures}
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Real-photo fixture attribution", "", "These photographs are separate from the repository's MIT-licensed source code.", "No downloads are made during tests. Photos are public-domain/CC0 as documented by the original scikit-image data maintainers.", "", "| Source | Author | License | Evidence |", "| --- | --- | --- | --- |"]
    for name, metadata in source_metadata.items():
        lines.append(f"| [{name}]({metadata['source']}) | {metadata['author']} | {metadata['license']} | [scikit-image documentation]({metadata['license_evidence']}) |")
    lines += ["", "The manifest records original source hashes, actual fixture hashes, decoded-pixel hashes, dimensions and derivation steps. Alpha, portrait, wide and thumbnail variants are explicitly derived from the six photos. Test masks/boxes are geometric probes and are not semantic annotations.", "", "Rebuild only when intentionally updating fixtures: install scikit-image 0.25.2 and Pillow 12.3.0, then run `python tools/prepare_real_images.py --source-dir <skimage/data>`."]
    (DESTINATION / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    contact = Image.new("RGB", (5 * 192, 2 * 180), "#f1f5ef")
    draw = ImageDraw.Draw(contact)
    for index, (identifier, _, image, _) in enumerate(prepared):
        x, y = index % 5 * 192, index // 5 * 180
        preview = image.convert("RGBA")
        preview.thumbnail((176, 144), Image.Resampling.LANCZOS)
        position = (x + (192 - preview.width) // 2, y + (150 - preview.height) // 2)
        contact.paste(preview, position, preview)
        draw.text((x + 8, y + 153), identifier, fill="#16312f")
    destination = ROOT / "docs" / "images" / "real-fixtures.png"
    contact.save(destination)
    print(f"Prepared {len(fixtures)} offline fixtures from {len(SOURCES)} real photographs: {DESTINATION}")


if __name__ == "__main__":
    main()
