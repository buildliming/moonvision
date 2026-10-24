"""Build the MoonBit JavaScript bridge used by the CLI and browser demo."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(["moon", "build", "--target", "js", "--release", "bridge"], cwd=ROOT, check=True)
    candidates = list((ROOT / "_build" / "js" / "release").rglob("bridge.js"))
    if len(candidates) != 1:
        raise SystemExit(f"Expected one bridge.js, found {candidates}")
    destination = ROOT / "web" / "moonvision.js"
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(candidates[0], destination)
    print(f"Built {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
