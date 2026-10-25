"""Count library MoonBit source, excluding tests, generated interfaces and demos."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    packages = {}
    for package in ("moonaug", "moonwarp", "interop"):
        files = []
        for path in sorted((ROOT / package).glob("*.mbt")):
            if path.name.endswith(("_test.mbt", "_wbtest.mbt")):
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            files.append({"path": path.relative_to(ROOT).as_posix(), "physical_lines": len(lines), "nonblank_noncomment_lines": sum(bool(line.strip()) and not line.lstrip().startswith("//") for line in lines)})
        packages[package] = {"physical_lines": sum(item["physical_lines"] for item in files), "nonblank_noncomment_lines": sum(item["nonblank_noncomment_lines"] for item in files), "files": files}
    report = {"scope": "library packages only; tests, generated interfaces, Python, CLI and browser excluded", "packages": packages, "physical_lines": sum(item["physical_lines"] for item in packages.values()), "nonblank_noncomment_lines": sum(item["nonblank_noncomment_lines"] for item in packages.values())}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, item in packages.items():
            print(f"{name}: {item['physical_lines']} source lines / {item['nonblank_noncomment_lines']} nonblank, noncomment")
        print(f"Total: {report['physical_lines']} source lines / {report['nonblank_noncomment_lines']} nonblank, noncomment")


if __name__ == "__main__":
    main()
