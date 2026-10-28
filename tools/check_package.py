"""Build the Mooncakes archive and audit its file list against the Git index."""
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# A project budget, not a claimed registry limit: keep source releases small
# after a 2.55 MB archive was rejected by the publishing service with HTTP 413.
MAX_ARCHIVE_BYTES = 1048576
REQUIRED = {
    "moon.mod", "LICENSE", "README.md",
    "moonaug/moon.pkg", "moonwarp/moon.pkg", "interop/moon.pkg",
    "web/samples/manifest.json", "web/samples/ATTRIBUTION.md",
    "examples/real_pipeline.py", "examples/real_pipeline.mjs",
    "tools/moonvision_client.mjs", "tools/test_cli.mjs",
    "docs/improvement-review.md",
}
FORBIDDEN = {
    "web/moonvision.js", "web/moonvision.js.map",
    "tools/real_bridge_runner.mjs", "项目申报书.md",
    "AGENTS.md",
}


def main():
    manifest = (ROOT / "moon.mod").read_text(encoding="utf-8")
    metadata = {}
    for key in ("name", "version"):
        match = re.search(rf'^\s*{key}\s*=\s*"([^"\r\n]+)"', manifest, re.MULTILINE)
        if not match:
            raise ValueError(f"moon.mod is missing a string {key}")
        metadata[key] = match.group(1)
    result = subprocess.run(
        ["moon", "package", "--list"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError("moon package --list failed:\n" + result.stdout)
    filename = metadata["name"].replace("/", "-") + "-" + metadata["version"] + ".zip"
    archive = ROOT / "_build" / "publish" / filename
    with zipfile.ZipFile(archive) as package:
        entries = [entry.filename for entry in package.infolist() if not entry.is_dir()]
    files = set(entries)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0"))
    problems = []
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        problems.append(f"archive exceeds the {MAX_ARCHIVE_BYTES}-byte project budget; keep presentation assets on GitHub")
    if len(entries) != len(files):
        problems.append("archive contains duplicate file entries")
    for path in sorted(files - tracked):
        problems.append("untracked archive file: " + path)
    for path in sorted(REQUIRED - files):
        problems.append("required archive file missing: " + path)
    for path in sorted(files):
        if path in FORBIDDEN or path.startswith(("artifacts/", "docs/images/")):
            problems.append("forbidden archive file: " + path)
    if problems:
        raise RuntimeError("Package audit failed:\n" + "\n".join(problems))
    print(f"Package audit passed: {metadata['name']}@{metadata['version']} ({len(files)} tracked files, {archive.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
