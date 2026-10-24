"""Run the repository's required checks; fail immediately on any failed command."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", choices=["wasm-gc", "wasm", "js", "native"], help="test selected backends; default: all four")
    parser.add_argument("--references", action="store_true", help="also check regenerated fixtures using pinned Python dependencies")
    args = parser.parse_args()
    run("moon", "check", "--target", "all", "--deny-warn")
    for target in args.target or ["wasm-gc", "wasm", "js", "native"]:
        run("moon", "test", "--target", target, "--deny-warn")
    run("moon", "run", "examples/main")
    run("moon", "run", "examples/pipeline")
    run(sys.executable, "tools/build_web.py")
    run("node", "tools/test_bridge.mjs")
    if args.references:
        run(sys.executable, "verification/generate_references.py", "--check-numerical")
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
