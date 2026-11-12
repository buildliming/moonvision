"""Run reproducible checks and save the toolchain, commands, and results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["wasm-gc", "wasm", "js", "native"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", choices=TARGETS, help="check, build, test, and run examples on selected backends; default: all four")
    parser.add_argument("--references", action="store_true", help="also check regenerated fixtures using pinned Python dependencies")
    parser.add_argument("--real-images", action="store_true", help="also compare offline real photos against NumPy/PyTorch")
    parser.add_argument("--timeout", type=int, default=600, help="maximum seconds per command (default: 600)")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "validation.json", help="save command results as JSON")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    targets = list(dict.fromkeys(args.target or TARGETS))
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "targets": targets,
        "command_timeout_seconds": args.timeout,
        "references": args.references,
        "real_images": args.real_images,
        "passed": False,
        "steps": [],
    }

    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(*command_args):
        command = list(map(str, command_args))
        print("+ " + " ".join(command), flush=True)
        started = time.monotonic()
        step = {"command": command, "passed": False}
        report["steps"].append(step)
        save_report()
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
            step.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
            result.check_returncode()
            step["passed"] = True
        except (OSError, subprocess.SubprocessError) as error:
            step["error"] = str(error)
            raise
        finally:
            step["seconds"] = round(time.monotonic() - started, 3)
            save_report()

    def interface_snapshot():
        excluded = {"_build", "_target", "target", ".mooncakes", ".venv", "artifacts", "node_modules"}
        return {
            str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
            for path in ROOT.rglob("pkg.generated.mbti")
            if not excluded.intersection(path.relative_to(ROOT).parts)
        }

    save_report()
    try:
        run("moon", "version", "--all")
        run("node", "--version")
        if (ROOT / ".git").exists():
            run("git", "rev-parse", "HEAD")
            run("git", "status", "--short")
        run("moon", "fmt", "--check")
        interfaces = interface_snapshot()
        run("moon", "info")
        report["interfaces_stable"] = interfaces == interface_snapshot()
        if not report["interfaces_stable"]:
            raise SystemExit("Generated interfaces changed; review pkg.generated.mbti and rerun verification.")
        for target in targets:
            run("moon", "check", "--target", target, "--deny-warn")
            run("moon", "build", "--target", target, "--deny-warn")
            run("moon", "test", "--target", target, "--deny-warn")
            for example in ["examples/main", "examples/pipeline", "examples/annotated"]:
                run("moon", "run", example, "--target", target)
        run(sys.executable, "tools/build_web.py")
        run("node", "tools/test_bridge.mjs")
        run("node", "tools/test_cli.mjs")
        if args.references:
            run(sys.executable, "verification/generate_references.py", "--check-numerical")
            run(sys.executable, "verification/generate_extended_references.py", "--check-numerical")
            run(sys.executable, "verification/generate_real_image_references.py", "--check-numerical")
            run(sys.executable, "verification/check_private_tensor_fields.py")
        if args.real_images:
            run(sys.executable, "examples/real_pipeline.py")
            run(sys.executable, "verification/test_real_images.py")
            run(sys.executable, "verification/test_stress_images.py")
        report["passed"] = True
    except (OSError, subprocess.SubprocessError, SystemExit) as error:
        report["error"] = str(error)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(f"Validation report: {report_path}", flush=True)
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
