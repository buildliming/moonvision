"""Compile public-API controls and rejected external field-access probes.

The probes run in an isolated temporary module containing a copy of MoonWarp.
They never put deliberately invalid MoonBit source in the working repository.
Each rejection must be preceded by a successful public-API compile control.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE = "shunge/moonvision"


def main():
    # _build is excluded from MoonBit package discovery, keeping invalid probe
    # files invisible to concurrent repository checks. It is also writable in
    # restricted desktop environments whose system temp directory is protected.
    temporary_root = (ROOT / "_build" / "access-probes").resolve()
    if not temporary_root.is_relative_to(ROOT.resolve()):
        raise SystemExit("Refusing a probe base outside the repository build directory")
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moonvision-access-", dir=temporary_root) as directory:
        workspace = Path(directory).resolve()
        if workspace.parent != temporary_root or not workspace.name.startswith("moonvision-access-"):
            raise SystemExit("Refusing unexpected temporary probe path")
        (workspace / "moon.mod").write_text(f'name = "{MODULE}"\n', encoding="utf-8")
        source_package = workspace / "moonwarp"
        source_package.mkdir()
        for source in (ROOT / "moonwarp").glob("*.mbt"):
            if source.name.endswith(("_test.mbt", "_wbtest.mbt")):
                continue
            shutil.copyfile(source, source_package / source.name)
        shutil.copyfile(ROOT / "moonwarp" / "moon.pkg", source_package / "moon.pkg")
        probe_package = workspace / "probe"
        probe_package.mkdir()
        (probe_package / "moon.pkg").write_text(f'import {{ "{MODULE}/moonwarp" @warp }}\n', encoding="utf-8")
        declaration = """fn probe() -> Unit raise @warp.WarpError {
  let t = @warp.Tensor::new([1, 1, 1, 1], [1.0])
  let g = @warp.Grid::new([1, 1, 1, 2], [0.0, 0.0])
  BODY
}
"""
        probes = {
            "public_api_control": "ignore(t.shape()); ignore(t.data()); ignore(g.shape()); ignore(g.data())",
            "tensor_values": "t.values[0] = 2.0; ignore(g)",
            "tensor_dimensions": "t.dims[0] = 0; ignore(g)",
            "grid_values": "g.values[0] = 2.0; ignore(t)",
            "grid_dimensions": "g.dims[0] = 0; ignore(t)",
        }
        reports = []
        for name, body in probes.items():
            (probe_package / "probe.mbt").write_text(declaration.replace("BODY", body), encoding="utf-8")
            result = subprocess.run(
                ["moon", "check", "probe", "--target", "js", "--target-dir", str(workspace / "build" / name)],
                cwd=workspace, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if name == "public_api_control":
                if result.returncode != 0:
                    raise SystemExit(f"Public API control did not compile:\n{result.stdout}\n{result.stderr}")
            elif result.returncode == 0:
                raise SystemExit(f"Private field leak: external probe {name} compiled successfully")
            else:
                output = result.stdout + result.stderr
                if not any(word in output.lower() for word in ("field", "private", "inaccessible", "abstract type")):
                    raise SystemExit(f"Probe {name} failed for an unexpected reason:\n{output}")
            reports.append({"probe": name, "exit_code": result.returncode, "expected": "compile" if name == "public_api_control" else "reject"})
        print(json.dumps({"passed": True, "probes": reports}, indent=2))


if __name__ == "__main__":
    main()
