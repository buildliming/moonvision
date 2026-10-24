"""Serve the local browser demo, building it first if needed."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build", action="store_true", help="rebuild the MoonBit bridge first")
    args = parser.parse_args()
    if args.build or not (ROOT / "web" / "moonvision.js").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_web.py")], cwd=ROOT, check=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "web"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"MoonVision demo: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
