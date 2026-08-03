"""
Preview the Quarto documentation locally.

The docs are published from GitHub Actions, which needs Pages enabled on the
repository. While the repository is private, this is the way to look at them.

    uv run python preview_docs.py              # live preview, opens a browser
    uv run python preview_docs.py --render     # build once into docs/_site
    uv run python preview_docs.py --port 5000  # pick the port yourself

Nothing in docs/ is executed at render time, so no data folder is needed.
Quarto comes from the `quarto-cli` package in the dev group:

    uv sync --group dev
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"


def find_quarto() -> str:
    """Quarto from the active environment, else whatever is on PATH."""
    for candidate in (
        Path(sys.prefix) / "Scripts" / "quarto.exe",  # Windows venv
        Path(sys.prefix) / "bin" / "quarto",  # POSIX venv
    ):
        if candidate.is_file():
            return str(candidate)
    return "quarto"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--render", action="store_true",
                        help="build once instead of starting the live preview")
    parser.add_argument("--port", type=int, default=None,
                        help="port for the preview server")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    quarto = find_quarto()
    cmd = [quarto, "render" if args.render else "preview", str(DOCS_DIR)]
    if not args.render:
        if args.port:
            cmd += ["--port", str(args.port)]
        if args.no_browser:
            cmd += ["--no-browser"]

    print(f"Running: {' '.join(cmd)}\n")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Quarto not found. Install it with:\n"
              "    uv sync --group dev\n"
              "or from https://quarto.org/docs/get-started/", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
