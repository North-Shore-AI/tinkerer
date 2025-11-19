#!/usr/bin/env python3
"""CLI helper that boots the dashboard server via the repo's virtualenv."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_VENV = Path(".venv")


def _venv_python(venv_path: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve Thinker dashboard via a venv python")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtualenv path (default: .venv)")
    parser.add_argument("--port", type=int, default=43117, help="Port for dashboard.server (default: 43117)")
    args = parser.parse_args()

    venv_path = args.venv.resolve()
    python_path = _venv_python(venv_path)
    if not python_path.exists():
        print(f"[serve-dashboard] missing interpreter at {python_path}. Create the venv and install deps first.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    bin_dir = python_path.parent
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])

    url = f"http://127.0.0.1:{args.port}"
    print(f"[serve-dashboard] activating {venv_path} and serving {url}")
    cmd = [str(python_path), "-m", "dashboard.server", "--port", str(args.port)]
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        print(f"[serve-dashboard] dashboard exited with code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
