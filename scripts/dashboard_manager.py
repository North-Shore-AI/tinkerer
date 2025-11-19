#!/usr/bin/env python3
"""Interactive menu for managing the dashboard server lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

DEFAULT_CMD = [
    sys.executable,
    "scripts/serve_dashboard.py",
    "--venv",
    ".venv",
    "--port",
    "43117",
]


class DashboardProcess:
    def __init__(self, command: List[str]):
        self.command = command
        self.process: Optional[subprocess.Popen[str]] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            print("[manager] Server already running.")
            return
        print(f"[manager] Starting dashboard: {' '.join(self.command)}")
        env = os.environ.copy()
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout_thread = threading.Thread(
            target=self._stream_output,
            args=(self.process.stdout, "STDOUT"),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._stream_output,
            args=(self.process.stderr, "STDERR"),
            daemon=True,
        )
        self.stdout_thread.start()
        self.stderr_thread.start()

    def stop(self) -> None:
        if not self.is_running:
            print("[manager] Server is not running.")
            return
        print("[manager] Sending SIGTERM...")
        assert self.process
        self.process.terminate()
        self._wait_for_exit()

    def kill(self) -> None:
        if not self.is_running:
            print("[manager] Server is not running.")
            return
        print("[manager] Force killing process...")
        assert self.process
        self.process.kill()
        self._wait_for_exit()

    def restart(self) -> None:
        print("[manager] Restarting dashboard...")
        if self.is_running:
            self.stop()
        self.start()

    def status(self) -> None:
        if self.is_running:
            assert self.process
            print(f"[manager] RUNNING (pid={self.process.pid})")
        else:
            print("[manager] STOPPED")

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _wait_for_exit(self) -> None:
        if not self.process:
            return
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[manager] Process did not exit in time.")
        self.process = None

    def _stream_output(self, pipe, label: str) -> None:
        if not pipe:
            return
        for line in pipe:
            sys.stdout.write(f"[dashboard {label}] {line}")
        pipe.close()


def ensure_scripts_executable() -> None:
    path = Path("scripts/serve_dashboard.py")
    if path.exists():
        path.chmod(path.stat().st_mode | 0o111)


def menu_loop(manager: DashboardProcess) -> None:
    actions = {
        "1": ("Start", manager.start),
        "2": ("Stop", manager.stop),
        "3": ("Restart", manager.restart),
        "4": ("Kill", manager.kill),
        "5": ("Status", manager.status),
        "6": ("Quit", None),
    }
    while True:
        print("\nDashboard Manager")
        for key, (label, _) in actions.items():
            print(f"  {key}. {label}")
        choice = input("Select an option: ").strip()
        if choice not in actions:
            print("[manager] Invalid selection.")
            continue
        label, action = actions[choice]
        if label == "Quit":
            print("[manager] Exiting dashboard manager.")
            break
        action()


def main() -> int:
    ensure_scripts_executable()
    manager = DashboardProcess(command=DEFAULT_CMD)
    try:
        menu_loop(manager)
    except KeyboardInterrupt:
        print("\n[manager] Interrupted by user.")
    finally:
        if manager.is_running:
            manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
