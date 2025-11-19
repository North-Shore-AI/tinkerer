"""Utility helpers for writing structured stage metrics to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    temp_path.replace(path)


@dataclass
class MetricsEmitter:
    """Writes stage manifests + dashboard index entries for CLI runs."""

    project_root: Path | None = None
    artifacts_dir: str = "artifacts"
    dashboard_dir: str = "dashboard_data"

    def __post_init__(self) -> None:
        self._root = (self.project_root or _default_repo_root()).resolve()
        self._artifacts_root = self._root / self.artifacts_dir
        self._dashboard_root = self._root / self.dashboard_dir
        self._index_path = self._dashboard_root / "index.json"

    def emit(self, stage: str, payload: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        timestamp = _iso_timestamp()
        slug = stage.replace(" ", "_")
        run_identifier = run_id or f"{slug}_{timestamp.replace(':', '').replace('-', '')}"

        manifest = {
            "stage": stage,
            "run_id": run_identifier,
            "timestamp": timestamp,
            "payload": payload or {},
        }

        manifest_path = self._write_manifest(stage=slug, run_id=run_identifier, manifest=manifest)
        index_entry = {
            "stage": stage,
            "run_id": run_identifier,
            "timestamp": timestamp,
            "manifest_path": str(self._relative_to_root(manifest_path)),
        }
        self._update_index(index_entry)
        return {
            "manifest_path": manifest_path,
            "index_entry": index_entry,
        }

    def _write_manifest(self, stage: str, run_id: str, manifest: Dict[str, Any]) -> Path:
        stage_dir = self._artifacts_root / stage / run_id
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        return manifest_path

    def _relative_to_root(self, path: Path) -> Path:
        try:
            return path.relative_to(self._root)
        except ValueError:
            return path

    def _load_index(self) -> Dict[str, Any]:
        if not self._index_path.exists():
            return {"runs": []}
        with self._index_path.open("r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {"runs": []}

    def _update_index(self, entry: Dict[str, Any]) -> None:
        index = self._load_index()
        runs = index.setdefault("runs", [])
        runs = [existing for existing in runs if existing.get("run_id") != entry["run_id"]]
        runs.append(entry)
        index["runs"] = sorted(runs, key=lambda item: item.get("timestamp", ""), reverse=True)
        _write_json(self._index_path, index)


def emit(stage: str, payload: Dict[str, Any], run_id: Optional[str] = None, *, project_root: Path | None = None) -> Dict[str, Any]:
    """Convenience wrapper used by the CLI to record stage metrics."""

    emitter = MetricsEmitter(project_root=project_root)
    return emitter.emit(stage=stage, payload=payload, run_id=run_id)
