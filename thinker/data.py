"""
Data bootstrap utilities (e.g., download + convert SciFact).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "cns-support-models" / "scripts"
FEVER_RAW_DIR = REPO_ROOT / "cns-support-models" / "data" / "raw" / "fever"
SCIFACT_PROCESSED_DATASET = (
    REPO_ROOT / "cns-support-models" / "data" / "processed" / "scifact_claim_extractor.jsonl"
)
SCIFACT_CLEAN_DATASET = (
    REPO_ROOT / "cns-support-models" / "data" / "processed" / "scifact_claim_extractor_clean.jsonl"
)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_make_scifact() -> None:
    run(["make", "scifact"], cwd=REPO_ROOT / "cns-support-models")


def run_data_setup(
    dataset: str,
    claims_path: Path,
    corpus_path: Path,
    output_path: Path,
    clean_output: Path | None,
    filter_invalid: bool,
    fever_claims: Path,
    fever_wiki_dir: Path,
    fever_output: Path,
    fever_include_nei: bool,
    *,
    skip_validation: bool,
    validation_mode: str,
    similarity_threshold: float,
) -> None:
    if dataset == "scifact":
        _setup_scifact(
            claims_path=claims_path,
            corpus_path=corpus_path,
            output_path=output_path,
            clean_output=clean_output,
            filter_invalid=filter_invalid,
            skip_validation=skip_validation,
            validation_mode=validation_mode,
            similarity_threshold=similarity_threshold,
        )
    elif dataset == "fever":
        _setup_fever(
            claims_path=fever_claims,
            wiki_dir=fever_wiki_dir,
            output_path=fever_output,
            include_nei=fever_include_nei,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


def _setup_scifact(
    *,
    claims_path: Path,
    corpus_path: Path,
    output_path: Path,
    clean_output: Path | None,
    filter_invalid: bool,
    skip_validation: bool,
    validation_mode: str,
    similarity_threshold: float,
) -> None:
    claims_path = claims_path.resolve()
    corpus_path = corpus_path.resolve()
    output_path = output_path.resolve()

    run_make_scifact()
    if not SCIFACT_PROCESSED_DATASET.exists():
        raise FileNotFoundError(
            f"SciFact dataset not found at {SCIFACT_PROCESSED_DATASET}. "
            "Did `make scifact` finish successfully?"
        )

    if SCIFACT_PROCESSED_DATASET.resolve() != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCIFACT_PROCESSED_DATASET, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_output = clean_output.resolve() if clean_output else None
    if skip_validation and clean_output is None:
        return
    validator = SCRIPTS_ROOT / "validate_dataset.py"
    cmd = [
            sys.executable,
            str(validator),
            str(output_path),
            "--claims-json",
            str(claims_path),
        ]
    if validation_mode == "embedding":
        cmd.extend(
            [
                "--corpus-json",
                str(corpus_path),
                "--evidence-mode",
                "embedding",
                "--similarity-threshold",
                str(similarity_threshold),
            ]
        )
    else:
        cmd.extend(["--evidence-mode", "exact"])
    if clean_output:
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--write-clean", str(clean_output)])
        if filter_invalid:
            cmd.append("--filter-invalid")
    run(cmd, cwd=REPO_ROOT)


def _ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}. Please provide the raw FEVER files.")
    return path


def _setup_fever(
    *,
    claims_path: Path,
    wiki_dir: Path,
    output_path: Path,
    include_nei: bool,
) -> None:
    if not claims_path.exists() or not wiki_dir.exists():
        run_fever_download()
    claims_path = _ensure_exists(claims_path.resolve(), "FEVER claims file")
    wiki_dir = _ensure_exists(wiki_dir.resolve(), "FEVER wiki directory")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "convert_fever.py"),
        "--claims",
        str(claims_path),
        "--wiki-dir",
        str(wiki_dir),
        "--out",
        str(output_path),
    ]
    if include_nei:
        cmd.append("--include-nei")
    run(cmd, cwd=REPO_ROOT)


def run_fever_download() -> None:
    script = SCRIPTS_ROOT / "download_fever.sh"
    if not script.exists():
        raise FileNotFoundError("download_fever.sh script missing; cannot fetch FEVER automatically.")
    try:
        run(["bash", str(script), str(FEVER_RAW_DIR)], cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external failure
        raise RuntimeError(
            "Automatic FEVER download failed (site may require manual access). "
            "Please download the official FEVER release into "
            f"{FEVER_RAW_DIR} and rerun the command.\nOriginal error: {exc}"
        ) from exc
