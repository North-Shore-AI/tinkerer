"""Simple Antagonist runner that inspects evaluation artifacts and emits flags."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AntagonistConfig:
    input_path: Path
    output_path: Path
    chirality_threshold: float = 0.55
    high_chirality_threshold: float = 0.65
    entailment_threshold: float = 0.5
    evidence_overlap_threshold: float = 0.2


class AntagonistRunner:
    """Applies lightweight polarity/entailment heuristics per the Antagonist RFC."""

    def __init__(self, config: AntagonistConfig):
        self.config = config

    def run(self) -> Dict[str, Any]:
        records = self._load_input()
        flags: List[Dict[str, Any]] = []
        for entry in records:
            flag = self._evaluate_entry(entry)
            if flag:
                flags.append(flag)

        self._write_flags(flags)
        summary = {
            "input": str(self.config.input_path),
            "output": str(self.config.output_path),
            "total_records": len(records),
            "flagged_records": len(flags),
        }
        print(
            f"[antagonist] inspected {summary['total_records']} records "
            f"and emitted {summary['flagged_records']} flags → {self.config.output_path}",
            flush=True,
        )
        return summary

    def _load_input(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.config.input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _evaluate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        validation = entry.get("validation", {})
        chirality_info = entry.get("chirality") or {}
        chirality_score = chirality_info.get("score") or 0.0
        fisher_rao = chirality_info.get("fisher_rao_distance")
        evidence_overlap = chirality_info.get("evidence_overlap")
        polarity_conflict = bool(chirality_info.get("polarity_conflict"))
        entailment_score = float(validation.get("entailment_score") or 0.0)
        citation_valid = validation.get("citation_valid", True)

        issues: List[Dict[str, Any]] = []
        severity = "LOW"

        # Citation validity check - CRITICAL issue, always HIGH severity
        if not citation_valid:
            issues.append(
                {
                    "issue_type": "CITATION_INVALID",
                    "details": {
                        "citation_valid": citation_valid,
                        "reason": "Model cited documents not present in source corpus",
                    },
                }
            )
            severity = "HIGH"

        if chirality_score >= self.config.chirality_threshold:
            issues.append(
                {
                    "issue_type": "POLARITY_CONTRADICTION",
                    "details": {
                        "chirality_score": chirality_score,
                        "fisher_rao_distance": fisher_rao,
                        "evidence_overlap": evidence_overlap,
                    },
                }
            )
            severity = max(severity, "MEDIUM", key=self._severity_rank)

        if polarity_conflict:
            issues.append(
                {
                    "issue_type": "POLARITY_CONFLICT",
                    "details": {"chirality_score": chirality_score},
                }
            )
            severity = max(severity, "HIGH", key=self._severity_rank)

        if evidence_overlap is not None and evidence_overlap >= self.config.evidence_overlap_threshold:
            if chirality_score >= self.config.high_chirality_threshold:
                severity = max(severity, "HIGH", key=self._severity_rank)

        if entailment_score < self.config.entailment_threshold:
            issues.append(
                {
                    "issue_type": "WEAK_ENTAILMENT",
                    "details": {"entailment_score": entailment_score},
                }
            )
            severity = max(severity, "MEDIUM", key=self._severity_rank)

        if not issues:
            return None

        return {
            "claim_id": entry.get("claim_id"),
            "severity": severity,
            "issues": issues,
            "metrics": {
                "chirality_score": chirality_score,
                "fisher_rao_distance": fisher_rao,
                "evidence_overlap": evidence_overlap,
                "polarity_conflict": polarity_conflict,
                "entailment_score": entailment_score,
                "citation_valid": citation_valid,
                "beta1": entry.get("beta1"),
            },
        }

    def _write_flags(self, flags: List[Dict[str, Any]]) -> None:
        output_path = self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for flag in flags:
                fh.write(json.dumps(flag) + "\n")

    @staticmethod
    def _severity_rank(value: str) -> int:
        ordering = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        return ordering.get(value.upper(), 0)
