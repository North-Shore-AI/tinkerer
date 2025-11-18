"""Test suite for Antagonist runner."""

import json
import tempfile
from pathlib import Path

import pytest

from thinker.antagonist import AntagonistConfig, AntagonistRunner


class TestAntagonistConfig:
    """Test AntagonistConfig dataclass."""

    def test_default_thresholds(self):
        config = AntagonistConfig(
            input_path=Path("input.jsonl"), output_path=Path("output.jsonl")
        )
        assert config.chirality_threshold == 0.55
        assert config.high_chirality_threshold == 0.65
        assert config.entailment_threshold == 0.5
        assert config.evidence_overlap_threshold == 0.2

    def test_custom_thresholds(self):
        config = AntagonistConfig(
            input_path=Path("input.jsonl"),
            output_path=Path("output.jsonl"),
            chirality_threshold=0.6,
            high_chirality_threshold=0.7,
            entailment_threshold=0.75,
            evidence_overlap_threshold=0.3,
        )
        assert config.chirality_threshold == 0.6
        assert config.high_chirality_threshold == 0.7
        assert config.entailment_threshold == 0.75
        assert config.evidence_overlap_threshold == 0.3


class TestCitationInvalid:
    """Test CITATION_INVALID issue detection."""

    def test_flags_invalid_citation(self, tmp_path):
        """Test that invalid citations are flagged as HIGH severity."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # Create test data with invalid citation
        test_data = {
            "claim_id": 1,
            "validation": {
                "citation_valid": False,
                "entailment_score": 0.0,
            },
            "chirality": {"score": 0.5, "fisher_rao_distance": 10.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["flagged_records"] == 1

        # Read output
        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["claim_id"] == 1
        assert flag["severity"] == "HIGH"
        assert any(i["issue_type"] == "CITATION_INVALID" for i in flag["issues"])
        assert flag["metrics"]["citation_valid"] is False

    def test_passes_valid_citation(self, tmp_path):
        """Test that valid citations don't trigger CITATION_INVALID."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # Create test data with valid citation
        test_data = {
            "claim_id": 1,
            "validation": {
                "citation_valid": True,
                "entailment_score": 0.9,
            },
            "chirality": {"score": 0.3, "fisher_rao_distance": 5.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        # Should not flag anything (low chirality, high entailment, valid citation)
        assert summary["flagged_records"] == 0


class TestChiralityThresholds:
    """Test chirality-based issue detection."""

    def test_polarity_contradiction_at_threshold(self, tmp_path):
        """Test that chirality at threshold triggers POLARITY_CONTRADICTION."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.8, "citation_valid": True},
            "chirality": {
                "score": 0.55,  # Exactly at threshold
                "fisher_rao_distance": 15.0,
                "evidence_overlap": 0.1,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["severity"] == "MEDIUM"
        assert any(i["issue_type"] == "POLARITY_CONTRADICTION" for i in flag["issues"])

    def test_high_chirality_escalates_severity(self, tmp_path):
        """Test that high chirality + evidence overlap escalates to HIGH."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.8, "citation_valid": True},
            "chirality": {
                "score": 0.65,  # High threshold
                "fisher_rao_distance": 20.0,
                "evidence_overlap": 0.5,  # Above threshold
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["severity"] == "HIGH"

    def test_below_threshold_not_flagged(self, tmp_path):
        """Test that chirality below threshold doesn't trigger POLARITY_CONTRADICTION."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.8, "citation_valid": True},
            "chirality": {
                "score": 0.54,  # Below threshold
                "fisher_rao_distance": 10.0,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["flagged_records"] == 0


class TestEntailmentThresholds:
    """Test entailment-based issue detection."""

    def test_weak_entailment_at_threshold(self, tmp_path):
        """Test that entailment at threshold triggers WEAK_ENTAILMENT."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {
                "entailment_score": 0.49,  # Below threshold (0.5)
                "citation_valid": True,
            },
            "chirality": {"score": 0.3, "fisher_rao_distance": 5.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["severity"] == "MEDIUM"
        assert any(i["issue_type"] == "WEAK_ENTAILMENT" for i in flag["issues"])
        assert flag["metrics"]["entailment_score"] == 0.49

    def test_strong_entailment_not_flagged(self, tmp_path):
        """Test that strong entailment doesn't trigger WEAK_ENTAILMENT."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {
                "entailment_score": 0.9,  # Above threshold
                "citation_valid": True,
            },
            "chirality": {"score": 0.3, "fisher_rao_distance": 5.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["flagged_records"] == 0


class TestPolarityConflict:
    """Test polarity conflict detection."""

    def test_polarity_conflict_high_severity(self, tmp_path):
        """Test that polarity conflict triggers HIGH severity."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.5, "citation_valid": True},
            "chirality": {
                "score": 0.6,
                "fisher_rao_distance": 15.0,
                "polarity_conflict": True,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["severity"] == "HIGH"
        assert any(i["issue_type"] == "POLARITY_CONFLICT" for i in flag["issues"])


class TestMultipleIssues:
    """Test handling of multiple issues."""

    def test_multiple_issues_combined(self, tmp_path):
        """Test that multiple issues are all captured."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 133,  # Real example from HIGH severity review
            "validation": {
                "citation_valid": False,
                "entailment_score": 0.0,
            },
            "chirality": {
                "score": 0.6546,
                "fisher_rao_distance": 22.64,
                "evidence_overlap": 0.6,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Should have 3 issues: CITATION_INVALID, POLARITY_CONTRADICTION, WEAK_ENTAILMENT
        assert len(flag["issues"]) == 3
        issue_types = {i["issue_type"] for i in flag["issues"]}
        assert "CITATION_INVALID" in issue_types
        assert "POLARITY_CONTRADICTION" in issue_types
        assert "WEAK_ENTAILMENT" in issue_types

        # CITATION_INVALID should escalate to HIGH
        assert flag["severity"] == "HIGH"

    def test_citation_invalid_overrides_other_severity(self, tmp_path):
        """Test that CITATION_INVALID always results in HIGH severity."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {
                "citation_valid": False,
                "entailment_score": 0.9,  # Good entailment
            },
            "chirality": {
                "score": 0.3,  # Low chirality
                "fisher_rao_distance": 5.0,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Even with good metrics, citation invalid should force HIGH
        assert flag["severity"] == "HIGH"


class TestSeverityEscalation:
    """Test severity escalation logic."""

    def test_severity_rank_ordering(self):
        """Test that severity ranking is correct."""
        runner = AntagonistRunner(AntagonistConfig(Path("in"), Path("out")))
        assert runner._severity_rank("LOW") == 0
        assert runner._severity_rank("MEDIUM") == 1
        assert runner._severity_rank("HIGH") == 2
        assert runner._severity_rank("low") == 0
        assert runner._severity_rank("high") == 2

    def test_max_severity_selection(self, tmp_path):
        """Test that max severity is selected correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # MEDIUM (chirality) + MEDIUM (weak entailment) = MEDIUM
        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.3, "citation_valid": True},
            "chirality": {"score": 0.6, "fisher_rao_distance": 15.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        assert flag["severity"] == "MEDIUM"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self, tmp_path):
        """Test that empty input produces no flags."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # Create empty file
        input_file.touch()

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["total_records"] == 0
        assert summary["flagged_records"] == 0

    def test_missing_validation_dict(self, tmp_path):
        """Test handling of missing validation dict."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            # No validation dict
            "chirality": {"score": 0.6},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Should still flag based on chirality, with defaults for missing values
        assert flag["claim_id"] == 1
        assert "POLARITY_CONTRADICTION" in [i["issue_type"] for i in flag["issues"]]

    def test_missing_chirality_dict(self, tmp_path):
        """Test handling of missing chirality dict."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"entailment_score": 0.3, "citation_valid": True},
            # No chirality dict
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Should flag for weak entailment, not chirality
        assert flag["claim_id"] == 1
        assert "WEAK_ENTAILMENT" in [i["issue_type"] for i in flag["issues"]]
        assert "POLARITY_CONTRADICTION" not in [i["issue_type"] for i in flag["issues"]]

    def test_none_values(self, tmp_path):
        """Test handling of None values in metrics."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {
                "entailment_score": None,  # None instead of float
                "citation_valid": True,
            },
            "chirality": {
                "score": None,
                "fisher_rao_distance": None,
                "evidence_overlap": None,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Should convert None to 0.0 and flag for weak entailment
        assert flag["metrics"]["entailment_score"] == 0.0
        assert flag["metrics"]["chirality_score"] == 0.0


class TestBatchProcessing:
    """Test processing of multiple records."""

    def test_multiple_records(self, tmp_path):
        """Test that multiple records are processed correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        records = [
            # HIGH severity - citation invalid
            {
                "claim_id": 1,
                "validation": {"citation_valid": False, "entailment_score": 0.0},
                "chirality": {"score": 0.6},
                "beta1": 0,
            },
            # MEDIUM severity - weak entailment
            {
                "claim_id": 2,
                "validation": {"citation_valid": True, "entailment_score": 0.3},
                "chirality": {"score": 0.4},
                "beta1": 0,
            },
            # No issues
            {
                "claim_id": 3,
                "validation": {"citation_valid": True, "entailment_score": 0.9},
                "chirality": {"score": 0.3},
                "beta1": 0,
            },
        ]

        with input_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["total_records"] == 3
        assert summary["flagged_records"] == 2

        # Read all flags
        flags = []
        with output_file.open() as f:
            for line in f:
                flags.append(json.loads(line))

        assert len(flags) == 2
        assert flags[0]["claim_id"] == 1
        assert flags[0]["severity"] == "HIGH"
        assert flags[1]["claim_id"] == 2
        assert flags[1]["severity"] == "MEDIUM"

    def test_selective_flagging_rate(self, tmp_path):
        """Test that flagging rate matches expected patterns."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # Create 50 records mimicking SciFact distribution
        records = []
        for i in range(50):
            if i < 2:
                # 4% HIGH severity (citation invalid)
                record = {
                    "claim_id": i,
                    "validation": {"citation_valid": False, "entailment_score": 0.0},
                    "chirality": {"score": 0.65, "evidence_overlap": 0.5},
                    "beta1": 0,
                }
            elif i < 46:
                # 88% MEDIUM severity (various issues)
                record = {
                    "claim_id": i,
                    "validation": {"citation_valid": True, "entailment_score": 0.4},
                    "chirality": {"score": 0.6},
                    "beta1": 0,
                }
            else:
                # 8% no issues
                record = {
                    "claim_id": i,
                    "validation": {"citation_valid": True, "entailment_score": 0.9},
                    "chirality": {"score": 0.3},
                    "beta1": 0,
                }
            records.append(record)

        with input_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        summary = runner.run()

        assert summary["total_records"] == 50
        assert summary["flagged_records"] == 46  # 92% flagging rate

        # Count severities
        flags = []
        with output_file.open() as f:
            for line in f:
                flags.append(json.loads(line))

        high_count = sum(1 for f in flags if f["severity"] == "HIGH")
        medium_count = sum(1 for f in flags if f["severity"] == "MEDIUM")

        assert high_count == 2
        assert medium_count == 44


class TestOutputFormat:
    """Test output format compliance."""

    def test_output_jsonl_format(self, tmp_path):
        """Test that output is valid JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 1,
            "validation": {"citation_valid": True, "entailment_score": 0.3},
            "chirality": {"score": 0.6, "fisher_rao_distance": 15.0},
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        # Verify valid JSONL
        with output_file.open() as f:
            for line in f:
                flag = json.loads(line)  # Should not raise
                assert "claim_id" in flag
                assert "severity" in flag
                assert "issues" in flag
                assert "metrics" in flag

    def test_output_schema(self, tmp_path):
        """Test that output has required fields."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        test_data = {
            "claim_id": 42,
            "validation": {"citation_valid": False, "entailment_score": 0.0},
            "chirality": {
                "score": 0.65,
                "fisher_rao_distance": 20.0,
                "evidence_overlap": 0.6,
                "polarity_conflict": False,
            },
            "beta1": 0,
        }

        with input_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        config = AntagonistConfig(input_path=input_file, output_path=output_file)
        runner = AntagonistRunner(config)
        runner.run()

        with output_file.open() as f:
            flag = json.loads(f.readline())

        # Required top-level fields
        assert flag["claim_id"] == 42
        assert flag["severity"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(flag["issues"], list)
        assert isinstance(flag["metrics"], dict)

        # Required metric fields
        metrics = flag["metrics"]
        assert "chirality_score" in metrics
        assert "fisher_rao_distance" in metrics
        assert "evidence_overlap" in metrics
        assert "polarity_conflict" in metrics
        assert "entailment_score" in metrics
        assert "citation_valid" in metrics
        assert "beta1" in metrics

        # Issue structure
        for issue in flag["issues"]:
            assert "issue_type" in issue
            assert "details" in issue
