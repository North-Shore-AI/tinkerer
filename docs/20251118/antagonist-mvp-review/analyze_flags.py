"""Analyze Antagonist flags from the latest evaluation run."""

import json
from collections import Counter
from pathlib import Path
import sys


def analyze_flags(flags_path: Path):
    """Analyze antagonist flags and generate report."""

    if not flags_path.exists():
        print(f"Error: {flags_path} not found")
        return

    with flags_path.open() as f:
        flags = [json.loads(line) for line in f if line.strip()]

    print("=" * 80)
    print("ANTAGONIST FLAG ANALYSIS")
    print("=" * 80)
    print(f"\nSource: {flags_path}")
    print(f"Total flags: {len(flags)}")

    # Issue type distribution
    issues = [i['issue_type'] for flag in flags for i in flag['issues']]
    print("\n" + "-" * 80)
    print("ISSUE TYPE DISTRIBUTION")
    print("-" * 80)
    issue_counts = Counter(issues)
    for issue_type, count in issue_counts.most_common():
        pct = (count / len(flags)) * 100
        print(f"  {issue_type:30s} {count:4d} ({pct:5.1f}%)")

    # Severity distribution
    print("\n" + "-" * 80)
    print("SEVERITY DISTRIBUTION")
    print("-" * 80)
    severities = [flag['severity'] for flag in flags]
    severity_counts = Counter(severities)
    for severity, count in sorted(severity_counts.items(), key=lambda x: {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}[x[0]]):
        pct = (count / len(flags)) * 100
        print(f"  {severity:10s} {count:4d} ({pct:5.1f}%)")

    # High-severity flags (manual review required)
    print("\n" + "-" * 80)
    print("HIGH SEVERITY FLAGS (MANUAL REVIEW REQUIRED)")
    print("-" * 80)
    high = [f for f in flags if f['severity'] == 'HIGH']
    if high:
        print(f"  Found {len(high)} HIGH severity flags:")
        for flag in high:
            print(f"    claim_id: {flag['claim_id']}")
            for issue in flag['issues']:
                print(f"      - {issue['issue_type']}: {issue['details']}")
    else:
        print("  None found ✅")

    # Metric statistics
    print("\n" + "-" * 80)
    print("METRIC STATISTICS")
    print("-" * 80)

    entailments = [f['metrics']['entailment_score'] for f in flags if f['metrics'].get('entailment_score') is not None]
    if entailments:
        print(f"  Entailment scores (n={len(entailments)}):")
        print(f"    min:  {min(entailments):.4f}")
        print(f"    max:  {max(entailments):.4f}")
        print(f"    mean: {sum(entailments)/len(entailments):.4f}")
        print(f"    <0.5 (weak): {sum(1 for e in entailments if e < 0.5)} ({sum(1 for e in entailments if e < 0.5)/len(entailments)*100:.1f}%)")

    chiralities = [f['metrics']['chirality_score'] for f in flags if f['metrics'].get('chirality_score') is not None]
    if chiralities:
        print(f"\n  Chirality scores (n={len(chiralities)}):")
        print(f"    min:  {min(chiralities):.4f}")
        print(f"    max:  {max(chiralities):.4f}")
        print(f"    mean: {sum(chiralities)/len(chiralities):.4f}")
        print(f"    ≥0.55 (trigger): {sum(1 for c in chiralities if c >= 0.55)} ({sum(1 for c in chiralities if c >= 0.55)/len(chiralities)*100:.1f}%)")
        print(f"    ≥0.65 (high):    {sum(1 for c in chiralities if c >= 0.65)} ({sum(1 for c in chiralities if c >= 0.65)/len(chiralities)*100:.1f}%)")

    fisher_raos = [f['metrics']['fisher_rao_distance'] for f in flags if f['metrics'].get('fisher_rao_distance') is not None]
    if fisher_raos:
        print(f"\n  Fisher-Rao distances (n={len(fisher_raos)}):")
        print(f"    min:  {min(fisher_raos):.4f}")
        print(f"    max:  {max(fisher_raos):.4f}")
        print(f"    mean: {sum(fisher_raos)/len(fisher_raos):.4f}")

    # Claim IDs with multiple issues
    print("\n" + "-" * 80)
    print("CLAIMS WITH MULTIPLE ISSUES")
    print("-" * 80)
    multi_issue = [f for f in flags if len(f['issues']) > 1]
    print(f"  Claims with >1 issue: {len(multi_issue)} ({len(multi_issue)/len(flags)*100:.1f}%)")
    if multi_issue:
        issue_combo_counts = Counter([tuple(sorted(i['issue_type'] for i in f['issues'])) for f in multi_issue])
        print("  Most common combinations:")
        for combo, count in issue_combo_counts.most_common(5):
            print(f"    {' + '.join(combo):60s} {count:3d}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    weak_entailment_pct = sum(1 for e in entailments if e < 0.5) / len(entailments) * 100 if entailments else 0
    if weak_entailment_pct > 50:
        print("  ⚠️  CRITICAL: {:.1f}% of flags have weak entailment (<0.5)".format(weak_entailment_pct))
        print("      → This confirms the Proposer semantic quality issue")
        print("      → Priority: Fix Proposer training (increase CNS_CLAIM_EVIDENCE_WEIGHT to 2.0+)")

    if len(high) == 0:
        print("  ✅ No HIGH severity flags - thresholds may be too lenient")
        print("      → Consider lowering high_chirality_threshold from 0.65 to 0.60")

    if len(multi_issue) / len(flags) > 0.5:
        print("  ℹ️  {:.1f}% of flags have multiple issues".format(len(multi_issue) / len(flags) * 100))
        print("      → This is expected when Proposer has systemic quality issues")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    # Default path
    default_path = Path(__file__).parent.parent.parent.parent / "runs" / "thinker_eval" / "scifact_dev_eval_antagonist_flags.jsonl"

    if len(sys.argv) > 1:
        flags_path = Path(sys.argv[1])
    else:
        flags_path = default_path

    analyze_flags(flags_path)
