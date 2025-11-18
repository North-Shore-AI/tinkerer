"""Extract HIGH severity claims for manual review."""

import json
from pathlib import Path

# Read evaluation data
eval_path = Path(__file__).parent.parent.parent.parent / "runs" / "thinker_eval" / "scifact_dev_eval.jsonl"
flags_path = Path(__file__).parent.parent.parent.parent / "runs" / "thinker_eval" / "scifact_dev_eval_antagonist_flags.jsonl"

# Load flags to find HIGH severity claim IDs
high_severity_ids = []
with flags_path.open() as f:
    for line in f:
        if line.strip():
            flag = json.loads(line)
            if flag['severity'] == 'HIGH':
                high_severity_ids.append(flag['claim_id'])

print(f"HIGH severity claim IDs: {high_severity_ids}\n")

# Extract full records for those claims
with eval_path.open() as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            if record['claim_id'] in high_severity_ids:
                print("=" * 80)
                print(f"CLAIM ID: {record['claim_id']}")
                print("=" * 80)

                print("\n--- PROMPT (truncated) ---")
                prompt_lines = record['prompt'].split('\n')
                # Print first 15 and last 5 lines
                for i, line in enumerate(prompt_lines[:15]):
                    print(line)
                if len(prompt_lines) > 20:
                    print(f"\n... [{len(prompt_lines) - 20} lines omitted] ...\n")
                for line in prompt_lines[-5:]:
                    print(line)

                print("\n--- COMPLETION ---")
                print(record['completion'])

                print("\n--- VALIDATION RESULTS ---")
                val = record['validation']
                print(f"  Schema valid:        {val['schema_valid']}")
                print(f"  Citation valid:      {val['citation_valid']}")
                print(f"  Entailment score:    {val['entailment_score']:.4f}")
                print(f"  Entailment pass:     {val['entailment_pass']}")
                print(f"  Semantic similarity: {val['semantic_similarity']:.4f}")
                print(f"  Similarity pass:     {val['similarity_pass']}")
                print(f"  Overall pass:        {val['overall_pass']}")

                print("\n--- TOPOLOGY METRICS ---")
                print(f"  β₁ (Betti):          {record['beta1']}")
                print(f"  Cycles:              {record['cycles']}")

                print("\n--- CHIRALITY METRICS ---")
                chir = record['chirality']
                print(f"  Chirality score:     {chir['score']:.4f}")
                print(f"  Fisher-Rao distance: {chir['fisher_rao_distance']:.4f}")
                print(f"  Evidence overlap:    {chir['evidence_overlap']:.4f}")
                print(f"  Polarity conflict:   {chir['polarity_conflict']}")

                print("\n")
