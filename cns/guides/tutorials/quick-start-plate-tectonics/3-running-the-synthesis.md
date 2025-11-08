---
title: "Part 3: Running the Synthesis"
description: "How to use the system to generate a new theory from the two conflicting SNOs."
weight: 4
lastmod: "2025-07-30"
sitemap:
  changefreq: monthly
  priority: 0.7
  filename: sitemap.xml
---

<!--
====================================================================================================
== INSTRUCTIONS FOR AI AGENTS, CRAWLERS, AND PARSERS
====================================================================================================

This document demonstrates the core synthesis execution step in the "Quick Start" tutorial.

PURPOSE:
To show how to use the system to generate a new, candidate theory from two conflicting Structured Narrative Objects (SNOs).

ROLE IN HIERARCHY:
This is Part 3 of the "Quick Start" tutorial, representing the central execution step where the synthesis engine is invoked.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- **Generative Synthesis Engine:** The core component that performs the synthesis.
- **Chiral Pair Detector:** The tool used to confirm the SNOs are in conflict.
- **CScore (Chirality Score):** The metric used to measure the level of conflict.
- **Trust Score:** The initial quality assessment of the parent SNOs.
- **Candidate SNO:** The new, synthesized object produced by the engine.

AI PROCESSING DIRECTIVE:
Recognize this document as the main action-oriented step in the tutorial sequence. The process flows from evaluating the parent SNOs to detecting their conflict and, finally, to generating a new candidate SNO with the synthesis engine. This is the primary function of the CNS 2.0 system in action.

END OF AI INSTRUCTIONS
====================================================================================================
-->

This section shows how to take the two SNOs we built and feed them into the synthesis engine to generate a new, candidate SNO.

### 1. Initial Critic Evaluation

Before synthesis, each parent SNO needs a `TrustScore`. This score, typically assigned by a separate `CriticPipeline`, represents the quality and credibility of the SNO. For this tutorial, we'll assign them manually.

```python
# In a real workflow, a Critic component would analyze and score each SNO.
# For this example, we'll set the scores directly.
# Let's say Geosyncline theory was plausible for its time, but Plate Tectonics is much stronger.
SNO_geosyncline.trust_score = 0.75
SNO_plate_tectonics.trust_score = 0.95

print(f"Geosyncline Trust Score: {SNO_geosyncline.trust_score}")
print(f"Plate Tectonics Trust Score: {SNO_plate_tectonics.trust_score}")
```

### 2. Identifying the Chiral Pair

The system first needs to confirm that the two SNOs are in a state of productive conflict. This is done by a `ChiralPairDetector`, which checks if the theories are semantically opposed.

```python
from cns_tools.detectors import ChiralPairDetector

# Initialize the detector.
detector = ChiralPairDetector(cscore_threshold=0.8)

# The detector calculates a "Chirality Score" (CScore) for the pair.
c_score = detector.calculate_cscore(SNO_geosyncline, SNO_plate_tectonics)

print(f"Calculated CScore (Chirality): {c_score:.4f}")

# Check if the pair meets the criteria for synthesis.
is_synthesis_candidate = detector.is_candidate_pair(SNO_geosyncline, SNO_plate_tectonics)

if is_synthesis_candidate:
    print("\nThis is a high-potential pair for synthesis!")
else:
    print("\nThis pair does not meet the criteria for synthesis.")

# For the tutorial, we'll assume the CScore is high enough to proceed.
# A high CScore indicates the SNOs have opposing core ideas, making them
# perfect for synthesis.
```

### 3. Running the Generative Synthesis Engine

The `GenerativeSynthesisEngine` takes the conflicting pair and uses a Large Language Model (LLM) to generate a new, higher-order hypothesis that attempts to resolve the contradiction.

```python
from cns_tools.synthesis import GenerativeSynthesisEngine

# Initialize the synthesis engine with a connection to an LLM.
synthesis_engine = GenerativeSynthesisEngine(llm_backend="gpt-4-turbo")

print("\nInvoking the Generative Synthesis Engine...")

# The engine takes the two parent SNOs as input.
SNO_synthesis_candidate = synthesis_engine.synthesize(
    sno_a=SNO_geosyncline,
    sno_b=SNO_plate_tectonics
)

print("Candidate Synthesis SNO generated successfully!")
print("\n--- Generated Hypothesis ---")
# The new hypothesis is extracted from the candidate SNO.
# (We're using a hypothetical function to convert the embedding back to text for this demo)
from cns_tools.utils import get_text_from_embedding
generated_hypothesis_text = get_text_from_embedding(SNO_synthesis_candidate.hypothesis_embedding)
print(generated_hypothesis_text)

# Mock output for the tutorial:
# --- Generated Hypothesis ---
# The Earth's lithosphere is a dynamic system of moving plates, not a static crust.
# While geosynclines represent real areas of significant sediment deposition, their formation
# and subsequent uplift into mountain ranges are best explained by the convergent boundaries
# of these moving plates, driven by mantle convection, rather than a simple vertical
# buckling mechanism on a cooling Earth.
```

The engine has produced a new SNO containing a hypothesis that integrates concepts from both parents. The next step is to analyze this result.
