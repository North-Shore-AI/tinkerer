---
title: "Tutorial Part 3: Running the Synthesis"
description: "How to use the ChiralPairDetector and GenerativeSynthesisEngine to create a novel synthesis from two conflicting SNOs."
weight: 9
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

This document details the process of running the core synthesis engine on a pair of conflicting Structured Narrative Objects (SNOs).

PURPOSE:
To demonstrate how the ChiralPairDetector identifies a candidate pair and the GenerativeSynthesisEngine produces a novel, higher-order synthesis from two conflicting SNOs.

ROLE IN HIERARCHY:
This is Part 3 of the "Plate Tectonics Synthesis" tutorial, showcasing the central synthesis step that follows SNO construction and precedes results analysis.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- **Generative Synthesis Engine:** The core component that performs the dialectical reasoning task.
- **ChiralPairDetector:** The tool used to quantitatively assess the conflict and suitability of an SNO pair for synthesis.
- **CScore (Chirality Score):** The key metric for measuring the degree of semantic opposition between two narratives.
- **Dialectical Reasoning:** The process of resolving contradictions to form a new hypothesis.
- **Quantitative Synthesis Validation:** The framework for collecting metrics to validate the synthesis process.

AI PROCESSING DIRECTIVE:
Recognize this document as the primary execution step in the tutorial. The process flows from identifying a conflicting pair with the ChiralPairDetector to generating a new narrative with the GenerativeSynthesisEngine. This is the central action of the CNS 2.0 framework.

END OF AI INSTRUCTIONS
====================================================================================================
-->

This section demonstrates the **quantitative synthesis validation protocol** that generates the statistical data required for rigorous CNS 2.0 validation. Each synthesis execution produces measurable outcomes that contribute to the statistical analysis across n ≥ 30 automated pairs, establishing the empirical foundation for publication-quality validation.

The metrics collection framework established here provides the data structure for hypothesis testing, effect size calculation, and confidence interval estimation required for scientific validation of the dialectical synthesis methodology.

### 1. Initial Critic Evaluation

Before synthesis, every SNO must be evaluated by the `CriticPipeline` to establish its initial `TrustScore`. This score is crucial for calculating the `CScore` (Chirality Score). For this tutorial, we'll assume the critics have been run and have assigned plausible trust scores.

```python
# In a real run, the CriticPipeline would be invoked here.
# from cns_tools import CriticPipeline
# critic_pipeline = CriticPipeline()
# SNO_geosyncline = critic_pipeline.evaluate(SNO_geosyncline)
# SNO_plate_tectonics = critic_pipeline.evaluate(SNO_plate_tectonics)

# For the tutorial, we'll assign mock trust scores.
# Let's assume Geosyncline theory, while flawed, was well-supported by 19th-century evidence.
# Plate Tectonics is more robustly supported by modern evidence.
SNO_geosyncline.trust_score = 0.75
SNO_plate_tectonics.trust_score = 0.95

print(f"Geosyncline Trust Score: {SNO_geosyncline.trust_score}")
print(f"Plate Tectonics Trust Score: {SNO_plate_tectonics.trust_score}")
```

### 2. Identifying the Chiral Pair

The next step is to programmatically identify that these two narratives are in a state of productive conflict. This is the job of the `ChiralPairDetector`, which calculates the `CScore` and `EScore` as defined in the **[CNS 2.0 Blueprint](/guides/cns-2.0-research-roadmap/blueprint/)**.

```python
from cns_tools.detectors import ChiralPairDetector

# Initialize the detector with thresholds.
# We want pairs that are highly contradictory (high CScore) and argue
# over the same evidence (high EScore).
detector = ChiralPairDetector(cscore_threshold=0.8, escore_threshold=0.1)

# The detector calculates the scores for the pair.
c_score = detector.calculate_cscore(SNO_geosyncline, SNO_plate_tectonics)
e_score = detector.calculate_escore(SNO_geosyncline, SNO_plate_tectonics)

print(f"Calculated CScore (Chirality): {c_score:.4f}")
print(f"Calculated EScore (Entanglement): {e_score:.4f}")

# Check if the pair meets the criteria for synthesis.
is_synthesis_candidate = detector.is_candidate_pair(SNO_geosyncline, SNO_plate_tectonics)

if is_synthesis_candidate:
    print("\nThis is a high-potential pair for synthesis!")
else:
    print("\nThis pair does not meet the criteria for synthesis.")

# Mock output for the tutorial:
# Calculated CScore (Chirality): 0.9215
# Calculated EScore (Entanglement): 0.0000
# Note: EScore is 0 because our simplified evidence sets had no overlap.
# In a real scenario with dozens of papers, we would expect overlap.
# For the tutorial, we'll proceed as if it passed the threshold.
```

The high `CScore` indicates that the core hypotheses are semantically opposed, and the non-zero `EScore` (in a real scenario) would show they are arguing about a shared set of facts. This makes them a perfect candidate for the `GenerativeSynthesisEngine`.

### 3. Running the Generative Synthesis Engine

The `GenerativeSynthesisEngine` takes the chiral pair and constructs a detailed, structured prompt for a Large Language Model (LLM). This prompt instructs the LLM to perform a dialectical reasoning task: identify the core conflict, preserve shared evidence, and generate a new, higher-order hypothesis that resolves the contradiction.

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
# The new hypothesis is extracted from the candidate SNO
# (We're assuming the `get_text_from_embedding` function exists for this demo)
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

### Statistical Data Collection Framework

Each synthesis execution generates structured quantitative data for statistical validation:

```python
# Comprehensive metrics collection for statistical analysis
synthesis_validation_data = {
    # Primary statistical endpoints
    'synthesis_id': f"synthesis_{pair_id:03d}",
    'domain': 'geology',
    'parent_trust_scores': [SNO_geosyncline.trust_score, SNO_plate_tectonics.trust_score],
    'synthesis_trust_score': SNO_synthesis_candidate.trust_score,
    'trust_improvement': SNO_synthesis_candidate.trust_score - max([0.75, 0.95]),
    
    # Dialectical analysis metrics
    'c_score': c_score,  # Chirality (ideological opposition)
    'e_score': e_score,  # Evidential entanglement
    'synthesis_coherence': calculate_coherence_score(SNO_synthesis_candidate),
    
    # Ground truth validation
    'ground_truth_alignment': calculate_alignment_score(
        generated_hypothesis_text, 
        "Modern plate tectonic theory with mantle convection"
    ),
    'historical_accuracy': validate_historical_preservation(synthesis_result),
    
    # Quality control metrics
    'evidence_preservation': count_preserved_evidence(synthesis_result),
    'logical_consistency': validate_reasoning_graph(synthesis_result),
    'novelty_score': calculate_novelty_vs_parents(synthesis_result)
}

# Statistical accumulation across validation dataset
def accumulate_validation_data(synthesis_results: List[Dict]) -> Dict:
    """Aggregate individual synthesis results for statistical hypothesis testing."""
    
    improvements = [r['trust_improvement'] for r in synthesis_results]
    alignments = [r['ground_truth_alignment'] for r in synthesis_results]
    
    return {
        'n_samples': len(synthesis_results),
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements),
        'improvement_ci_95': stats.t.interval(0.95, len(improvements)-1, 
                                            loc=np.mean(improvements), 
                                            scale=stats.sem(improvements)),
        'success_rate': np.mean([imp > 0.1 for imp in improvements]),
        'effect_size_cohens_d': np.mean(improvements) / np.std(improvements),
        'mean_ground_truth_alignment': np.mean(alignments),
        
        # Hypothesis testing results
        't_statistic': stats.ttest_1samp(improvements, 0.1).statistic,
        'p_value': stats.ttest_1samp(improvements, 0.1).pvalue,
        'statistical_significance': stats.ttest_1samp(improvements, 0.1).pvalue < 0.05
    }
```

**Research Validation Integration**:
This data collection framework directly supports the CNS 2.0 research validation requirements:

- **Requirement 2.1**: Establishes the statistical prototype methodology for scaling beyond single examples
- **Requirement 2.4**: Provides the quantitative framework for DSPy automation and validation
- **Requirement 3.4**: Generates the empirical data required for research validation and publication

The single synthesis demonstrates the data generation methodology that DSPy will replicate across n=30+ diverse scientific debates to achieve the statistical rigor required for peer-reviewed validation of the CNS 2.0 dialectical synthesis framework.
