---
title: "2. Defining the Task for DSPy"
description: "A code-heavy guide to setting up the core DSPy components: the Signature, the Metric, and the training Examples."
weight: 3
lastmod: "2025-07-30"
sitemap:
  changefreq: monthly
  priority: 0.6
---

<!--
====================================================================================================
== INSTRUCTIONS FOR AI AGENTS, CRAWLERS, AND PARSERS
====================================================================================================

PURPOSE:
This document provides a code-heavy, practical guide to defining a task for the DSPy framework by setting up its three core components: the Signature, the Metric, and the training Examples.

ROLE IN HIERARCHY:
This is the second, implementation-focused step in the "DSPy Self-Optimization" tutorial. It translates the concepts from the introduction into concrete Python code.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- DSPy Signature
- DSPy Metric
- DSPy Examples
- Task Definition
- Critic Pipeline Integration

AI PROCESSING DIRECTIVE:
Recognize this file as a sequential step in a tutorial. The code blocks are central to this document and demonstrate the practical setup required before running the DSPy optimizer.

END OF AI INSTRUCTIONS
====================================================================================================
-->

Before we can optimize our synthesis module, we need to formally define the task for DSPy. This involves three key components:

1.  **The Signature:** Defines the inputs and outputs of our task.
2.  **The Metric:** A function that scores how "good" a generated output is.
3.  **The Examples:** A small training set of high-quality input/output pairs.

Let's walk through the code for each.

### 1. The Signature: `ChiralPairToSynthesis`

A DSPy `Signature` is a declarative specification of what our module needs to do. For our task, we want to take two opposing narratives and their shared evidence, and produce a new, synthesized hypothesis.

We can define this in a simple Python class. The docstring is important, as DSPy uses it to guide the LLM.

```python
import dspy

class ChiralPairToSynthesis(dspy.Signature):
    """
    Synthesizes a novel, higher-order hypothesis from two opposing narratives (a thesis and an antithesis) that are grounded in a shared set of evidence.
    The synthesis must reconcile the conflict and explain the same evidence.
    """
    
    # Input Fields
    thesis = dspy.InputField(desc="The central claim of the first narrative.")
    antithesis = dspy.InputField(desc="The central claim of the opposing narrative.")
    shared_evidence = dspy.InputField(desc="A summary of the key evidence that both narratives attempt to explain.")
    
    # Output Field
    synthesized_hypothesis = dspy.OutputField(desc="A novel hypothesis that resolves the core contradiction between the thesis and antithesis.")

```
This signature clearly tells the LLM what its inputs (`thesis`, `antithesis`, `shared_evidence`) and expected output (`synthesized_hypothesis`) are, along with a description of the overall goal.

### 2. The Metric: The `CriticPipelineMetric`

This is the most crucial component for integrating DSPy with CNS 2.0. The metric is how we teach DSPy what "good" looks like. Instead of relying on simple string matching (like BLEU or ROUGE), we will use our own **CNS Critic Pipeline** as the quality score.

For this tutorial, we'll simulate the critic pipeline. In a real implementation, this function would call the actual Grounding, Logic, and Novelty critics described in the **[Developer's Guide](/guides/building-cns-2.0-developers-guide/chapter-3-critic-pipeline/)**. The metric must return a score, typically between 0.0 (bad) and 1.0 (good).

```python
# In a real system, this would import and call the actual CNS critic modules.
# For this tutorial, we simulate them.
def simulate_cns_critic_pipeline(hypothesis: str, evidence: str) -> float:
    """
    Simulates the CNS critic pipeline, returning a score from 0.0 to 1.0.
    A real implementation would be much more complex.
    """
    score = 0.0
    # Grounding: Does the hypothesis seem plausible given the evidence?
    if "reconciles" in hypothesis.lower() and "plate tectonics" in evidence.lower():
        score += 0.4
    # Logic: Is the hypothesis internally consistent? (Simple check)
    if len(hypothesis.split()) > 10 and len(hypothesis.split()) < 50:
        score += 0.3
    # Novelty: Is it more than just a simple average of the inputs?
    if "new model" in hypothesis.lower() or "unifying theory" in hypothesis.lower():
        score += 0.3
    return min(score, 1.0) # Ensure score is max 1.0

def critic_pipeline_metric(gold, pred, trace=None):
    """
    A DSPy-compatible metric that uses our simulated CNS critic pipeline.
    'gold' is the dspy.Example object, 'pred' is the module's prediction.
    """
    # We get the inputs from the gold standard example
    thesis = gold.thesis
    antithesis = gold.antithesis
    shared_evidence = gold.shared_evidence
    
    # The prediction object contains the generated output
    synthesized_hypothesis = pred.synthesized_hypothesis
    
    # We run our critic pipeline on the *generated* hypothesis
    score = simulate_cns_critic_pipeline(synthesized_hypothesis, shared_evidence)
    
    # The metric should ideally return True for success, False for failure.
    # We'll define success as a score > 0.8
    return score > 0.8

```

This metric acts as the bridge between DSPy's optimization process and our system's own definition of quality. DSPy will learn to generate prompts that produce hypotheses earning a high score from our critic.

### 3. The Examples: Our Training Set

Finally, we need a small training set of high-quality examples. These are `dspy.Example` objects that conform to our `ChiralPairToSynthesis` signature. A good example provides a clear demonstration of the kind of reasoning we want the system to perform.

```python
# Our training set of 3 high-quality examples
trainset = [
    dspy.Example(
        thesis="The continents are fixed in place and ocean basins are permanent features, with mountains forming from vertical uplift.",
        antithesis="The continents drift across the Earth's surface, colliding to form mountains and creating new ocean basins.",
        shared_evidence="Shared evidence includes the jigsaw-puzzle fit of continents like Africa and South America, the presence of identical fossil species on widely separated continents, and the discovery of mid-ocean ridges.",
        synthesized_hypothesis="A unifying theory of plate tectonics reconciles these views: The Earth's lithosphere is divided into rigid plates that move. Continental drift is the result of this plate motion. Mountains form at convergent boundaries, and new ocean crust is created at divergent boundaries like mid-ocean ridges."
    ).with_inputs('thesis', 'antithesis', 'shared_evidence'),
    
    dspy.Example(
        thesis="Light is composed of particles (corpuscles) that travel in straight lines, which explains reflection.",
        antithesis="Light is a wave that propagates through an ethereal medium, which explains diffraction and interference.",
        shared_evidence="Shared evidence includes the observation that light travels in straight lines (forming shadows), reflects off surfaces, and also exhibits diffraction and interference patterns.",
        synthesized_hypothesis="A new model of wave-particle duality reconciles the conflict: Light exhibits properties of both waves and particles. It propagates as an electromagnetic wave but interacts with matter as discrete packets of energy called photons."
    ).with_inputs('thesis', 'antithesis', 'shared_evidence'),

    dspy.Example(
        thesis="Evolution occurs through the inheritance of acquired characteristics, where traits developed during an organism's life are passed to offspring.",
        antithesis="Evolution occurs through natural selection, where random variations that improve survival are preferentially passed to offspring.",
        shared_evidence="Shared evidence includes the observation of adaptation in species, the existence of vestigial structures, and the fossil record showing gradual change over time.",
        synthesized_hypothesis="The modern evolutionary synthesis reconciles these ideas: Natural selection acts upon genetic variations (mutations) that occur randomly. Acquired characteristics are not inherited, but the genetic potential for adaptation is passed down, providing the raw material for selection."
    ).with_inputs('thesis', 'antithesis', 'shared_evidence')
]
```
With our `Signature`, `Metric`, and `Examples` defined, we now have a fully specified task. In the next section, we will feed these components to the DSPy compiler to automatically generate an optimized synthesis prompt.
