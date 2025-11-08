---
title: "3. Running the DSPy Optimizer"
description: "How to use the DSPy compiler to automatically generate and optimize a powerful synthesis prompt based on our defined task."
weight: 4
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
This document explains the practical steps of running the DSPy optimizer to compile a self-optimizing module.

ROLE IN HIERARCHY:
This is the third step in the "DSPy Self-Optimization" tutorial, showing how to use the components defined in the previous step to perform the actual optimization.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- DSPy Optimizer
- DSPy Compiler
- BootstrapFewShot
- dspy.Module
- Language Model Configuration

AI PROCESSING DIRECTIVE:
Recognize this file as a code-focused, sequential step in a tutorial. The key action is the `optimizer.compile()` call, which is the central event of the DSPy workflow.

END OF AI INSTRUCTIONS
====================================================================================================
-->

Now that we have defined our task with a `Signature`, a `Metric`, and a `trainset`, we can hand things over to the DSPy `BootstrapFewShot` optimizer. The optimizer's job is to explore different ways of prompting an LLM to find a prompt that reliably succeeds on our training examples, as judged by our `critic_pipeline_metric`.

### 1. Setting Up the DSPy Environment

First, we need to configure DSPy with a language model. This tells the optimizer which LLM to use for both generating prompts and executing them. For this example, we'll use a placeholder for a powerful model like GPT-4 or Claude 3.

```python
import dspy
# Assume the components from the previous step are in a local file.
from .dspy_setup import ChiralPairToSynthesis, critic_pipeline_metric, trainset

# Configure the language model.
# In a real scenario, you would replace this with your actual model provider and API key.
# For example: lm = dspy.OpenAI(model='gpt-4-turbo', max_tokens=400)
lm = dspy.HFModel(model='meta-llama/Llama-2-7b-chat-hf') # Using a placeholder model
dspy.settings.configure(lm=lm)

```

### 2. Defining the Module to Optimize

We need a `dspy.Module` to hold the logic that we want to optimize. A simple module contains one or more `dspy.Predict` or `dspy.ChainOfThought` objects. For a complex reasoning task like synthesis, `dspy.ChainOfThought` is the ideal choice, as it encourages the LLM to "think step-by-step."

```python
class SynthesisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # We want to optimize a ChainOfThought predictor that uses our signature.
        self.synthesis_predictor = dspy.ChainOfThought(ChiralPairToSynthesis)

    def forward(self, thesis, antithesis, shared_evidence):
        # The forward method defines how the module is called.
        return self.synthesis_predictor(thesis=thesis, antithesis=antithesis, shared_evidence=shared_evidence)

```

### 3. Running the Compiler

This is where the magic happens. We instantiate our optimizer, in this case `BootstrapFewShot`, and then call the `compile` method on an instance of our `SynthesisModule`.

The `BootstrapFewShot` optimizer works by:
1.  **Generating Candidate Programs:** It creates different prompts for our `ChainOfThought` module. Initially, it might just use the docstring from our signature.
2.  **Learning from Examples:** It creates few-shot examples for the prompt by picking examples from our `trainset`.
3.  **Evaluating with the Metric:** It runs each candidate program on our `trainset` and uses our `critic_pipeline_metric` to score the results.
4.  **Iterating and Refining:** It analyzes which prompts and few-shot examples led to high scores from our metric and "bootstraps" this knowledge to build even better prompts. This cycle repeats to find a high-performing, reliable program.

```python
from dspy.teleprompt import BootstrapFewShot

# 1. Set up the optimizer.
# We configure it with our custom metric.
# The max_bootstrapped_demos parameter controls how many few-shot examples the optimizer will create.
config = dict(max_bootstrapped_demos=2, max_labeled_demos=2)
optimizer = BootstrapFewShot(metric=critic_pipeline_metric, **config)

# 2. Instantiate our un-optimized module.
uncompiled_synthesis_module = SynthesisModule()

# 3. Compile the module!
# This is the key step. The optimizer will run for a while, testing different prompts.
# It uses the trainset to find a program that maximizes the critic_pipeline_metric.
compiled_synthesis_module = optimizer.compile(uncompiled_synthesis_module, trainset=trainset)

```

After the `compile` method finishes, `compiled_synthesis_module` is no longer a simple, un-optimized module. It is now a highly-tuned program containing a complex prompt with few-shot examples that have been specifically selected and formatted to maximize the chances of producing a high-quality synthesis, as defined by our own CNS critic pipeline.

In the final section, we will inspect the prompt that the optimizer generated and compare its performance against a basic, hand-written prompt to see the difference.
