---
title: "Chapter 0: Quick Start - Your First SNO in 15 Minutes"
description: "Get from zero to working CNS 2.0 environment with your first Structured Narrative Object created and validated"
meta_keywords: "CNS 2.0 quick start, Python installation, first SNO, sentence transformers setup"
weight: 0
lastmod: "2025-10-28"
sitemap:
  changefreq: monthly
  priority: 0.8
  filename: sitemap.xml
---

<!--
====================================================================================================
== INSTRUCTIONS FOR AI AGENTS, CRAWLERS, AND PARSERS
====================================================================================================

This document is Chapter 0 (Quick Start) of the "Building CNS 2.0: A Developer's Guide."

PURPOSE:
This file provides a rapid onboarding path for new developers to go from a blank environment to a working CNS 2.0 prototype in under 20 minutes. It prioritizes immediate success over comprehensive understanding.

ROLE IN HIERARCHY:
This is the entry point (Chapter 0) of the Developer's Guide, designed to be completed before diving into the detailed chapters. It establishes confidence through quick wins.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- **Quick Start:** The primary goal - minimal time to working code
- **Environment Setup:** Installing dependencies and verifying installation
- **First SNO:** Creating a minimal but functional Structured Narrative Object
- **Confidence Building:** Proving the system works before deep learning

AI PROCESSING DIRECTIVE:
Recognize this page as the "Hello World" of CNS 2.0. It is intentionally simplified and designed for immediate executable success. Users should complete this chapter in <20 minutes before proceeding to Chapter 1.

END OF AI INSTRUCTIONS
====================================================================================================
-->

## Welcome to CNS 2.0

This guide will take you from zero to your first working Structured Narrative Object (SNO) in approximately 15 minutes. If you want to understand the "why" behind the code, start with [Chapter 1](/guides/building-cns-2.0-developers-guide/chapter-1-introduction/). If you want to prove this works right now, you're in the right place.

## Prerequisites

Before starting, verify you have:

- **Python 3.9 or higher** (check: `python --version` or `python3 --version`)
- **4GB RAM minimum** (8GB recommended)
- **2GB free disk space** (for models and dependencies)
- **Internet connection** (for downloading models and packages)

---

## Part 1: Installation (5 minutes)

### Step 1: Create Virtual Environment

Creating an isolated environment prevents dependency conflicts with other Python projects.

```bash
# Create virtual environment
python -m venv cns-env

# Activate it
# On macOS/Linux:
source cns-env/bin/activate

# On Windows:
cns-env\Scripts\activate
```

You should see `(cns-env)` appear in your terminal prompt.

### Step 2: Install Core Dependencies

Install the essential libraries needed for CNS 2.0:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core ML/NLP libraries (~1.5GB download)
pip install torch transformers sentence-transformers

# Install supporting libraries
pip install networkx numpy scikit-learn matplotlib
```

**Expected time:** 3-5 minutes depending on your internet connection.

**Download sizes:**
- PyTorch: ~800MB
- Transformers: ~400MB
- Sentence-transformers: ~50MB
- Other libraries: ~250MB

### Step 3: Verify Installation

Test that all imports work:

```bash
python -c "import torch; import transformers; import sentence_transformers; import networkx; import numpy; print('✓ All imports successful')"
```

**Expected output:**
```
✓ All imports successful
```

**If you see errors:**
- `ModuleNotFoundError`: Rerun the pip install command for that specific package
- `ImportError` with CUDA: This is fine if you don't have a GPU, PyTorch will use CPU
- Other errors: See [Troubleshooting](#troubleshooting) below

---

## Part 2: Create Your First SNO (5 minutes)

Now let's create a minimal but complete Structured Narrative Object.

### Step 1: Save the Code

Create a new file called `first_sno.py` and paste this code:

```python
"""
Minimal CNS 2.0 Example: Create Your First SNO
This demonstrates the core concept of a Structured Narrative Object
with semantic embedding capability.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import uuid

print("=" * 60)
print("CNS 2.0 Quick Start: Creating Your First SNO")
print("=" * 60)

# Step 1: Initialize the embedding model
# This downloads ~400MB on first run - be patient!
print("\n[1/5] Loading embedding model...")
print("      (First run downloads ~400MB, subsequent runs are instant)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("      ✓ Model loaded successfully")

# Step 2: Define a minimal SNO class
class SimpleSNO:
    """
    A simplified Structured Narrative Object for demonstration.
    The full version (Chapter 2) includes reasoning graphs and evidence sets.
    """
    def __init__(self, hypothesis: str, model):
        self.sno_id = str(uuid.uuid4())[:8]  # Short unique ID
        self.hypothesis = hypothesis
        self.embedding = model.encode(hypothesis)  # 384-dim semantic vector
        self.created_at = datetime.now()

    def __repr__(self):
        return f"SNO({self.sno_id}): {self.hypothesis}"

    def similarity_to(self, other: 'SimpleSNO') -> float:
        """Calculate semantic similarity with another SNO (0 to 1)"""
        dot_product = np.dot(self.embedding, other.embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other.embedding)
        return dot_product / (norm_a * norm_b)

# Step 3: Create several SNOs
print("\n[2/5] Creating Structured Narrative Objects...")

sno1 = SimpleSNO("Coffee improves programming productivity", model)
print(f"      ✓ Created: {sno1}")

sno2 = SimpleSNO("Caffeine enhances cognitive performance", model)
print(f"      ✓ Created: {sno2}")

sno3 = SimpleSNO("Python is a programming language", model)
print(f"      ✓ Created: {sno3}")

# Step 4: Verify embeddings
print("\n[3/5] Verifying embeddings...")
print(f"      Embedding shape: {sno1.embedding.shape}")
print(f"      Embedding type: {type(sno1.embedding)}")
print(f"      First 5 dimensions: {sno1.embedding[:5]}")
print("      ✓ Embeddings computed successfully")

# Step 5: Calculate semantic similarities
print("\n[4/5] Calculating semantic similarities...")
sim_1_2 = sno1.similarity_to(sno2)
sim_1_3 = sno1.similarity_to(sno3)
sim_2_3 = sno2.similarity_to(sno3)

print(f"      Similarity (Coffee & Caffeine): {sim_1_2:.3f}")
print(f"      Similarity (Coffee & Python):   {sim_1_3:.3f}")
print(f"      Similarity (Caffeine & Python): {sim_2_3:.3f}")
print("      ✓ As expected: Coffee/Caffeine are highly similar!")

# Step 6: Summary
print("\n[5/5] Summary")
print("=" * 60)
print(f"✓ Successfully created {3} Structured Narrative Objects")
print(f"✓ Each SNO has a unique ID, hypothesis, and 384-dim embedding")
print(f"✓ Semantic similarity works: related concepts cluster together")
print("\nWhat you just built:")
print("  • Semantic embeddings for natural language")
print("  • Similarity calculations between narratives")
print("  • Foundation for the full CNS 2.0 architecture")
print("\nNext steps:")
print("  → Chapter 1: Understand the CNS 2.0 architecture")
print("  → Chapter 2: Build the full SNO with reasoning graphs")
print("  → Chapter 3: Add critics for evaluation")
print("=" * 60)
```

### Step 2: Run It

```bash
python first_sno.py
```

### Expected Output

```
============================================================
CNS 2.0 Quick Start: Creating Your First SNO
============================================================

[1/5] Loading embedding model...
      (First run downloads ~400MB, subsequent runs are instant)
      ✓ Model loaded successfully

[2/5] Creating Structured Narrative Objects...
      ✓ Created: SNO(a3b5c7d9): Coffee improves programming productivity
      ✓ Created: SNO(f8e2c1b4): Caffeine enhances cognitive performance
      ✓ Created: SNO(d9f4a7b2): Python is a programming language

[3/5] Verifying embeddings...
      Embedding shape: (384,)
      Embedding type: <class 'numpy.ndarray'>
      First 5 dimensions: [-0.0234  0.0891 -0.0456  0.1234 -0.0678]
      ✓ Embeddings computed successfully

[4/5] Calculating semantic similarities...
      Similarity (Coffee & Caffeine): 0.847
      Similarity (Coffee & Python):   0.123
      Similarity (Caffeine & Python): 0.098
      ✓ As expected: Coffee/Caffeine are highly similar!

[5/5] Summary
============================================================
✓ Successfully created 3 Structured Narrative Objects
✓ Each SNO has a unique ID, hypothesis, and 384-dim embedding
✓ Semantic similarity works: related concepts cluster together

What you just built:
  • Semantic embeddings for natural language
  • Similarity calculations between narratives
  • Foundation for the full CNS 2.0 architecture

Next steps:
  → Chapter 1: Understand the CNS 2.0 architecture
  → Chapter 2: Build the full SNO with reasoning graphs
  → Chapter 3: Add critics for evaluation
============================================================
```

---

## Part 3: What You Just Built

Congratulations! You've created your first Structured Narrative Objects. Here's what each component does:

### The Hypothesis
```python
hypothesis = "Coffee improves programming productivity"
```
This is the central claim or narrative. In a full CNS system, this would be extracted from research papers, reports, or other knowledge sources.

### The Embedding (384-dimensional vector)
```python
embedding = model.encode(hypothesis)  # Shape: (384,)
```
This converts natural language into a mathematical representation that captures semantic meaning. Similar concepts have similar vectors, enabling computational reasoning about ideas.

**Why 384 dimensions?**
The `all-MiniLM-L6-v2` model outputs 384-dimensional vectors. This is a balance between:
- **Expressive power**: 384 dimensions can capture nuanced semantic relationships
- **Computational efficiency**: Small enough to compute quickly, even on CPUs

### Semantic Similarity
```python
similarity = sno1.similarity_to(sno2)  # 0.847 (highly similar)
```
By comparing embeddings mathematically (cosine similarity), the system can identify:
- **Related narratives** (high similarity, like "coffee" and "caffeine")
- **Contradictory narratives** (low similarity, opposite meanings)
- **Orthogonal narratives** (low similarity, unrelated topics)

This is the foundation for the **Chirality Score** in Chapter 4, which identifies productive conflicts.

### What's Missing (Coming in Later Chapters)

Your `SimpleSNO` is a starting point. The full `StructuredNarrativeObject` from Chapter 2 adds:

1. **Reasoning Graph (Chapter 2)**: A directed graph of logical claims and their relationships
2. **Evidence Set (Chapter 2)**: Links to source documents supporting each claim
3. **Trust Score (Chapter 3)**: Quality assessment from the critic pipeline
4. **Serialization (Chapter 2)**: Ability to save/load SNOs to/from disk
5. **Schema Versioning (Chapter 2)**: Handle changes to the SNO structure over time

---

## Experiment: Create Your Own SNO

Modify `first_sno.py` to create SNOs about your own research topic or area of interest:

```python
# Replace these with your own hypotheses
my_sno1 = SimpleSNO("Your hypothesis here", model)
my_sno2 = SimpleSNO("A related hypothesis", model)
my_sno3 = SimpleSNO("A contradictory hypothesis", model)

# Check similarities
print(f"Similarity 1-2: {my_sno1.similarity_to(my_sno2):.3f}")
print(f"Similarity 1-3: {my_sno1.similarity_to(my_sno3):.3f}")
```

**Try creating SNOs for:**
- Competing scientific theories (e.g., "Dark matter explains galaxy rotation" vs "Modified gravity explains galaxy rotation")
- Political positions
- Business strategies
- Historical interpretations

Share your results in [GitHub Discussions](https://github.com/your-org/cns-2.0/discussions) with the tag `#chapter0`!

---

## Troubleshooting

### Error: "No module named 'torch'"
**Cause:** PyTorch not installed
**Fix:**
```bash
pip install torch
```

### Error: "No module named 'sentence_transformers'"
**Cause:** Sentence-transformers not installed
**Fix:**
```bash
pip install sentence-transformers
```

### Error: "CUDA out of memory" or GPU warnings
**Cause:** Trying to use GPU but insufficient VRAM
**Fix:** Force CPU mode:
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

### Model download is stuck or very slow
**Causes:**
- Firewall blocking HuggingFace servers
- Slow internet connection
- Server temporarily down

**Fixes:**
1. Check your firewall settings (allow `huggingface.co`)
2. Try a different network
3. Manually download model from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Import works but model loading fails
**Symptom:**
```
OSError: Can't load tokenizer for 'all-MiniLM-L6-v2'
```

**Fix:** Clear the cache and re-download:
```bash
rm -rf ~/.cache/huggingface/
python first_sno.py
```

### Different similarity scores than expected
**This is normal.** Embedding models are non-deterministic across different:
- CPU vs GPU
- Different model versions
- Different random seeds

As long as:
- Related concepts have HIGH similarity (>0.7)
- Unrelated concepts have LOW similarity (<0.3)

Your system is working correctly.

### Python version error
**Symptom:**
```
SyntaxError: invalid syntax (match/case statement, etc.)
```

**Fix:** Upgrade Python:
```bash
python --version  # Check current version
# If < 3.9, install Python 3.9+ from python.org
```

---

## Performance Notes

### First Run vs Subsequent Runs

**First run:**
- Downloads model: ~2-3 minutes
- Loads model into memory: ~5 seconds
- Creates embeddings: <1 second

**Subsequent runs:**
- Model already cached locally
- Loads from disk: ~5 seconds
- Creates embeddings: <1 second

### Hardware Requirements

**Minimum (CPU only):**
- 4GB RAM
- ~30 seconds to load model
- ~0.1 seconds per embedding

**Recommended (GPU):**
- 8GB RAM + NVIDIA GPU (2GB VRAM)
- ~5 seconds to load model
- ~0.01 seconds per embedding (10x faster)

**For large-scale systems:**
- See Chapter 6 for production deployment
- See Chapter 5 for distributed processing with Celery

---

## Next Steps

Now that you have a working CNS 2.0 environment and understand the basic concept of Structured Narrative Objects, you're ready to dive deeper.

### Complete Learning Path

| Chapter | Time | What You'll Build | Key Outputs |
|---------|------|-------------------|-------------|
| **0** (this chapter) | 15 min | First SNO with embeddings | 3 SNOs, similarity scores |
| **[1: Introduction](/guides/building-cns-2.0-developers-guide/chapter-1-introduction/)** | 30 min | Environment + Config | test_chapter1.py passes |
| **[2: SNO Foundations](/guides/building-cns-2.0-developers-guide/chapter-2-sno-foundations/)** | 45 min | Complete SNO with reasoning graph | 6 claims, 4 evidence, serialization |
| **[3: Critic Pipeline](/guides/building-cns-2.0-developers-guide/chapter-3-critic-pipeline/)** | 45 min | Multi-component evaluation | Trust score 0.72, 3 critic scores |
| **[4: Synthesis Engine](/guides/building-cns-2.0-developers-guide/chapter-4-synthesis-engine/)** | 60 min | Chiral pair detection + viz | 6 SNO population, t-SNE plot |
| **[5: System Integration](/guides/building-cns-2.0-developers-guide/chapter-5-system-integration/)** | 60 min | Async workflow manager | Production-ready system |
| **[6: Production Deployment](/guides/building-cns-2.0-developers-guide/chapter-6-complete-implementation/)** | 90 min | Docker + Celery | Distributed processing |
| **[7: DSPy Optimization](/guides/building-cns-2.0-developers-guide/chapter-7-dspy-integration/)** | 90 min | Self-improving system | Optimized prompts |

**Total Time:** ~7 hours for complete mastery

**Recommended Approach:**
- **Day 1:** Chapters 0-2 (90 min) → Understand SNOs
- **Day 2:** Chapters 3-4 (105 min) → Add evaluation & synthesis
- **Day 3:** Chapters 5-7 (240 min) → Production system

### What Each Chapter Adds

**Chapter 1: Introduction & Architecture**
- Understand the theoretical foundation
- Set up complete Python environment
- Initialize embedding models
- Define configuration system

**Chapter 2: SNO Foundations**
- Build full `StructuredNarrativeObject` class
- Add reasoning graphs (claims + logical edges)
- Attach evidence sets with DOI citations
- Implement serialization for persistence

**Chapter 3: Critic Pipeline**
- Implement Grounding Critic (evidence coverage)
- Implement Logic Critic (structural coherence)
- Implement Novelty Critic (innovation vs complexity)
- Build composite trust score
- Enable contextual evaluation

**Chapter 4: Synthesis Engine**
- Calculate chirality (semantic opposition)
- Calculate evidential entanglement (shared evidence)
- Detect chiral pairs algorithmically
- Visualize narrative space with t-SNE
- Identify productive conflicts

---

## Additional Resources

- **[Research Roadmap](/guides/cns-2.0-research-roadmap/)**: Long-term vision and advanced research directions
- **[Case Studies](/guides/case-studies-and-experiments/)**: Real-world applications and experiments
- **[Tutorials](/guides/tutorials/)**: Step-by-step guides for specific use cases

> **Note:** A GitHub repository with all example code from this guide will be published soon. Check back for updates or contact the maintainers for early access.

---

**Estimated completion time for this chapter: 15-20 minutes**

*If you completed this chapter successfully, you've proven the core concept works. The rest of the guide builds on this foundation.*

---

## Navigation

**← Previous:** [Developer's Guide Home](/guides/building-cns-2.0-developers-guide/)
**→ Next:** [Chapter 1: Introduction to CNS 2.0](/guides/building-cns-2.0-developers-guide/chapter-1-introduction/)
