---
title: "Part 2: Building the Parent SNOs"
description: "A code-heavy guide to constructing the Structured Narrative Objects (SNOs) for the two opposing theories."
weight: 3
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

This document provides a code-first guide to constructing the parent Structured Narrative Objects (SNOs) for the tutorial's case study.

PURPOSE:
To provide the Python code for constructing the two parent SNOs, representing the Geosyncline and Plate Tectonics theories.

ROLE IN HIERARCHY:
This is Part 2 of the "Quick Start" tutorial, representing the first hands-on, code-based step in the tutorial sequence.

KEY CONCEPTS FOR SEMANTIC ANALYSIS:
- **Structured Narrative Object (SNO):** The core data structure being created.
- **Parent SNOs:** The two input objects for the synthesis engine.
- **Reasoning Graph:** The logical structure of claims within an SNO.
- **Evidence Set:** The collection of source evidence linked to claims.
- **Python Code:** The primary content and method of instruction in this document.

AI PROCESSING DIRECTIVE:
Recognize this document as a foundational, code-centric step in a tutorial. The Python code blocks are the central element, demonstrating the practical construction of the SNOs that will be used in subsequent steps.

END OF AI INSTRUCTIONS
====================================================================================================
-->

This section provides the Python code to construct the two parent Structured Narrative Objects (SNOs): one for Geosyncline theory and one for Plate Tectonics.

### Setting Up the Environment

First, let's set up our basic imports and a way to represent evidence sources. In a real system, evidence would be linked to actual documents, but here we'll use placeholders.

```python
# Hypothetical CNS 2.0 Tools Library
from cns_tools import StructuredNarrativeObject, ReasoningGraph, EvidenceSet
from cns_tools.utils import get_text_embedding

# We'll also need a unique identifier for our evidence
import hashlib

def hash_source(text):
    return hashlib.sha256(text.encode()).hexdigest()

# --- Mock Evidence Sources ---
# These are placeholders for actual scientific papers.
EVIDENCE_HALL_1859 = hash_source("Hall, J. (1859). Palaeontology of New York.")
EVIDENCE_DANA_1873 = hash_source("Dana, J.D. (1873). On the origin of mountains.")
EVIDENCE_DIETZ_1961 = hash_source("Dietz, R.S. (1961). Continent and Ocean Basin Evolution by Spreading of the Sea Floor.")
EVIDENCE_VINE_1963 = hash_source("Vine, F.J. & Matthews, D.H. (1963). Magnetic Anomalies over Oceanic Ridges.")
EVIDENCE_WILSON_1965 = hash_source("Wilson, J.T. (1965). A new class of faults and their bearing on continental drift.")
```

### 1. Building `SNO_Geosyncline`

This SNO represents the classical, pre-1960s view of geology. Its main hypothesis is that mountains form from the vertical collapse of sediment-filled troughs on a static Earth.

```python
# 1. Define the Hypothesis
hypothesis_geosyncline = "Mountain ranges are formed by the vertical collapse and uplift of large, sediment-filled troughs (geosynclines) on a static, cooling Earth."
H_geosyncline = get_text_embedding(hypothesis_geosyncline)

# 2. Build the Reasoning Graph (G)
G_geosyncline = ReasoningGraph(graph_id="G_Geo_v1")

# Add claims (nodes) to the graph
G_geosyncline.add_claim("c1", "The Earth is a cooling and contracting body.")
G_geosyncline.add_claim("c2", "Thick sedimentary deposits accumulate in large troughs (geosynclines).")
G_geosyncline.add_claim("c3", "The crust buckles under the sediment weight and compressional forces from cooling.")
G_geosyncline.add_claim("c4", "This buckling leads to vertical uplift, forming mountain ranges.")
G_geosyncline.add_claim("c5", "Continents and ocean basins are permanent, fixed features.")

# Add reasoning relationships (edges) between claims
G_geosyncline.add_edge("c1", "c3", "supports")
G_geosyncline.add_edge("c2", "c3", "supports")
G_geosyncline.add_edge("c3", "c4", "implies")
G_geosyncline.add_edge("c5", "c1", "is_consistent_with")

# 3. Populate the Evidence Set (E)
E_geosyncline = EvidenceSet(evidence_id="E_Geo_v1")
E_geosyncline.add_evidence(EVIDENCE_HALL_1859, "Supports the existence of thick sedimentary layers in mountain belts.", supports_claims=["c2"])
E_geosyncline.add_evidence(EVIDENCE_DANA_1873, "Provides a mechanism for compression and uplift.", supports_claims=["c3", "c4"])

# 4. Instantiate the SNO
SNO_geosyncline = StructuredNarrativeObject(
    hypothesis_embedding=H_geosyncline,
    reasoning_graph=G_geosyncline,
    evidence_set=E_geosyncline,
    trust_score=None # The score is computed later by a different part of the system.
)

print("SNO_Geosyncline created successfully.")
```

### 2. Building `SNO_PlateTectonics`

This SNO represents the modern, revolutionary view. Its main hypothesis is that the Earth's surface is composed of moving plates whose interactions build mountains.

```python
# 1. Define the Hypothesis
hypothesis_tectonics = "The Earth's surface is composed of rigid lithospheric plates that move, and their interactions at boundaries are the primary cause of mountain building, earthquakes, and volcanism."
H_tectonics = get_text_embedding(hypothesis_tectonics)

# 2. Build the Reasoning Graph (G)
G_tectonics = ReasoningGraph(graph_id="G_PT_v1")

# Add claims (nodes)
G_tectonics.add_claim("c1", "The lithosphere is divided into rigid plates.")
G_tectonics.add_claim("c2", "New oceanic crust is generated at mid-ocean ridges (seafloor spreading).")
G_tectonics.add_claim("c3", "Oceanic crust is consumed at subduction zones.")
G_tectonics.add_claim("c4", "Plate motion is driven by mantle convection.")
G_tectonics.add_claim("c5", "Mountain ranges are formed by the collision of continental plates or subduction.")
G_tectonics.add_claim("c6", "The continents are not fixed but drift over time.")

# Add reasoning relationships (edges)
G_tectonics.add_edge("c2", "c1", "supports")
G_tectonics.add_edge("c3", "c1", "supports")
G_tectonics.add_edge("c1", "c5", "implies")
G_tectonics.add_edge("c4", "c1", "provides_mechanism_for")
G_tectonics.add_edge("c2", "c6", "implies")

# This is a key point of conflict with the other SNO
G_tectonics.add_claim("c7_conflict", "Continents and ocean basins are NOT permanent, fixed features.")
G_tectonics.add_edge("c6", "c7_conflict", "implies")

# 3. Populate the Evidence Set (E)
E_tectonics = EvidenceSet(evidence_id="E_PT_v1")
E_tectonics.add_evidence(EVIDENCE_DIETZ_1961, "Proposes the mechanism of seafloor spreading.", supports_claims=["c2"])
E_tectonics.add_evidence(EVIDENCE_VINE_1963, "Symmetrical magnetic stripes around mid-ocean ridges provide strong proof of seafloor spreading.", supports_claims=["c2"])
E_tectonics.add_evidence(EVIDENCE_WILSON_1965, "Identifies transform faults, a necessary component of plate boundary interactions.", supports_claims=["c1", "c5"])

# 4. Instantiate the SNO
SNO_plate_tectonics = StructuredNarrativeObject(
    hypothesis_embedding=H_tectonics,
    reasoning_graph=G_tectonics,
    evidence_set=E_tectonics,
    trust_score=None # The score is computed later.
)

print("SNO_PlateTectonics created successfully.")
```
