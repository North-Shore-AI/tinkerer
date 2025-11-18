"""Compute simple Betti numbers and cycle diagnostics for CLAIM/RELATION graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx


Relation = Tuple[str, str, str]


@dataclass
class GraphStats:
    nodes: int
    edges: int
    components: int
    beta1: int
    cycles: List[List[str]]
    polarity_conflict: bool


def _normalize_claim_id(identifier: str) -> str:
    return identifier.strip().lower()


def _determine_polarity_conflict(relations: Sequence[Relation], target: str = "c1") -> bool:
    """Returns True when a claim receives both support and refute edges."""
    target = _normalize_claim_id(target)
    polarities = {"supports": False, "refutes": False}
    for src, label, dst in relations:
        if _normalize_claim_id(dst) != target:
            continue
        lbl = label.lower()
        if lbl in polarities:
            polarities[lbl] = True
        if polarities["supports"] and polarities["refutes"]:
            return True
    return False


def compute_graph_stats(claim_ids: Iterable[str], relations: Sequence[Relation]) -> GraphStats:
    """Build a reasoning graph and compute topology metrics."""
    graph = nx.DiGraph()
    normalized_claims = {_normalize_claim_id(cid) for cid in claim_ids}
    for claim_id in normalized_claims:
        graph.add_node(claim_id)

    normalized_relations: List[Relation] = []
    for src, label, dst in relations:
        src_norm = _normalize_claim_id(src)
        dst_norm = _normalize_claim_id(dst)
        if src_norm not in normalized_claims or dst_norm not in normalized_claims:
            continue
        lbl = label.lower()
        normalized_relations.append((src_norm, lbl, dst_norm))
        graph.add_edge(src_norm, dst_norm, label=lbl)

    undirected = graph.to_undirected()
    components = nx.number_connected_components(undirected) if undirected.number_of_nodes() else 0
    beta1 = max(0, undirected.number_of_edges() - undirected.number_of_nodes() + components)
    cycles = list(nx.simple_cycles(graph))
    return GraphStats(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        components=components,
        beta1=beta1,
        cycles=cycles,
        polarity_conflict=_determine_polarity_conflict(normalized_relations),
    )
