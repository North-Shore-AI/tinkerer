"""Chirality + Fisher-Rao helpers for CNS 3.0 metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class FisherRaoStats:
    """Lightweight diagonal Fisher-Rao approximation."""

    mean: np.ndarray
    inv_var: np.ndarray


def build_fisher_rao_stats(vectors: Sequence[Sequence[float]] | np.ndarray, epsilon: float = 1e-6) -> FisherRaoStats:
    """Compute diagonal Fisher-Rao statistics from embedding vectors."""
    matrix = np.asarray(vectors)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0 or matrix.shape[0] == 0:
        raise ValueError("Cannot build Fisher-Rao stats from empty vector set")
    mean = matrix.mean(axis=0)
    var = matrix.var(axis=0) + epsilon  # add jitter to avoid divide-by-zero
    inv_var = 1.0 / var
    return FisherRaoStats(mean=mean, inv_var=inv_var)


def fisher_rao_distance(vec_a: np.ndarray, vec_b: np.ndarray, stats: FisherRaoStats) -> float:
    """Mahalanobis-style distance using the diagonal Fisher information."""
    diff = vec_a - vec_b
    return float(np.sqrt(np.dot(diff * stats.inv_var, diff)))


@dataclass
class ChiralityResult:
    fisher_rao_distance: float
    evidence_overlap: float
    polarity_conflict: bool
    chirality_score: float


class ChiralityAnalyzer:
    """Encodes CNS chirality as Fisher-Rao distance + evidence tension."""

    def __init__(self, embedder, stats: FisherRaoStats):
        self.embedder = embedder
        self.stats = stats

    def _encode(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros_like(self.stats.mean)
        vector = self.embedder.encode([text], convert_to_numpy=True)[0]
        if vector.shape != self.stats.mean.shape:
            # Pad or trim to match stats dimensions (defensive).
            target = self.stats.mean.shape[0]
            current = vector.shape[0]
            if current > target:
                vector = vector[:target]
            else:
                vector = np.pad(vector, (0, target - current))
        return vector

    def compare(
        self,
        thesis: str,
        antithesis: str,
        *,
        evidence_overlap: float,
        polarity_conflict: bool,
    ) -> ChiralityResult:
        vec_thesis = self._encode(thesis)
        vec_antithesis = self._encode(antithesis)
        distance = fisher_rao_distance(vec_thesis, vec_antithesis, self.stats)

        # Normalize Fisher-Rao distance into [0, 1]
        norm_distance = distance / (distance + 1.0)

        # Evidence overlap encourages low chirality when overlap is low.
        overlap_factor = 1.0 - min(max(evidence_overlap, 0.0), 1.0)

        # Polarity conflict is a discrete penalty term.
        conflict_penalty = 0.25 if polarity_conflict else 0.0

        chirality_score = min(1.0, norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty)
        return ChiralityResult(
            fisher_rao_distance=distance,
            evidence_overlap=evidence_overlap,
            polarity_conflict=polarity_conflict,
            chirality_score=chirality_score,
        )
