from collections.abc import Sequence

import numpy as np


def _unit(vector: Sequence[float]) -> np.ndarray | None:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else None


def _mean_direction(vectors: Sequence[Sequence[float]]) -> np.ndarray | None:
    units = [unit for unit in (_unit(vector) for vector in vectors) if unit is not None]
    return np.mean(units, axis=0) if units else None


def refine_query_vector(
    query: Sequence[float],
    relevant: Sequence[Sequence[float]] = (),
    irrelevant: Sequence[Sequence[float]] = (),
    relevant_weight: float = 0.5,
    irrelevant_weight: float = 0.3,
) -> list[float]:
    """Rocchio relevance feedback: pull the query toward relevant books, away from irrelevant ones.

        q' = unit(q) + relevant_weight * mean(unit(relevant)) - irrelevant_weight * mean(unit(irrelevant))

    Every vector is normalized first so the weights mean the same thing regardless of
    embedding scale, and the result is returned as a unit vector. With no usable
    feedback (or a degenerate zero result) the original query is returned unchanged.
    """
    base = _unit(query)
    if base is None:
        return list(query)

    refined = base.copy()
    pull = _mean_direction(relevant)
    if pull is not None:
        refined = refined + relevant_weight * pull
    push = _mean_direction(irrelevant)
    if push is not None:
        refined = refined - irrelevant_weight * push

    norm = float(np.linalg.norm(refined))
    if norm == 0:
        return list(query)
    return (refined / norm).tolist()
