from collections.abc import Sequence

import numpy as np


def mmr_select(
    vectors: Sequence[Sequence[float]],
    relevance: Sequence[float],
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance: indices of up to `k` candidates, in pick order.

    Each pick maximizes `lambda * relevance - (1 - lambda) * max_cosine_to_already_picked`,
    so near-duplicates of an already-picked item are penalized. `lambda_mult=1.0`
    degenerates to plain top-k by relevance.
    """
    if len(vectors) != len(relevance):
        raise ValueError("vectors and relevance must have the same length")
    count = len(vectors)
    if k <= 0 or count == 0:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    rel = np.asarray(relevance, dtype=np.float32)

    picked = np.zeros(count, dtype=bool)
    selected: list[int] = []
    max_sim_to_selected = np.zeros(count, dtype=np.float32)

    for _ in range(min(k, count)):
        if selected:
            scores = lambda_mult * rel - (1.0 - lambda_mult) * max_sim_to_selected
        else:
            scores = rel.copy()
        scores[picked] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        picked[best] = True
        similarity = matrix @ matrix[best]
        max_sim_to_selected = (
            similarity if len(selected) == 1 else np.maximum(max_sim_to_selected, similarity)
        )

    return selected
