from collections.abc import Mapping

from app.core.config import settings
from app.data.repositories.feedback_repository import FeedbackVotes


def feedback_adjustment(
    votes: FeedbackVotes,
    weight: float | None = None,
    prior_strength: float | None = None,
    min_votes: int | None = None,
) -> float:
    """Score shift for one (query, book) pair from its relevant/irrelevant votes.

    adj = (up - down) / (up + down + prior) stays strictly inside (-1, 1) and is
    pulled toward 0 while votes are few; the result is capped by `weight`, so votes
    can reorder near-ties but never lift a genuinely unrelated book on their own.
    Pairs with fewer than `min_votes` votes are ignored (resists a lone bad actor).
    """
    weight = settings.feedback_weight if weight is None else weight
    prior_strength = settings.feedback_prior_strength if prior_strength is None else prior_strength
    min_votes = settings.feedback_min_votes if min_votes is None else min_votes

    total = votes.up + votes.down
    if total < max(min_votes, 1):
        return 0.0
    return weight * (votes.up - votes.down) / (total + prior_strength)


def compute_adjustments(votes: Mapping[str, FeedbackVotes]) -> dict[str, float]:
    """isbn13 -> score shift, only for books whose votes actually move the score."""
    adjustments = {isbn: feedback_adjustment(entry) for isbn, entry in votes.items()}
    return {isbn: delta for isbn, delta in adjustments.items() if delta != 0.0}
