import pytest

from app.data.repositories.feedback_repository import FeedbackVotes
from app.domain.feedback_scoring import compute_adjustments, feedback_adjustment

W, K, MIN = 0.05, 5.0, 3


def _adj(up, down, **overrides):
    params = {"weight": W, "prior_strength": K, "min_votes": MIN, **overrides}
    return feedback_adjustment(FeedbackVotes(up=up, down=down), **params)


def test_fewer_votes_than_the_minimum_are_ignored():
    assert _adj(2, 0) == 0.0
    assert _adj(1, 1) == 0.0
    assert _adj(0, 0) == 0.0


def test_relevant_votes_raise_and_irrelevant_votes_lower_the_score():
    assert _adj(5, 0) > 0
    assert _adj(0, 5) < 0


def test_the_shift_is_symmetric():
    assert _adj(6, 0) == pytest.approx(-_adj(0, 6))


def test_balanced_votes_cancel_out():
    assert _adj(4, 4) == 0.0


def test_formula_matches_the_documented_shrinkage():
    # (5 - 0) / (5 + 0 + 5) * 0.05
    assert _adj(5, 0) == pytest.approx(0.025)


def test_more_agreeing_votes_move_the_score_further_but_never_past_the_weight():
    shifts = [_adj(n, 0) for n in (3, 5, 10, 50, 1000)]
    assert shifts == sorted(shifts)
    assert all(0 < shift < W for shift in shifts)


def test_min_votes_below_one_still_ignores_pairs_with_no_votes():
    assert _adj(0, 0, min_votes=0) == 0.0


def test_defaults_come_from_settings(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "feedback_weight", 0.1)
    monkeypatch.setattr(settings, "feedback_prior_strength", 0.0)
    monkeypatch.setattr(settings, "feedback_min_votes", 1)
    assert feedback_adjustment(FeedbackVotes(up=1, down=0)) == pytest.approx(0.1)


def test_compute_adjustments_keeps_only_books_that_actually_move(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "feedback_weight", W)
    monkeypatch.setattr(settings, "feedback_prior_strength", K)
    monkeypatch.setattr(settings, "feedback_min_votes", MIN)
    result = compute_adjustments(
        {
            "up": FeedbackVotes(up=6, down=0),
            "down": FeedbackVotes(up=0, down=6),
            "too_few": FeedbackVotes(up=1, down=0),
            "tie": FeedbackVotes(up=3, down=3),
        }
    )
    assert set(result) == {"up", "down"}
    assert result["up"] > 0 > result["down"]
