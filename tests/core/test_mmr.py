import pytest

from app.core.mmr import mmr_select

# A and A_NEAR are near-duplicates; B points in a different direction.
A = [1.0, 0.0]
A_NEAR = [0.99, 0.01]
B = [0.0, 1.0]


def test_first_pick_is_always_the_most_relevant():
    assert mmr_select([B, A, A_NEAR], [0.5, 0.9, 0.8], k=1)[0] == 1


def test_lambda_one_is_plain_top_k_by_relevance():
    picked = mmr_select([A, A_NEAR, B], [0.9, 0.89, 0.6], k=2, lambda_mult=1.0)
    assert picked == [0, 1]


def test_lower_lambda_prefers_a_diverse_second_pick_over_a_near_duplicate():
    picked = mmr_select([A, A_NEAR, B], [0.9, 0.89, 0.6], k=2, lambda_mult=0.5)
    assert picked == [0, 2]


def test_k_larger_than_pool_returns_every_candidate_once():
    picked = mmr_select([A, A_NEAR, B], [0.9, 0.89, 0.6], k=10)
    assert sorted(picked) == [0, 1, 2]


def test_non_positive_k_or_empty_pool_returns_empty():
    assert mmr_select([A], [0.9], k=0) == []
    assert mmr_select([], [], k=3) == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        mmr_select([A, B], [0.9], k=1)


def test_zero_vector_does_not_crash_or_produce_nan_picks():
    picked = mmr_select([[0.0, 0.0], A, B], [0.1, 0.9, 0.8], k=3)
    assert sorted(picked) == [0, 1, 2]
