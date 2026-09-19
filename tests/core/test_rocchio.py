import numpy as np
import pytest

from app.core.rocchio import refine_query_vector


def _cos(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


Q = [1.0, 0.0, 0.0]
NEAR = [0.0, 1.0, 0.0]   # a "relevant" book in another direction
FAR = [0.0, 0.0, 1.0]    # an "irrelevant" book in yet another


def test_no_feedback_returns_the_original_direction():
    refined = refine_query_vector(Q, [], [])
    assert _cos(refined, Q) == pytest.approx(1.0)


def test_relevant_books_pull_the_query_toward_them():
    refined = refine_query_vector(Q, [NEAR], [])
    assert _cos(refined, NEAR) > _cos(Q, NEAR)
    assert _cos(refined, Q) < 1.0  # it moved


def test_irrelevant_books_push_the_query_away_from_them():
    q = [1.0, 0.0, 1.0]
    refined = refine_query_vector(q, [], [FAR])
    assert _cos(refined, FAR) < _cos(q, FAR)


def test_both_directions_combine():
    refined = refine_query_vector([1.0, 1.0, 1.0], [NEAR], [FAR])
    assert _cos(refined, NEAR) > _cos([1.0, 1.0, 1.0], NEAR)
    assert _cos(refined, FAR) < _cos([1.0, 1.0, 1.0], FAR)


def test_exact_result_matches_the_rocchio_formula():
    # unit(q) + 0.5 * unit(near) = [1, 0.5, 0] -> normalized
    refined = refine_query_vector(Q, [NEAR], [], relevant_weight=0.5)
    expected = np.array([1.0, 0.5, 0.0]) / np.linalg.norm([1.0, 0.5, 0.0])
    assert refined == pytest.approx(expected.tolist(), abs=1e-6)


def test_output_is_a_unit_vector():
    refined = refine_query_vector([3.0, 4.0, 0.0], [[0.0, 2.0, 5.0]], [[7.0, 0.0, 1.0]])
    assert np.linalg.norm(refined) == pytest.approx(1.0, abs=1e-6)


def test_scale_of_the_inputs_does_not_change_the_result():
    small = refine_query_vector([1.0, 0.0, 0.0], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]])
    large = refine_query_vector([50.0, 0.0, 0.0], [[0.0, 0.001, 0.0]], [[0.0, 0.0, 900.0]])
    assert small == pytest.approx(large, abs=1e-5)


def test_more_marked_books_average_instead_of_stacking():
    one = refine_query_vector(Q, [NEAR], [])
    three = refine_query_vector(Q, [NEAR, NEAR, NEAR], [])
    assert one == pytest.approx(three, abs=1e-6)


def test_zero_weights_leave_the_query_unchanged():
    refined = refine_query_vector(Q, [NEAR], [FAR], relevant_weight=0.0, irrelevant_weight=0.0)
    assert refined == pytest.approx(Q, abs=1e-6)


def test_zero_vectors_in_the_feedback_are_ignored():
    with_zero = refine_query_vector(Q, [[0.0, 0.0, 0.0], NEAR], [[0.0, 0.0, 0.0]])
    plain = refine_query_vector(Q, [NEAR], [])
    assert with_zero == pytest.approx(plain, abs=1e-6)


def test_a_zero_query_is_returned_as_is():
    assert refine_query_vector([0.0, 0.0, 0.0], [NEAR], []) == [0.0, 0.0, 0.0]


def test_a_cancelling_result_falls_back_to_the_original_query():
    # relevant == -query with equal weight cancels the query out exactly
    refined = refine_query_vector(Q, [[-1.0, 0.0, 0.0]], [], relevant_weight=1.0)
    assert refined == list(Q)


def test_mismatched_dimensions_raise_so_callers_can_fail_open():
    with pytest.raises(ValueError):
        refine_query_vector([1.0, 0.0, 0.0], [[1.0, 0.0]], [])
