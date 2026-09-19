import pytest
from pydantic import ValidationError

from app.models.schemas import SearchFeedback, SemanticSearchRequest


def test_feedback_is_optional_and_defaults_to_none():
    assert SemanticSearchRequest(query="detective").feedback is None


def test_marks_default_to_empty_lists():
    feedback = SearchFeedback()
    assert feedback.relevant == [] and feedback.irrelevant == []


def test_isbns_are_normalized_and_deduplicated_in_order():
    feedback = SearchFeedback(relevant=[" 978013468599x ", "9780134685991", "978013468599X"])
    assert feedback.relevant == ["978013468599X", "9780134685991"]


@pytest.mark.parametrize("bad", ["abc", "123", "97801346859912345", "978-0134685991", ""])
def test_invalid_isbns_are_rejected(bad):
    with pytest.raises(ValidationError):
        SearchFeedback(relevant=[bad])
    with pytest.raises(ValidationError):
        SearchFeedback(irrelevant=[bad])


def test_at_most_five_marks_per_side():
    five = [f"97800000000{i:02d}" for i in range(5)]
    assert len(SearchFeedback(relevant=five).relevant) == 5
    with pytest.raises(ValidationError):
        SearchFeedback(relevant=five + ["9780000000099"])
    with pytest.raises(ValidationError):
        SearchFeedback(irrelevant=five + ["9780000000099"])


def test_request_accepts_the_client_json_shape():
    body = SemanticSearchRequest.model_validate(
        {
            "query": "detective novel",
            "feedback": {"relevant": ["9780134685991"], "irrelevant": ["9780000000001"]},
        }
    )
    assert body.feedback.relevant == ["9780134685991"]
    assert body.feedback.irrelevant == ["9780000000001"]
