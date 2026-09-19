import pytest

from app.domain.question_analysis import classify_complexity, needs_condense


@pytest.mark.parametrize(
    "question",
    [
        "Who is the narrator of the story?",
        "Kitabın yazarı kim?",
        "In what year does the story begin?",
    ],
)
def test_short_factual_questions_are_simple(question):
    assert classify_complexity(question) == "simple"


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the first part of the book",
        "Bu bölümün özetini çıkarır mısın?",
        "Why does the protagonist leave home?",
        "Ana karakterin gelişimini açıkla",
        "Compare the two brothers",
        "Who is she? Where does she live?",
        "x" * 160,
    ],
)
def test_synthesis_style_or_multi_part_questions_are_complex(question):
    assert classify_complexity(question) == "complex"


@pytest.mark.parametrize(
    "question",
    [
        "why?",
        "Peki ya sonra?",
        "What happens to him after the trial ends in the story?",
        "Onun kardeşi kim olduğunu anlatan bölüm hangisi?",
        "Can you tell me more about that part of the story?",
    ],
)
def test_follow_ups_need_the_chat_history(question):
    assert needs_condense(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is the name of the ship Captain Ahab commands?",
        "Kaptan Ahab'ın komuta ettiği geminin adı nedir sence?",
        "Where does the story of Anna Karenina take place initially?",
    ],
)
def test_self_contained_questions_skip_the_condense_call(question):
    assert needs_condense(question) is False
