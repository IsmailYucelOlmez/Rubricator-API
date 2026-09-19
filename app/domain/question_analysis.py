import re
from typing import Literal

Complexity = Literal["simple", "complex"]

# Questions that ask for synthesis over many passages rather than one fact.
_COMPLEX_HINT_RE = re.compile(
    r"\b("
    r"summar\w*|overview|compare|comparison|contrast|explain|analy\w+|theme\w*|"
    r"relationship\w*|why|how does|main (idea|point|argument|character)s?|"
    r"özet\w*|karşılaştır\w*|analiz\w*|açıkla\w*|tema\w*|ilişki\w*|neden|niçin|"
    r"ana (fikir|karakter|konu)\w*|genel olarak"
    r")\b",
    re.IGNORECASE,
)

_COMPLEX_MIN_LENGTH = 150

# Anaphora / follow-up markers: the question cannot be understood without history.
_FOLLOW_UP_RE = re.compile(
    r"\b("
    r"it|its|this|that|these|those|they|them|their|he|him|his|she|her|hers|"
    r"also|again|else|more|instead|previous|earlier|above|"
    r"o|onu|ona|onun|onda|ondan|bu|bunu|buna|bunun|bunda|bundan|şu|şunu|şuna|şunun|"
    r"onlar\w*|bunlar\w*|şunlar\w*|peki|ya|ayrıca|başka|devam|önceki"
    r")\b",
    re.IGNORECASE,
)

_SELF_CONTAINED_MIN_LENGTH = 25


def classify_complexity(question: str) -> Complexity:
    """Cheap heuristic (no LLM call): does answering need many passages?"""
    text = question.strip()
    if len(text) >= _COMPLEX_MIN_LENGTH:
        return "complex"
    if text.count("?") >= 2:
        return "complex"
    if _COMPLEX_HINT_RE.search(text):
        return "complex"
    return "simple"


def needs_condense(question: str) -> bool:
    """Whether a follow-up must be rewritten into a standalone question using chat history."""
    text = question.strip()
    if len(text) < _SELF_CONTAINED_MIN_LENGTH:
        return True
    return _FOLLOW_UP_RE.search(text) is not None
