"""Conservative punctuation-based sentence segmentation for optional API pauses."""

import re

PAUSE_SECONDS = 0.3
# Western punctuation requires whitespace; CJK punctuation need not have spaces.
_BOUNDARY = re.compile(r"""(?:[.!?]+["'”’）)\]]*(?=\s|$)|[。！？]+["'”’」』）)\]]*)""")
_ABBREVIATION = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)|\b[A-Za-z]|"
    r"\b(?:[A-Za-z]\.)+[A-Za-z])\.$",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Keep punctuation, decimals, initials and common abbreviations intact.

    This is a heuristic, not language-specific linguistic segmentation.
    """
    sentences = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()
        if match.group() == "." and _ABBREVIATION.search(candidate):
            continue
        if candidate:
            sentences.append(candidate)
        start = end
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences or [text]
