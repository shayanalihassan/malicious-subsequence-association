import re
from typing import List, Sequence, Tuple

TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
WORD_RE = re.compile(r"^\w+(?:'\w+)?$")
SINGLE_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'_-]*$")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def detokenize(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
            continue
        if re.match(r"[,.;:!?)]", token):
            text += token
        elif token in {"'s", "n't"}:
            text += token
        elif text.endswith("("):
            text += token
        else:
            text += " " + token
    return text

def word_positions(tokens: Sequence[str]) -> List[int]:
    return [i for i, token in enumerate(tokens) if WORD_RE.match(token)]


def locate_word_char_spans(text: str) -> List[Tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\w+(?:'\w+)?", text)]


def is_contiguous(indices: Sequence[int]) -> bool:
    return all(right == left + 1 for left, right in zip(indices, indices[1:]))


def format_subsequence(words: Sequence[str], indices: Sequence[int]) -> str:
    parts: List[str] = []
    for pos, word_idx in enumerate(indices):
        if pos > 0 and word_idx != indices[pos - 1] + 1:
            parts.append("...")
        parts.append(words[word_idx])
    return " ".join(parts)

def sanitize_fill_word(raw_value: str, fallback: str) -> str:
    candidate = raw_value.strip().split()[0] if raw_value.strip() else ""
    candidate = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9'_-]+$", "", candidate)
    return candidate if SINGLE_WORD_RE.match(candidate) else fallback