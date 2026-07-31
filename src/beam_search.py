import random

from typing import Dict, Iterable, List, Sequence

from src.text_utils import tokenize, WORD_RE

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "being",
    "by",
    "can",
    "do",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "next",
    "of",
    "on",
    "or",
    "real",
    "simple",
    "someone",
    "that",
    "the",
    "to",
    "way",
    "what",
    "without",
    "write",
}

HARMFUL_HINTS = {
    "break",
    "caught",
    "computer",
    "email",
    "fake",
    "getting",
    "id",
    "noticed",
    "poison",
    "sabotage",
    "shoplift",
}

def find_expected_trigger(tokens: Sequence[str], expected_trigger: str) -> tuple[int, ...] | None:
    if not expected_trigger:
        return None
    expected_tokens = [tok.lower() for tok in tokenize(expected_trigger) if WORD_RE.match(tok)]
    word_tokens = [tok.lower() for tok in tokens if WORD_RE.match(tok)]
    for start in range(len(word_tokens) - len(expected_tokens) + 1):
        if word_tokens[start : start + len(expected_tokens)] == expected_tokens:
            return tuple(range(start, start + len(expected_tokens)))
    return None

def compute_word_priorities(
    words: Sequence[str],
    prompt_type: str,
    expected_trigger: str,
) -> List[float]:
    trigger_tokens = {
        token.lower() for token in tokenize(expected_trigger) if WORD_RE.match(token)
    }
    priorities: List[float] = []
    for word in words:
        lowered = word.lower()
        score = 0.0
        if lowered not in STOPWORDS:
            score += 2.0
        if prompt_type == "simple_harmful" and lowered in HARMFUL_HINTS:
            score += 3.0
        if lowered in trigger_tokens:
            score += 4.0
        priorities.append(score)
    return priorities


def candidate_prior_score(
    indices: Sequence[int],
    prompt_type: str,
    expected_trigger_indices: tuple[int, ...] | None,
    word_priorities: Sequence[float],
) -> float:
    score = sum(word_priorities[word_idx] for word_idx in indices)
    score += 0.5 * sum(1 for left, right in zip(indices, indices[1:]) if right == left + 1)
    if expected_trigger_indices and tuple(indices) == expected_trigger_indices:
        score += 5.0
    if prompt_type == "benign":
        score -= 0.1 * max(0, len(indices) - 1)
    return score


def generate_extensions(
    subsequence: tuple[int, ...],
    word_count: int,
    branching_factor: int,
    word_priorities: Sequence[float],
) -> List[tuple[int, ...]]:
    used = set(subsequence)
    expansions: List[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    min_idx = min(subsequence)
    max_idx = max(subsequence)

    def add_extension(word_idx: int) -> None:
        if word_idx in used:
            return
        extension = tuple(sorted((*subsequence, word_idx)))
        if extension in seen:
            return
        seen.add(extension)
        expansions.append(extension)

    if min_idx - 1 >= 0:
        add_extension(min_idx - 1)
    if max_idx + 1 < word_count:
        add_extension(max_idx + 1)

    center = (min_idx + max_idx) / 2
    ranked_candidates = sorted(
        (word_idx for word_idx in range(word_count) if word_idx not in used),
        key=lambda word_idx: (-word_priorities[word_idx], abs(word_idx - center), word_idx),
    )
    for word_idx in ranked_candidates:
        if len(expansions) >= branching_factor:
            break
        add_extension(word_idx)
    return expansions[:branching_factor]


def rank_beam_candidates(candidate_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        candidate_rows,
        key=lambda row: (
            -float(row["search_score"]),
            -float(row["candidate_prior_score"]),
            bool(row["candidate_is_contiguous"]),
            str(row["candidate_subsequence"]),
        ),
    )

def choose_mask_positions(
    tokens: Sequence[str],
    kept_token_indexes: set[int],
    mask_probability: float,
    rng: random.Random,
) -> List[int]:
    eligible = [
        idx
        for idx, token in enumerate(tokens)
        if idx not in kept_token_indexes and WORD_RE.match(token)
    ]
    if not eligible:
        return []
    chosen = [idx for idx in eligible if rng.random() < mask_probability]
    if not chosen:
        chosen = [rng.choice(eligible)]
    return chosen


def build_masked_template(tokens: Sequence[str], mask_positions: Sequence[int]) -> tuple[List[str], List[str]]:
    mask_lookup: Dict[int, str] = {}
    mask_ids: List[str] = []
    for number, token_idx in enumerate(mask_positions, start=1):
        mask_id = f"MASK{number}"
        mask_lookup[token_idx] = f"[{mask_id}]"
        mask_ids.append(mask_id)
    template = [mask_lookup.get(i, token) for i, token in enumerate(tokens)]
    return template, mask_ids

