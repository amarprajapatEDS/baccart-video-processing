"""Card class space — 13 ranks × 4 suits = 52 classes in [Rank][Suit] format."""
from __future__ import annotations

from typing import Dict, List

from config import CARD_RANKS, CARD_SUITS, all_card_labels


CARD_LABELS: List[str] = all_card_labels()
_LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(CARD_LABELS)}
_IDX_TO_LABEL: Dict[int, str] = {i: lbl for lbl, i in _LABEL_TO_IDX.items()}


def card_index(label: str) -> int:
    return _LABEL_TO_IDX[label]


def index_to_card(idx: int) -> str:
    return _IDX_TO_LABEL[idx]


def RANK_OF(label: str) -> str:
    return label[:-1]


def SUIT_OF(label: str) -> str:
    return label[-1]


def is_valid_label(label: str) -> bool:
    return label in _LABEL_TO_IDX


assert len(CARD_LABELS) == 52, f"expected 52 cards, got {len(CARD_LABELS)}"
assert set(CARD_RANKS) == set(RANK_OF(l) for l in CARD_LABELS)
assert set(CARD_SUITS) == set(SUIT_OF(l) for l in CARD_LABELS)
