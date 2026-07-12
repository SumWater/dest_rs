from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


CANONICAL_LABELS = (
    "elicit-pref",
    "no-need",
    "other-need",
    "promote-coordination",
    "self-need",
    "showing-empathy",
    "small-talk",
    "uv-part",
    "vouch-fair",
)


def load_canonical_labels(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical strategy label-space does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
        raise ValueError(f"Invalid canonical label-space in {path}: labels must be non-empty strings")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate canonical strategy labels in {path}: {labels}")
    if tuple(labels) != CANONICAL_LABELS:
        raise ValueError(
            f"Canonical label-space order/content changed in {path}. "
            f"Expected {list(CANONICAL_LABELS)}, got {labels}"
        )
    return labels


def validate_observed_labels(
    observed: Iterable[str], canonical: Sequence[str], *, require_all: bool, source: str
) -> None:
    observed_set = set(observed)
    canonical_set = set(canonical)
    unknown = sorted(observed_set - canonical_set)
    missing = sorted(canonical_set - observed_set)
    if unknown:
        raise ValueError(f"Unknown strategies in {source}: {unknown}")
    if require_all and missing:
        raise ValueError(f"Canonical strategies missing from {source}: {missing}")


def assert_exact_label_order(actual: Sequence[str], expected: Sequence[str], *, source: str) -> None:
    if list(actual) != list(expected):
        raise ValueError(
            f"Strategy label order mismatch for {source}. "
            f"Expected {list(expected)}, got {list(actual)}"
        )
