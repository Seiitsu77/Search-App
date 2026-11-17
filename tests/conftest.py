"""Test helpers: a deterministic bag-of-words embedder that needs no model download."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIM = 384


def hash_embed(texts: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            out[row, int.from_bytes(digest[:4], "little") % DIM] += 1.0
        norm = np.linalg.norm(out[row])
        if norm:
            out[row] /= norm
    return out


@pytest.fixture
def embedder():
    return hash_embed
