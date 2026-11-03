"""Exact and near-duplicate removal with MinHash + LSH (numpy only, deterministic).

Shingles are word n-grams of the normalised text, hashed with CRC32. Signatures use
universal hashing ``(a*x + b) mod p`` with p = 2**32 + 15 (all products fit in uint64).
LSH banding proposes candidate pairs; each candidate is confirmed with exact Jaccard.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from jobx.normalize.text import loose

_P = np.uint64((1 << 32) + 15)


def shingles(text: str, n: int = 3) -> set[int]:
    toks = re.findall(r"[a-z0-9+#]+", loose(text))
    if len(toks) < n:
        return {zlib.crc32(" ".join(toks).encode())}
    return {zlib.crc32(" ".join(toks[i : i + n]).encode()) for i in range(len(toks) - n + 1)}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, (1 << 32) - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 32) - 1, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: set[int]) -> np.ndarray:
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))[:, None]
        h = (x * self.a[None, :] + self.b[None, :]) % _P
        sig: np.ndarray = h.min(axis=0)
        return sig


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class DedupResult:
    keep: list[int]
    exact_dups: list[tuple[int, int]] = field(default_factory=list)  # (removed, kept)
    near_dups: list[tuple[int, int, float]] = field(default_factory=list)  # (removed, kept, jaccard)


def dedup(
    texts: Sequence[str],
    *,
    priority: Sequence[int] | None = None,
    num_perm: int = 128,
    bands: int = 32,
    shingle_words: int = 3,
    threshold: float = 0.8,
) -> DedupResult:
    """Remove exact and near duplicates.

    ``priority``: lower value wins when two items collide (e.g. test=0, val=1, train=2 so a
    train item that near-duplicates a test item is the one removed). Ties keep the earlier index.
    """
    n = len(texts)
    prio = list(priority) if priority is not None else [0] * n
    order = sorted(range(n), key=lambda i: (prio[i], i))
    res = DedupResult(keep=[])
    seen_exact: dict[str, int] = {}
    removed: set[int] = set()
    for i in order:
        key = hashlib.sha1(loose(texts[i]).encode()).hexdigest()
        if key in seen_exact:
            res.exact_dups.append((i, seen_exact[key]))
            removed.add(i)
        else:
            seen_exact[key] = i
    rows = num_perm // bands
    mh = MinHasher(num_perm)
    sh = {i: shingles(texts[i], shingle_words) for i in order if i not in removed}
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    kept: list[int] = []
    for i in order:
        if i in removed:
            continue
        sig = mh.signature(sh[i])
        cands: set[int] = set()
        keys = [(b, sig[b * rows : (b + 1) * rows].tobytes()) for b in range(bands)]
        for k in keys:
            cands.update(buckets.get(k, ()))
        best: tuple[int, float] | None = None
        for j in cands:
            jac = jaccard(sh[i], sh[j])
            if jac >= threshold and (best is None or jac > best[1]):
                best = (j, jac)
        if best is not None:
            res.near_dups.append((i, best[0], round(best[1], 4)))
            removed.add(i)
            continue
        kept.append(i)
        for k in keys:
            buckets[k].append(i)
    res.keep = sorted(kept)
    return res
