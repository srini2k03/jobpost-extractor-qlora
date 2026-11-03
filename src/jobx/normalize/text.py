"""Text helpers: alias matchers shared by the validator, regex baseline and highlighter."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache


def squash(s: str) -> str:
    """Lower-case, NFKC, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def loose(s: str) -> str:
    """Lower-case and drop everything except letters, digits, + and #."""
    s = squash(s)
    s = re.sub(r"[^a-z0-9+#]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class AliasHit:
    canonical: str
    alias: str
    start: int
    end: int


def _alias_pattern(alias: str) -> str:
    esc = re.escape(alias)
    # word-ish boundaries that also work for aliases ending in punctuation ("C++", "B.E.").
    return rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"


class AliasMatcher:
    """Find canonical values in free text through their aliases (longest match wins)."""

    def __init__(self, aliases: Mapping[str, Iterable[str]], case_sensitive: Iterable[str] = ()) -> None:
        cs = set(case_sensitive)
        # Text patterns come from the aliases only: bare canonical names such as "go", "ca" or "10th"
        # would fire on ordinary words ("go ahead", "10th floor"). Canonical names stay in ``lookup``.
        items: list[tuple[str, str]] = sorted(
            {(a, canon) for canon, al in aliases.items() for a in al}, key=lambda x: (-len(x[0]), x[0])
        )
        self._ci: list[tuple[re.Pattern[str], str, str]] = []
        for alias, canon in items:
            flags = 0 if alias in cs else re.IGNORECASE
            self._ci.append((re.compile(_alias_pattern(alias), flags), alias, canon))
        self.lookup: dict[str, str] = {loose(canon): canon for canon in aliases}
        for alias, canon in items:
            self.lookup.setdefault(loose(alias), canon)

    def find(self, text: str) -> list[AliasHit]:
        taken: list[tuple[int, int]] = []
        hits: list[AliasHit] = []
        for pat, alias, canon in self._ci:
            for m in pat.finditer(text):
                s, e = m.span()
                if any(s < te and ts < e for ts, te in taken):
                    continue
                taken.append((s, e))
                hits.append(AliasHit(canon, alias, s, e))
        hits.sort(key=lambda h: h.start)
        return hits

    def canonicals(self, text: str) -> list[str]:
        seen: dict[str, None] = {}
        for h in self.find(text):
            seen.setdefault(h.canonical, None)
        return list(seen)

    def canonicalise(self, value: str) -> str | None:
        """Map a single value (e.g. a predicted skill) to its canonical form, or None."""
        return self.lookup.get(loose(value))


@lru_cache(maxsize=1)
def matchers() -> dict[str, AliasMatcher]:
    from jobx import vocab

    return {
        "skills": AliasMatcher(vocab.SKILL_ALIASES, vocab.CASE_SENSITIVE_ALIASES),
        "locations": AliasMatcher(vocab.CITY_ALIASES, vocab.CASE_SENSITIVE_CITY_ALIASES),
        "work_mode": AliasMatcher(vocab.WORK_MODE_ALIASES, vocab.CASE_SENSITIVE_WORK_MODE),
        "employment_type": AliasMatcher(vocab.EMPLOYMENT_TYPE_ALIASES),
        "education": AliasMatcher(vocab.EDUCATION_ALIASES, vocab.CASE_SENSITIVE_EDUCATION),
        "seniority": AliasMatcher(vocab.SENIORITY_LINE_FORMS),
    }
