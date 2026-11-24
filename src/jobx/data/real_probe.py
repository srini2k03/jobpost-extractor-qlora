"""Probe public sources of REAL job postings for the held-out real-world test set.

Runs where the Hugging Face Hub is reachable (GitHub Actions). Two probes, both licence-first:

* ``datasets``: Hub search for job-posting datasets plus the candidates in ``configs/real_data.yaml``.
  For each dataset the declared licence (card metadata) and the gating flag are read first; rows are
  only streamed for datasets whose licence is in the allowlist and that are not gated. The report
  records counts and length statistics, plus at most two 160-character excerpts per allowed dataset.
* ``web``: a URL-domain scan of an ODC-By web corpus (FineWeb sample) for pages on Indian job boards.
  Only the ``url`` column is read (parquet column projection over HTTP range requests); the text of
  matching rows is read only by :func:`fetch_web_pool`.

Nothing here decides on its own that a source is usable: the report is committed, a person (or the
upgrade agent) picks a source, and the choice + licence are recorded in ``data/real/real_meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from jobx.data.fetch_real import looks_indian
from jobx.normalize.text import loose

SEARCH_TERMS: tuple[str, ...] = (
    "job postings",
    "job posting",
    "job descriptions",
    "job description",
    "job listings",
    "job ads",
    "linkedin jobs",
    "naukri",
    "indeed jobs",
    "india jobs",
    "vacancies",
    "recruitment",
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def card_licence(card: Any) -> str | None:
    """The ``license`` field of a dataset card (first entry when it is a list), lower-cased."""
    if card is None:
        return None
    lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return str(lic).lower() if lic else None


def licence_from_tags(tags: Sequence[str] | None) -> str | None:
    for t in tags or ():
        if t.startswith("license:"):
            return t.split(":", 1)[1].lower()
    return None


def guess_text_column(rows: Sequence[dict[str, Any]]) -> str | None:
    """The string column with the longest mean length (job descriptions are the longest field)."""
    if not rows:
        return None
    lens: dict[str, list[int]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str):
                lens.setdefault(k, []).append(len(v))
    if not lens:
        return None
    return max(lens, key=lambda k: float(np.mean(lens[k])))


def _quantiles(vals: Sequence[int]) -> dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 1) for q in (5, 25, 50, 75, 95)}


def row_stats(
    rows: Sequence[dict[str, Any]], text_col: str, *, min_chars: int = 200, max_chars: int = 3000
) -> dict[str, Any]:
    texts = [str(r.get(text_col) or "") for r in rows]
    in_range = [t for t in texts if min_chars <= len(t) <= max_chars]
    indian = [t for t in in_range if looks_indian(t)]
    uniq = {hashlib.sha1(loose(t).encode()).hexdigest() for t in texts if t}
    return {
        "rows_read": len(rows),
        "chars": _quantiles([len(t) for t in texts]),
        "in_length_range": len(in_range),
        "indian_in_length_range": len(indian),
        "unique_texts": len(uniq),
    }


def excerpt(text: str, n: int = 160) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:n] + ("…" if len(t) > n else "")


def probe_datasets(
    candidates: Iterable[str],
    allow: set[str],
    *,
    search: bool = True,
    search_limit: int = 40,
    rows_per_dataset: int = 2000,
    max_streamed: int = 25,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    ids: dict[str, str] = {c: "config" for c in candidates}
    search_log: dict[str, int] = {}
    if search:
        for term in SEARCH_TERMS:
            try:
                found = list(api.list_datasets(search=term, limit=search_limit, sort="downloads"))
            except Exception as e:  # network or API change: record and continue
                search_log[term] = -1
                print(f"search {term!r} failed: {type(e).__name__}: {e}")
                continue
            search_log[term] = len(found)
            for d in found:
                ids.setdefault(d.id, f"search:{term}")
    report: list[dict[str, Any]] = []
    streamed = 0
    for ds_id, origin in ids.items():
        entry: dict[str, Any] = {"id": ds_id, "found_by": origin}
        try:
            info = api.dataset_info(ds_id)
        except Exception as e:
            entry["error"] = f"dataset_info: {type(e).__name__}"
            report.append(entry)
            continue
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        lic = card_licence(card) or licence_from_tags(getattr(info, "tags", None))
        entry.update(
            {
                "licence": lic,
                "gated": bool(getattr(info, "gated", False)),
                "private": bool(getattr(info, "private", False)),
                "downloads": getattr(info, "downloads", None),
                "sha": getattr(info, "sha", None),
                "last_modified": str(getattr(info, "last_modified", "") or ""),
                "tags": [t for t in (getattr(info, "tags", None) or []) if not t.startswith("region:")][:20],
            }
        )
        ok_licence = lic in allow
        entry["licence_ok"] = ok_licence
        if not ok_licence or entry["gated"] or entry["private"]:
            entry["rows"] = "not read (licence not in allowlist, gated or private)"
            report.append(entry)
            continue
        if streamed >= max_streamed:
            entry["rows"] = "not read (stream budget used)"
            report.append(entry)
            continue
        streamed += 1
        try:
            rows = _stream_rows(ds_id, rows_per_dataset)
        except Exception as e:
            entry["rows"] = f"stream failed: {type(e).__name__}: {str(e)[:200]}"
            report.append(entry)
            continue
        col = guess_text_column(rows)
        entry["columns"] = sorted(rows[0].keys())[:40] if rows else []
        entry["text_column"] = col
        if col:
            entry["stats"] = row_stats(rows, col)
            entry["excerpts"] = [excerpt(str(r.get(col) or "")) for r in rows[:2]]
        report.append(entry)
    return {
        "probed_at": _now(),
        "licence_allowlist": sorted(allow),
        "search_terms": search_log,
        "n_datasets": len(report),
        "datasets": sorted(report, key=lambda e: (not e.get("licence_ok", False), e["id"])),
    }


LOCATION_HINTS = ("location", "city", "country", "region", "place", "state")


def india_row(row: dict[str, Any], text_col: str) -> bool:
    """India by a location-like column when there is one, else by the text heuristic."""
    from jobx.normalize.text import matchers

    locs = [
        str(v) for k, v in row.items() if isinstance(v, str) and any(h in k.lower() for h in LOCATION_HINTS)
    ]
    if locs:
        joined = " ".join(locs)
        return bool(re.search(r"\b(?:[Ii]ndia|INDIA|IN|IND)\b", joined)) or bool(
            matchers()["locations"].find(joined)
        )
    return looks_indian(str(row.get(text_col) or ""))


def probe_details(ids: Sequence[str], *, max_rows: int = 60000, excerpts: int = 3) -> dict[str, Any]:
    """Card text (provenance), columns, and India-row counts for a shortlist of allowed datasets."""
    from huggingface_hub import DatasetCard, HfApi

    api = HfApi()
    out: list[dict[str, Any]] = []
    for ds_id in ids:
        entry: dict[str, Any] = {"id": ds_id}
        try:
            info = api.dataset_info(ds_id)
            entry["licence"] = card_licence(getattr(info, "card_data", None)) or licence_from_tags(
                getattr(info, "tags", None)
            )
            entry["sha"] = getattr(info, "sha", None)
            entry["gated"] = bool(getattr(info, "gated", False))
            try:
                entry["card_excerpt"] = re.sub(r"\s+", " ", DatasetCard.load(ds_id).text)[:2500]
            except Exception as e:
                entry["card_excerpt"] = f"card not readable: {type(e).__name__}"
            rows = _stream_rows(ds_id, max_rows)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out.append(entry)
            continue
        col = guess_text_column(rows)
        entry["columns"] = sorted(rows[0].keys())[:60] if rows else []
        entry["text_column"] = col
        entry["rows_read"] = len(rows)
        if col:
            ind = [r for r in rows if india_row(r, col)]
            lens = [len(str(r.get(col) or "")) for r in ind]
            entry["india_rows"] = len(ind)
            entry["india_chars"] = _quantiles(lens)
            entry["india_in_300_3500"] = sum(1 for n in lens if 300 <= n <= 3500)
            entry["india_excerpts"] = [excerpt(str(r.get(col) or ""), 220) for r in ind[:excerpts]]
            entry["india_meta_examples"] = [
                {k: (str(v)[:80] if v is not None else None) for k, v in r.items() if k != col}
                for r in ind[:2]
            ]
        out.append(entry)
    return {"probed_at": _now(), "datasets": out}


def _stream_rows(ds_id: str, n: int) -> list[dict[str, Any]]:
    from datasets import get_dataset_split_names, load_dataset

    split = "train"
    try:
        names = get_dataset_split_names(ds_id)
        split = "train" if "train" in names else names[0]
    except Exception:
        pass
    ds = load_dataset(ds_id, split=split, streaming=True)
    out: list[dict[str, Any]] = []
    for row in ds:
        out.append({k: v for k, v in row.items() if isinstance(v, str | int | float | bool) or v is None})
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------------------------------------
# Web corpus: Indian job-board pages in an ODC-By crawl (FineWeb sample).

JOB_DOMAINS: tuple[str, ...] = (
    "naukri.com",
    "timesjobs.com",
    "shine.com",
    "foundit.in",
    "monsterindia.com",
    "freshersworld.com",
    "indeed.co.in",
    "in.indeed.com",
    "internshala.com",
    "apna.co",
    "hirist.com",
    "hirist.tech",
    "iimjobs.com",
    "instahyre.com",
    "cutshort.io",
    "jobsforher.com",
    "careerjet.co.in",
    "in.jooble.org",
    "glassdoor.co.in",
    "workindia.in",
    "placementindia.com",
    "clickindia.com",
    "fresherslive.com",
    "freshersnow.com",
    "freshershunt.in",
    "jobhai.com",
    "updazz.com",
    "firstnaukri.com",
    "headhonchos.com",
    "babajob.com",
    "olx.in",
    "quikr.com",
    "indeedjobs.com",
    "careers360.com",
    "angel.co",
    "wellfound.com",
)
JOB_PATH_RE = re.compile(r"/(job|jobs|career|careers|vacanc|opening|hiring|requirement|walk-?in)", re.I)


def match_job_domain(url: str) -> str | None:
    """The job-board domain a URL belongs to (suffix match on the host), else None."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    for d in JOB_DOMAINS:
        if host == d or host.endswith("." + d):
            return d
    return None


def _hf_parquet_files(repo: str, pattern: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return sorted(fs.glob(f"datasets/{repo}/{pattern}"))


def scan_web_urls(
    repo: str, pattern: str, *, max_files: int = 4, url_col: str = "url"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read only the URL column of up to ``max_files`` parquet files; return a report and the hits."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = _hf_parquet_files(repo, pattern)[:max_files]
    hits: list[dict[str, Any]] = []
    per_domain: Counter[str] = Counter()
    n_rows = 0
    for path in files:
        with fs.open(path, "rb", block_size=8 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            offset = 0
            for rg in range(pf.num_row_groups):
                col = pf.read_row_group(rg, columns=[url_col]).column(0).to_pylist()
                for i, u in enumerate(col):
                    d = match_job_domain(str(u or ""))
                    if d:
                        per_domain[d] += 1
                        hits.append({"file": path, "row_group": rg, "row": offset + i, "url": u, "domain": d})
                offset += len(col)
                n_rows += len(col)
        print(f"{path}: {offset} rows, {len(hits)} hits so far")
    report = {
        "repo": repo,
        "pattern": pattern,
        "files_scanned": files,
        "rows_scanned": n_rows,
        "hits": len(hits),
        "hits_with_job_path": sum(1 for h in hits if JOB_PATH_RE.search(urlparse(h["url"]).path or "")),
        "per_domain": dict(per_domain.most_common()),
    }
    return report, hits


def with_backoff(fn: Any, *, attempts: int = 6, base_s: float = 60.0) -> Any:
    """Call ``fn()``; on HTTP 429 / rate-limit errors wait ``base_s * attempt`` seconds and retry."""
    import time

    for k in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # huggingface_hub wraps requests errors in several types
            if k == attempts or not re.search(r"429|Too Many Requests|rate limit", str(e), re.I):
                raise
            print(f"rate limited ({type(e).__name__}); retry {k}/{attempts - 1} in {base_s * k:.0f}s")
            time.sleep(base_s * k)
    raise RuntimeError("unreachable")


def sample_hits(
    hits: Sequence[dict[str, Any]], *, n: int, per_domain: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministic sample of URL hits with a per-domain cap (before any text is read)."""
    rng = random.Random(seed)
    order = list(hits)
    rng.shuffle(order)
    per: Counter[str] = Counter()
    out = []
    for h in order:
        if per[h["domain"]] < per_domain:
            out.append(h)
            per[h["domain"]] += 1
        if len(out) >= n:
            break
    return out


def read_hit_texts(hits: Sequence[dict[str, Any]], *, text_col: str = "text") -> Iterator[dict[str, Any]]:
    """Read the text of the given hits, one row group at a time (text column only), with backoff."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for h in hits:
        by_group.setdefault((h["file"], int(h["row_group"])), []).append(h)
    for (path, rg), hs in sorted(by_group.items()):

        def _read(path: str = path, rg: int = rg) -> tuple[int, Any]:
            with fs.open(path, "rb", block_size=8 * 1024 * 1024) as fh:
                pf = pq.ParquetFile(fh)
                start = sum(pf.metadata.row_group(i).num_rows for i in range(rg))
                return start, pf.read_row_group(rg, columns=[text_col, "id", "date", "dump"])

        start, tbl = with_backoff(_read)
        texts = tbl.column(text_col).to_pylist()
        ids = tbl.column("id").to_pylist()
        dates = tbl.column("date").to_pylist()
        dumps = tbl.column("dump").to_pylist()
        for h in hs:
            j = int(h["row"]) - start
            yield {**h, "text": texts[j], "doc_id": ids[j], "crawl_date": dates[j], "dump": dumps[j]}


def is_job_post(text: str) -> bool:
    """Cheap filter for single job advertisements (vs. listing/index pages and articles)."""
    t = text.lower()
    cues = [
        r"\bexperience\b|\bexp\b|\byrs\b|\byears\b",
        r"\bskills?\b|\brequirements?\b|\bqualification|\beligibility\b|\bresponsibilit",
        r"\bsalary\b|\bctc\b|\blpa\b|\bpackage\b|\bstipend\b|₹|\binr\b|not disclosed",
        r"\blocation\b|\bjob location\b|\bbased in\b|\bcity\b",
        r"\bapply\b|\bjob (description|summary|role)\b|\bhiring\b|\bvacanc|\bopening",
    ]
    hits = sum(1 for c in cues if re.search(c, t))
    many_jobs = len(re.findall(r"\bapply now\b|\bview job\b|\bposted \d+ days? ago\b", t)) >= 4
    return hits >= 4 and not many_jobs


def build_pool(
    rows: Iterable[dict[str, Any]],
    *,
    n_pool: int,
    seed: int,
    min_chars: int = 300,
    max_chars: int = 3500,
    max_per_domain: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter to single job ads in a length range, exact-dedup, cap per domain, sample ``n_pool``."""
    seen: set[str] = set()
    keep: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for r in rows:
        text = re.sub(r"\n{3,}", "\n\n", str(r.get("text") or "")).strip()
        if not (min_chars <= len(text) <= max_chars):
            drops["length"] += 1
            continue
        if not is_job_post(text):
            drops["not_a_single_job_ad"] += 1
            continue
        h = hashlib.sha1(loose(text).encode()).hexdigest()
        if h in seen:
            drops["exact_duplicate"] += 1
            continue
        seen.add(h)
        keep.append({**r, "text": text})
    rng = random.Random(seed)
    rng.shuffle(keep)
    if max_per_domain:
        per: Counter[str] = Counter()
        capped = []
        for r in keep:
            if per[r["domain"]] < max_per_domain:
                capped.append(r)
                per[r["domain"]] += 1
        keep = capped
    pool = keep[:n_pool]
    stats = {
        "candidates_after_filters": len(keep),
        "dropped": dict(drops),
        "pool": len(pool),
        "pool_per_domain": dict(Counter(r["domain"] for r in pool).most_common()),
    }
    return pool, stats


def load_real_config(path: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return cfg


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["datasets", "details", "web", "web-pool"])
    ap.add_argument("--config", default="configs/real_data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--pool-out", default=None, help="web-pool: JSONL of sampled real postings")
    ap.add_argument(
        "--hits", default="results/real/web_hits.jsonl", help="URL hits (written by web, reused by web-pool)"
    )
    a = ap.parse_args(argv)
    cfg = load_real_config(a.config)
    allow = set(cfg["licence_allowlist"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.command == "datasets":
        rep = probe_datasets([c["id"] for c in cfg["candidates"]], allow)
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
        return 0
    if a.command == "details":
        rep = probe_details(cfg["shortlist"])
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
        return 0
    web = cfg["web_corpus"]
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(web["repo"])
    lic = card_licence(getattr(info, "card_data", None)) or licence_from_tags(getattr(info, "tags", None))
    if lic not in allow:
        out.write_text(
            json.dumps({"repo": web["repo"], "licence": lic, "skipped": "licence"}, indent=2) + "\n"
        )
        return 0
    hits_path = Path(a.hits)
    if a.command == "web-pool" and hits_path.exists():
        prev = json.loads(out.read_text()) if out.exists() else {}
        rep = {k: v for k, v in prev.items() if k != "pool"}
        hits = [json.loads(x) for x in hits_path.read_text().splitlines() if x.strip()]
        print(f"reusing {len(hits)} URL hits from {hits_path}")
    else:
        rep, hits = scan_web_urls(web["repo"], web["pattern"], max_files=a.max_files or web["max_files"])
        rep.update({"licence": lic, "sha": getattr(info, "sha", None), "probed_at": _now()})
        hits_path.parent.mkdir(parents=True, exist_ok=True)
        hits_path.write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in hits))
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    if a.command == "web-pool":
        chosen = sample_hits(
            hits, n=int(web["n_read"]), per_domain=int(web["max_per_domain"]), seed=int(cfg["seed"])
        )
        rep["hits_read"] = len(chosen)
        pool, pstats = build_pool(
            read_hit_texts(chosen),
            n_pool=web["n_pool"],
            seed=cfg["seed"],
            min_chars=web["min_chars"],
            max_chars=web["max_chars"],
            max_per_domain=web.get("max_per_domain"),
        )
        rep["pool"] = pstats
        if a.pool_out:
            p = Path(a.pool_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for k, r in enumerate(sorted(pool, key=lambda r: r["doc_id"])):
                    rec = {
                        "id": f"real-web-{k:04d}",
                        "text": r["text"],
                        "source": {
                            "corpus": web["repo"],
                            "corpus_revision": rep["sha"],
                            "corpus_licence": lic,
                            "doc_id": r["doc_id"],
                            "url": r["url"],
                            "domain": r["domain"],
                            "crawl": r["dump"],
                            "crawl_date": r["crawl_date"],
                        },
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
