"""Static results page for GitHub Pages (no extra dependencies).

The page follows the design language of the author's portfolio (https://srini2k03.github.io/): the shared
kit (``web/sb-kit.css`` / ``web/sb-kit.js``, copied unchanged) plus this project's own ``web/site.css``.
Every number on the page is read from ``results/results.json`` / ``results/results_table.md`` at build
time; nothing here is typed in by hand.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

WEB = Path(__file__).resolve().parents[1] / "web"
STATIC_FILES = ("sb-kit.css", "sb-kit.js", "site.css", "favicon.svg")
REPO = "https://github.com/srini2k03/jobpost-extractor-qlora"
RELEASE = f"{REPO}/releases/tag/model-v1"
PORTFOLIO = "https://srini2k03.github.io/"


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", s)


_NUM = re.compile(r"^-?\d+(\.\d+)?( \[-?\d+(\.\d+)?, -?\d+(\.\d+)?\])?$")


def _heat(cell: str) -> str:
    """Inline style for a per-field heat-map cell: stronger tint for higher exact match (40-100 %)."""
    try:
        v = float(cell)
    except ValueError:
        return ""
    return f' style="--h:{max(0.0, min(1.0, (v - 40) / 60)) * 42:.0f}%"'


def md_to_html(md: str, *, styled: bool = False, heading_shift: int = 0) -> str:
    """Render the small markdown subset of results_table.md.

    ``styled`` adds the design-kit table classes (scroll wrapper, numeric cells, "ours" rows, a heat map for
    per-field tables, collapsed all-pending tables); ``heading_shift`` moves heading levels (-1: #### -> h3).
    """
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    heading = ""
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if not styled:
                out.append(
                    "<table><thead><tr>"
                    + "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                    + "</tr></thead><tbody>"
                )
                for r in rows[2:]:
                    out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
                out.append("</tbody></table>")
                continue
            heat = heading.lower().startswith("per-field")
            body = rows[2:]
            numeric = [
                j > 0 and all(_NUM.match(r[j]) or r[j] == "-" for r in body if j < len(r))
                for j in range(len(rows[0]))
            ]
            cls = "sb-table jx-table" + (" jx-heat" if heat else "")
            t = [f'<div class="sb-table-wrap"><table class="{cls}"><thead><tr>']
            t += [
                f'<th scope="col"{" class=num" if numeric[j] else ""}>{_inline(c)}</th>'
                for j, c in enumerate(rows[0])
            ]
            t.append("</tr></thead><tbody>")
            for r in body:
                ours = "(ours)" in r[0]
                t.append(f"<tr{' class=is-ours' if ours else ''}>")
                for j, c in enumerate(r):
                    if j == 0:
                        t.append(f'<th scope="row">{_inline(c)}</th>')
                    elif j < len(numeric) and numeric[j]:
                        t.append(f'<td class="num"{_heat(c) if heat else ""}>{_inline(c)}</td>')
                    elif c.startswith("pending") or c.startswith("incomplete"):
                        t.append(f'<td><span class="jx-status jx-status--pending">{_inline(c)}</span></td>')
                    elif c.startswith("measured"):
                        t.append(f'<td><span class="jx-status">{_inline(c)}</span></td>')
                    else:
                        t.append(f"<td>{_inline(c)}</td>")
                t.append("</tr>")
            t.append("</tbody></table></div>")
            status_col = len(rows[0]) - 1 if rows[0][-1].lower() == "status" else None
            if status_col is not None and body and all(r[status_col].startswith("pending") for r in body):
                out.append(
                    f'<details class="jx-pending"><summary>Show the {len(body)} pending rows</summary>'
                    + "".join(t)
                    + "</details>"
                )
            else:
                out.append("".join(t))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            lvl = max(1, min(6, len(m.group(1)) + heading_shift))
            heading = m.group(2)
            cls = ' class="jx-table-title"' if styled else ""
            out.append(f"<h{lvl}{cls}>{_inline(m.group(2))}</h{lvl}>")
        elif ln.strip():
            txt = ln.strip()
            if styled and len(txt) > 2 and txt[0] == txt[-1] == "_":
                out.append(f'<p class="jx-note"><em>{_inline(txt[1:-1])}</em></p>')
            else:
                out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


WLLAMA_VERSION = "3.8.1"  # exact npm version of @wllama/wllama loaded from jsDelivr

# Placeholders are <!--NAME--> comments filled by build_site (no str.format: the page holds CSS/JS braces).
PAGE = """<!doctype html>
<html lang="en" data-theme="dark" class="no-js">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job posting to JSON</title>
<meta name="description" content="A LoRA fine-tune of Qwen2.5-0.5B that turns messy Indian job posts into one strict JSON object, running in your browser with llama.cpp (WebAssembly).">
<meta name="color-scheme" content="dark light">
<script>(function(){try{var t=localStorage.getItem('theme');document.documentElement.dataset.theme=(t==='light'||t==='dark')?t:'dark';}catch(e){document.documentElement.dataset.theme='dark';}document.documentElement.classList.remove('no-js');})();</script>
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400..600&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<link rel="stylesheet" href="sb-kit.css">
<link rel="stylesheet" href="site.css">
<script src="sb-kit.js" defer></script>
</head>
<body class="sb-grain">
<div class="sb-backdrop" aria-hidden="true"></div>
<a class="jx-skip" href="#main">Skip to content</a>
<header class="sb-nav">
  <div class="sb-container sb-nav__inner">
    <a class="sb-brand" href="https://srini2k03.github.io/" aria-label="Srinivasan B., portfolio">
      <span class="sb-brand__mark" aria-hidden="true">SB</span>
      <span>Srinivasan B.<span class="sb-brand__sub jx-brand-sub"> / Job-post &rarr; JSON</span></span>
    </a>
    <nav class="sb-nav__links" aria-label="Sections">
      <a class="sb-nav__link sb-nav__link--hide-sm" href="#demo">Demo</a>
      <a class="sb-nav__link sb-nav__link--hide-sm" href="#results">Results</a>
      <a class="sb-nav__link sb-nav__link--hide-sm" href="#caveats">Caveats</a>
      <a class="sb-nav__link" href="https://github.com/srini2k03/jobpost-extractor-qlora">Code</a>
      <button class="sb-icon-btn jx-theme" type="button" data-theme-toggle aria-label="Switch to light theme">
        <svg class="jx-i-sun" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <svg class="jx-i-moon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>
      </button>
    </nav>
  </div>
</header>

<main id="main">
<section class="sb-hero" aria-labelledby="page-title">
  <div class="sb-container">
    <p class="sb-kicker"><span class="sb-kicker__index">LLM fine-tuning</span><span class="sb-kicker__rule"></span><span>LoRA<span class="jx-hide-xs"> &middot; llama.cpp</span> &middot; 2026</span></p>
    <h1 id="page-title" class="sb-h1">Messy job posts in,<br> strict <span class="sb-serif">JSON</span> out.</h1>
    <p class="sb-lead">A LoRA fine-tune of Qwen2.5-0.5B-Instruct turns WhatsApp forwards, Hinglish and &ldquo;12&nbsp;LPA&rdquo; salaries
      into one strict 11-field JSON object, and returns <code>null</code> for anything the post does not state. Trained on free
      GitHub-hosted CPU runners.<!--LEAD_TAIL--></p>
    <div class="sb-hero__meta"><!--PILLS--></div>
    <div class="sb-hero__actions">
      <a class="sb-btn sb-btn--primary" href="<!--PRIMARY_HREF-->"><!--PRIMARY_TEXT-->
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" class="jx-ico"><path d="M12 5v14M6 13l6 6 6-6"/></svg></a>
      <a class="sb-btn" href="https://github.com/srini2k03/jobpost-extractor-qlora">Code
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" class="jx-ico"><path d="M7 17 17 7M8 7h9v9"/></svg></a>
      <a class="sb-btn sb-btn--ghost" href="https://github.com/srini2k03/jobpost-extractor-qlora/releases/tag/model-v1">Model files
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" class="jx-ico"><path d="M7 17 17 7M8 7h9v9"/></svg></a>
    </div>
    <dl class="sb-stats jx-stats"><!--STATS--></dl>
    <p class="jx-stats-note"><!--STATS_NOTE--></p>
  </div>
</section>

<!--DEMO-->

<section id="results" class="sb-section" aria-labelledby="results-title">
  <div class="sb-container">
    <p class="sb-kicker sb-reveal"><span class="sb-kicker__index">02</span><span class="sb-kicker__rule"></span>Results</p>
    <h2 id="results-title" class="sb-h2 sb-reveal">Measured by CI, not <span class="sb-serif">estimated</span>.</h2>
    <p class="sb-lead sb-reveal">Generated by CI from committed prediction files. Rows marked &ldquo;pending&rdquo; have not been run
      yet; no number on this page is estimated. <!--SPLIT_NOTE--></p>
    <div class="jx-results">
<!--CHARTS-->
    </div>
  </div>
</section>

<section id="tables" class="sb-section jx-tables-section" aria-labelledby="tables-title">
  <div class="sb-container">
    <p class="sb-kicker sb-reveal"><span class="sb-kicker__index">03</span><span class="sb-kicker__rule"></span>All results</p>
    <h2 id="tables-title" class="sb-h2 sb-reveal">Every row, every <span class="sb-serif">field</span>.</h2>
    <p class="sb-lead sb-reveal">The tables from the README, regenerated by <code>jobx results-table</code> from
      <code>results/metrics</code> and <code>results/benchmarks</code>. Percentages, with 95% bootstrap CIs (1,000 resamples).</p>
    <div class="sb-card jx-tables">
<!--BODY-->
    </div>
<!--CURVES-->
  </div>
</section>

<section id="caveats" class="sb-section" aria-labelledby="caveats-title">
  <div class="sb-container">
    <p class="sb-kicker sb-reveal"><span class="sb-kicker__index">04</span><span class="sb-kicker__rule"></span>Caveats</p>
    <h2 id="caveats-title" class="sb-h2 sb-reveal">What these numbers <span class="sb-serif">don&rsquo;t</span> say.</h2>
    <div class="sb-callout sb-callout--warn jx-caveats sb-reveal">
      <ul>
<!--CAVEATS-->
      </ul>
    </div>
  </div>
</section>
</main>

<footer class="sb-footer">
  <div class="sb-container sb-footer__inner">
    <div>
      <p class="jx-foot-name">&copy; 2026 Srinivasan Balasubramanian</p>
      <p class="jx-foot-how">Generated by <code>jobx site</code> from the committed results and published to GitHub Pages by
        GitHub Actions; when the model is present, a headless-Chromium check loads it and extracts valid JSON before publishing.</p>
    </div>
    <ul class="jx-foot-links">
      <li><a href="https://srini2k03.github.io/">Portfolio</a></li>
      <li><a href="https://github.com/srini2k03/jobpost-extractor-qlora">Code</a></li>
      <li><a href="https://github.com/srini2k03/jobpost-extractor-qlora/blob/main/MODEL_CARD.md">Model card</a></li>
      <li><a href="https://github.com/srini2k03/jobpost-extractor-qlora/blob/main/DATA_CARD.md">Data card</a></li>
    </ul>
  </div>
</footer>
</body>
</html>
"""

FIELD_LEGEND = (
    ("company", "company"),
    ("role", "role"),
    ("skill", "skills"),
    ("location", "locations"),
    ("work_mode", "work mode"),
    ("employment_type", "employment type"),
    ("education", "education"),
)

DEMO = """<section id="demo" class="sb-section jx-demo-section" aria-labelledby="demo-title">
  <div class="sb-container">
    <p class="sb-kicker sb-reveal"><span class="sb-kicker__index">01</span><span class="sb-kicker__rule"></span>Live demo</p>
    <h2 id="demo-title" class="sb-h2 sb-reveal">Try it in your <span class="sb-serif">browser</span>.</h2>
    <p class="sb-lead sb-reveal">The model runs entirely on your device with <a href="https://github.com/ngxson/wllama">wllama</a>
      (llama.cpp compiled to WebAssembly); nothing is sent to a server. First load: <span id="size-note">the model</span>.
      Inference is single-threaded (GitHub Pages cannot send the headers multi-threading needs), so an extraction takes
      roughly 20-60 s.</p>

    <div class="sb-card jx-demo">
      <ol class="jx-steps">
        <li class="jx-step" id="step-load">
          <span class="jx-step__n" aria-hidden="true">1</span>
          <div class="jx-step__main">
            <button id="load" class="sb-btn sb-btn--primary" type="button">Load model</button>
            <span class="jx-step__note">~<!--APPROX_MB--> MB, cached after first load</span>
          </div>
          <div class="jx-progress" aria-hidden="true">
            <div class="sb-meter"><span id="progress-bar" style="width:0%"></span></div>
            <span class="jx-progress__pct" id="progress-pct">0%</span>
          </div>
        </li>
        <li class="jx-step" id="step-extract">
          <span class="jx-step__n" aria-hidden="true">2</span>
          <div class="jx-step__main">
            <button id="extract" class="sb-btn" type="button" disabled>Extract</button>
            <label class="jx-check"><input type="checkbox" id="guided" checked>
              <span>JSON-schema grammar <span class="sb-faint">(guaranteed valid JSON)</span></span></label>
          </div>
        </li>
      </ol>
      <div id="status" class="jx-status-line" role="status" aria-live="polite" data-state="">Click "Load model" to start.</div>

      <div class="jx-panes">
        <div class="jx-pane jx-pane--in">
          <div class="jx-pane__head">
            <span class="sb-label" id="posting-label">Job posting</span>
            <div class="jx-view" role="group" aria-label="Left pane view">
              <button type="button" class="jx-view__btn" id="view-edit" aria-pressed="true">Edit</button>
              <button type="button" class="jx-view__btn" id="view-marks" aria-pressed="false" disabled>Highlights</button>
            </div>
          </div>
          <div id="examples" class="jx-chips" role="group" aria-label="Example postings"></div>
          <div class="jx-pane__body">
            <textarea id="posting" class="sb-textarea jx-text" aria-label="job posting" spellcheck="false"></textarea>
            <div class="jx-marks" id="marks-view" hidden>
              <div id="highlight" class="jx-text jx-highlight" tabindex="0" aria-label="Where each value was found">Run an extraction to highlight fields.</div>
            </div>
          </div>
          <ul class="jx-legend" aria-label="Highlight colours"><!--LEGEND--></ul>
        </div>
        <div class="jx-pane jx-pane--out">
          <div class="jx-pane__head">
            <span class="sb-label">Extracted JSON</span>
            <span id="valid-badge" class="sb-pill" hidden></span>
          </div>
          <pre id="json-out" class="sb-code jx-json" data-valid="" tabindex="0" aria-label="Extracted JSON">{}</pre>
          <p class="jx-hint">Hover a field to see where it came from in the posting.</p>
        </div>
      </div>
    </div>
    <noscript><p class="sb-callout sb-callout--warn">The demo needs JavaScript.</p></noscript>
  </div>
</section>
<script type="module" src="demo/demo.js"></script>"""

DEMO_PENDING = """<section id="demo" class="sb-section jx-demo-section" aria-labelledby="demo-title">
  <div class="sb-container">
    <p class="sb-kicker sb-reveal"><span class="sb-kicker__index">01</span><span class="sb-kicker__rule"></span>Live demo</p>
    <h2 id="demo-title" class="sb-h2 sb-reveal">Try it in your <span class="sb-serif">browser</span>.</h2>
    <div class="sb-card jx-demo jx-demo--pending sb-reveal">
      <span class="sb-pill sb-pill--pending">Demo pending</span>
      <p>The in-browser demo appears here once the fine-tuned model has been released (workflow
        <code>cpu_eval_release</code>).</p>
    </div>
  </div>
</section>"""


# ---------------------------------------------------------------------------------------------------------
# Numbers for the hero and the charts, read from results/results.json.


def _p(v: Any) -> str:
    """Same formatting as the results table: percentage with one decimal."""
    return f"{100 * float(v):.1f}"


def _measured(data: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [r for r in data.get("splits", {}).get(split, []) if r.get("status") == "measured"]


def headline_split(data: dict[str, Any]) -> str:
    """The split the page headlines: the real postings once measured, else the synthetic held-out split."""
    return "test_real" if _measured(data, "test_real") else "test_synth"


def _row(rows: list[dict[str, Any]], approach: str) -> dict[str, Any] | None:
    return next((r for r in rows if r.get("approach") == approach), None)


def _bench(data: dict[str, Any], quant: str) -> dict[str, Any] | None:
    return next((b for b in data.get("benchmarks", []) if quant in str(b.get("variant", ""))), None)


def short_name(name: str) -> str:
    """Compact chart label for an approach name from configs/approaches.yaml."""
    s = name
    if s.startswith("Regex baseline"):
        return "Regex rules"
    m = re.search(r"GGUF (\S+)", s)
    if "LoRA" in s and m:
        return f"LoRA 0.5B {m.group(1)}" + (" (this demo)" if "browser demo" in s else "")
    s = s.replace("GPT-OSS-120B via Groq (API, few-shot)", "GPT-OSS-120B few-shot")
    s = re.sub(r"Qwen2\.5-(\d\.\dB)-Instruct", r"\1", s)
    s = s.replace("+ guided JSON", "+ grammar")
    return s


def headline_stats(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(value, label, context) for the four hero tiles; 'pending' where a run has no committed numbers."""
    split = headline_split(data)
    rows = _measured(data, split)
    kind = "real" if split == "test_real" else "synthetic test"
    ours = _row(rows, "qwen2.5-0.5b-lora-cpu_f16")
    base = _row(rows, "qwen2.5-0.5b-cpu_zero_shot")
    api = _row(rows, "gpt-oss-120b-groq_few_shot")
    q4 = _bench(data, "Q4_K_M")
    out: list[tuple[str, str, str]] = []
    if ours:
        ctx = f"n = {ours['n']} {kind} posts"
        if base:
            ctx = f"base model {_p(base['metrics']['micro_f1'])}% &middot; " + ctx
        out.append((f"{_p(ours['metrics']['micro_f1'])}%", "Micro-F1 after fine-tuning", ctx))
        ctx = "fields filled in that should be null"
        if base:
            ctx = f"down from {_p(base['metrics']['hallucination_rate'])}% for the base model"
        out.append((f"{_p(ours['metrics']['hallucination_rate'])}%", "Hallucinated fields", ctx))
    else:
        out.append(("pending", "Micro-F1 after fine-tuning", "fine-tune not evaluated yet"))
        out.append(("pending", "Hallucinated fields", "fine-tune not evaluated yet"))
    if api:
        out.append(
            (
                f"{_p(api['metrics']['micro_f1'])}%",
                "GPT-OSS-120B few-shot",
                f"API model on an evenly spaced subset, n = {api['n']}",
            )
        )
    else:
        out.append(("pending", "GPT-OSS-120B few-shot", "API run pending"))
    if q4:
        out.append(
            (
                f"{float(q4['size_gb']):.2f} GB",
                "4-bit model in the browser",
                f"Q4_K_M &middot; {_p(q4['micro_f1'])}% micro-F1 &middot; {q4['tokens_per_s']} tokens/s on 4 CPU threads",
            )
        )
    else:
        out.append(("pending", "4-bit model in the browser", "benchmark pending"))
    return out


def _stats_html(stats: list[tuple[str, str, str]]) -> str:
    return "".join(
        f'<div class="sb-stat"><dt class="sb-stat__label">{label}</dt>'
        f'<dd class="sb-stat__value{" jx-pending-val" if value == "pending" else ""}">{value}</dd>'
        f'<dd class="sb-stat__ctx">{ctx}</dd></div>'
        for value, label, ctx in stats
    )


def _ci(r: dict[str, Any], key: str) -> tuple[float, float] | None:
    ci = (r.get("ci95") or {}).get(key)
    if not ci or ci[0] is None or ci[1] is None:
        return None
    return float(ci[0]) * 100, float(ci[1]) * 100


def _bar(v: float | None, ci: tuple[float, float] | None, label: str) -> str:
    if v is None:
        return f'<td data-label="{label}"><span class="jx-bar__val sb-faint">-</span></td>'
    style = f"--w:{v:.2f}" + (f";--lo:{ci[0]:.2f};--hi:{ci[1]:.2f}" if ci else "")
    whisker = '<span class="jx-bar__ci"></span>' if ci else ""
    title = f"{v:.1f}" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")
    return (
        f'<td data-label="{label}" title="{label}: {title}"><span class="jx-bar" style="{style}">'
        f'<span class="jx-bar__fill"></span>{whisker}<span class="jx-bar__val">{v:.1f}</span></span></td>'
    )


def approaches_chart(data: dict[str, Any]) -> str:
    split = headline_split(data)
    rows = sorted(_measured(data, split), key=lambda r: float(r["metrics"]["micro_f1"]))
    if not rows:
        return ""
    n_full = max(int(r.get("n") or 0) for r in rows)
    body = []
    for r in rows:
        m = r["metrics"]
        ours = "(ours)" in r["name"]
        n = int(r.get("n") or 0)
        tag = f' <span class="sb-tag jx-n">n = {n}</span>' if n != n_full else ""
        hall = m.get("hallucination_rate")
        body.append(
            f'<tr class="{"is-ours" if ours else ""}">'
            f'<th scope="row"><span class="jx-bars__name" title="{html.escape(r["name"])}">'
            f"{html.escape(short_name(r['name']))}</span>{tag}</th>"
            + _bar(float(m["micro_f1"]) * 100, _ci(r, "micro_f1"), "Micro-F1 %")
            + _bar(
                None if hall is None else float(hall) * 100, _ci(r, "hallucination_rate"), "Hallucinated %"
            )
            + "</tr>"
        )
    kind = "real test postings" if split == "test_real" else "synthetic held-out split"
    return f"""<figure class="sb-card sb-card--chart jx-chart jx-chart--wide sb-reveal">
  <figcaption>
    <p class="sb-card__title">Approaches compared</p>
    <p class="sb-card__sub">Micro-F1 (higher is better) and hallucinated fields (filled in when they should be null; lower
      is better) on the {kind}, n = {n_full} unless marked. Whiskers: 95% bootstrap CI.</p>
  </figcaption>
  <div class="jx-key" aria-hidden="true"><span><i class="jx-sw jx-sw--ours"></i>LoRA fine-tune (ours)</span>
    <span><i class="jx-sw"></i>Baselines</span></div>
  <table class="jx-bars">
    <thead><tr><th scope="col">Approach</th><th scope="col">Micro-F1 %</th><th scope="col">Hallucinated fields %</th></tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  <div class="sb-card__foot">Source: results/results.json ({split}) &middot;
    <a href="micro_f1_by_approach.png">matplotlib version</a></div>
</figure>"""


def benchmark_chart(data: dict[str, Any]) -> str:
    bench = [b for b in data.get("benchmarks", []) if b.get("size_gb")]
    if not bench:
        return ""
    smax = max(float(b["size_gb"]) for b in bench)
    tmax = max(float(b.get("tokens_per_s") or 0) for b in bench) or 1.0
    rows = []
    for b in bench:
        demo = "Q4_K_M" in b["variant"]
        name = html.escape(b["variant"].replace(" (llama.cpp)", ""))
        tag = ' <span class="sb-tag jx-demo-tag">in this demo</span>' if demo else ""
        size, tok = float(b["size_gb"]), float(b.get("tokens_per_s") or 0)
        f1 = "-" if b.get("micro_f1") is None else _p(b["micro_f1"])
        peak = "-" if b.get("peak_ram_gb") is None else f"{b['peak_ram_gb']}"
        cpu = re.sub(r"^CPU \((.*?), \d+ threads.*$", r"\1", str(b.get("hardware", "")))
        rows.append(
            f'<tr class="{"is-best" if demo else ""}"><th scope="row">{name}{tag}</th>'
            f'<td class="num"><span class="jx-mini"><span style="--w:{100 * size / smax:.1f}%"></span></span>{b["size_gb"]}</td>'
            f'<td class="num">{f1}</td>'
            f'<td class="num"><span class="jx-mini jx-mini--2"><span style="--w:{100 * tok / tmax:.1f}%"></span></span>{b.get("tokens_per_s", "-")}</td>'
            f'<td class="num">{peak}</td><td class="jx-cpu">{html.escape(cpu)}</td></tr>'
        )
    n = bench[0].get("n_examples")
    return f"""<figure class="sb-card sb-card--chart jx-chart sb-reveal">
  <figcaption>
    <p class="sb-card__title">Size, speed and accuracy after quantisation</p>
    <p class="sb-card__sub">The merged fine-tune as GGUF, run with llama.cpp on 4 CPU threads of a GitHub-hosted runner{f", {n} posts" if n else ""}.
      The 4-bit Q4_K_M file is the one this page downloads.</p>
  </figcaption>
  <div class="sb-table-wrap">
    <table class="sb-table jx-bench">
      <thead><tr><th scope="col">Variant</th><th scope="col" class="num">Size (GB)</th><th scope="col" class="num">Micro-F1 %</th>
        <th scope="col" class="num">Tokens/s</th><th scope="col" class="num">Peak memory (GB)</th><th scope="col">CPU</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
  <p class="jx-chart-note">Shared runners: tokens/s varies between runs and CPU models, so the CPU of each run is listed.</p>
  <div class="sb-card__foot">Source: results/benchmarks/*.json &middot; <a href="quant_tradeoff.png">matplotlib version</a></div>
</figure>"""


def caveats(data: dict[str, Any]) -> list[str]:
    split = headline_split(data)
    rows = _measured(data, split)
    notes = []
    if split == "test_synth":
        notes.append(
            "<strong>Synthetic test set.</strong> All scores come from <code>test_synth</code>: synthetic posts whose "
            "template families and companies never appear in training. The real-postings split (<code>test_real</code>) "
            "is pending, so these are not real-world accuracy figures."
        )
    else:
        notes.append(
            "<strong>Real postings.</strong> The headline split is <code>test_real</code>; the synthetic split is in the "
            "tables below."
        )
    notes.append(
        "<strong>The regex baseline is flattered by the synthetic data.</strong> Its patterns were written with every "
        "synthetic template in view and share alias tables and parsers with the generator, so read it as a sanity check "
        "of the evaluation stack rather than a competitor."
    )
    api = _row(rows, "gpt-oss-120b-groq_few_shot")
    if api:
        notes.append(
            f"<strong>The API model runs on a free tier</strong> over an evenly spaced subset (n = {api['n']}), so its "
            "confidence intervals are wider than those of rows scored on the full split."
        )
    notes += [
        "<strong>Shared runners.</strong> CPU benchmarks run on shared GitHub-hosted runners, so tokens/s varies between "
        "runs and CPU models.",
        "<strong>A 0.5B model.</strong> The CPU route trades accuracy for being trainable on CPU and small enough for a "
        "browser. The 1.5B/7B GPU route is provided but not run.",
        "<strong>Browser demo speed.</strong> GitHub Pages cannot send the COOP/COEP headers multi-threaded WASM needs, "
        "so the demo runs single-threaded.",
    ]
    return notes


def write_demo_assets(
    root: Path, out: Path, model_dir: Path, wllama_version: str = WLLAMA_VERSION
) -> dict[str, object]:
    """Copy the model's byte chunks to out/model/, write the manifest and the demo's data files.

    ``model_dir`` holds ``<name>.gguf.partNNN`` byte chunks of one GGUF file (``split -b``); the
    demo downloads them in order and joins them into a single file in the browser.
    """
    from jobx import prompts, vocab
    from jobx.baselines.guided import guided_schema

    parts = sorted(p for p in Path(model_dir).glob("*.gguf.part*"))
    if not parts:
        raise FileNotFoundError(f"no .gguf.partNNN chunks in {model_dir}")
    file_name = parts[0].name.split(".part")[0]
    mdir = out / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    for p in parts:
        if p.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError(f"{p.name} is >= 95 MB; split it (GitHub Pages / git file limit)")
        shutil.copy(p, mdir / p.name)
    manifest = {
        "file": file_name,
        "parts": [f"model/{p.name}" for p in parts],
        "size_mb": round(sum(p.stat().st_size for p in parts) / 1e6),
        "wllama": wllama_version,
    }
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ddir = out / "demo"
    ddir.mkdir(parents=True, exist_ok=True)
    js = (WEB / "demo.js").read_text()
    (ddir / "demo.js").write_text(js.replace("__WLLAMA_VERSION__", wllama_version))
    (ddir / "prompt.json").write_text(
        json.dumps(
            {"system": prompts.COMPACT_SYSTEM_PROMPT, "user_template": prompts.USER_TEMPLATE}, indent=2
        )
    )
    (ddir / "schema.json").write_text(json.dumps(guided_schema()))
    aliases = {
        "skills": vocab.SKILL_ALIASES,
        "locations": vocab.CITY_ALIASES,
        "work_mode": vocab.WORK_MODE_ALIASES,
        "employment_type": vocab.EMPLOYMENT_TYPE_ALIASES,
        "education": vocab.EDUCATION_ALIASES,
    }
    (ddir / "aliases.json").write_text(json.dumps(aliases))
    ex_path = root / "space" / "examples.json"
    examples = json.loads(ex_path.read_text()) if ex_path.exists() else []
    (ddir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    return manifest


def _fill(page: str, **parts: str) -> str:
    for k, v in parts.items():
        page = page.replace(f"<!--{k.upper()}-->", v)
    return page


def build_site(
    root: str | Path = ".",
    out: str | Path = "site",
    *,
    model_dir: str | Path | None = None,
    wllama_version: str = WLLAMA_VERSION,
) -> Path:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        shutil.copy(WEB / name, out / name)
    md = (root / "results/results_table.md").read_text()
    rj = root / "results" / "results.json"
    data: dict[str, Any] = json.loads(rj.read_text()) if rj.exists() else {}
    curves = ""
    png = root / "results/training/curves.png"
    if png.exists():
        shutil.copy(png, out / "curves.png")
        curves = """<figure class="sb-card jx-chart jx-curves sb-reveal">
  <figcaption>
    <p class="sb-card__title">Training curves</p>
    <p class="sb-card__sub">Train and validation loss, validation JSON micro-F1 and learning rate of the CPU LoRA run.</p>
  </figcaption>
  <div class="jx-png"><a href="curves.png"><img src="curves.png" alt="training loss, validation loss and JSON-F1 curves" loading="lazy"></a></div>
  <div class="sb-card__foot">Source: results/training/curves.png (matplotlib)</div>
</figure>"""
    for c in (
        sorted((root / "results" / "charts").glob("*.png")) if (root / "results" / "charts").exists() else []
    ):
        shutil.copy(c, out / c.name)
    charts = approaches_chart(data) + "\n" + benchmark_chart(data)
    demo = DEMO_PENDING
    has_demo = model_dir is not None
    if model_dir is not None:
        manifest = write_demo_assets(root, out, Path(model_dir), wllama_version)
        size_mb = int(str(manifest["size_mb"]))
        approx = round(size_mb, -1) if size_mb >= 50 else size_mb
        legend = "".join(
            f'<li><button type="button" class="jx-legend__item" data-f="{label}">'
            f'<i class="jx-sw f-{label}"></i>{text}</button></li>'
            for label, text in FIELD_LEGEND
        )
        demo = _fill(DEMO, approx_mb=str(approx), legend=legend)
    split = headline_split(data) if data else "test_synth"
    if split == "test_synth":
        n_synth = max((int(r.get("n") or 0) for r in _measured(data, "test_synth")), default=0)
        split_note = (
            f"Scores below are on <code>test_synth</code>: {f'{n_synth} ' if n_synth else ''}synthetic posts from "
            "template families and companies held out of training. The real-postings split is pending."
        )
    else:
        split_note = "Scores below are on <code>test_real</code>, real postings with human-verified labels."
    pills = (
        '<span class="sb-pill sb-pill--live">In-browser model live</span>'
        if has_demo
        else '<span class="sb-pill sb-pill--pending">Demo pending</span>'
    )
    pills += '<span class="sb-pill">Results committed by CI</span>'
    if split == "test_synth":
        pills += '<span class="sb-pill sb-pill--plain">Scores on a synthetic test split</span>'
    lead_tail = " The 4-bit model runs right here, in your browser." if has_demo else ""
    stats = headline_stats(data) if data else []
    if not has_demo:
        stats = [
            (v, "4-bit Q4_K_M model" if lbl == "4-bit model in the browser" else lbl, c)
            for v, lbl, c in stats
        ]
    stats_note = (
        "Synthetic held-out posts, not real-world accuracy: see the caveats below."
        if split == "test_synth"
        else ""
    )
    caveat_html = "\n".join(f"<li>{c}</li>" for c in caveats(data)) if data else ""
    page = _fill(
        PAGE,
        primary_href="#demo" if has_demo else "#results",
        primary_text="Try the demo" if has_demo else "See the results",
        lead_tail=lead_tail,
        pills=pills,
        stats=_stats_html(stats),
        stats_note=stats_note,
        demo=demo,
        split_note=split_note,
        charts=charts,
        body=md_to_html(md, styled=True, heading_shift=-1),
        curves=curves,
        caveats=caveat_html,
    )
    (out / "index.html").write_text(page)
    for name in ("results.json",):
        src = root / "results" / name
        if src.exists():
            shutil.copy(src, out / name)
    return out / "index.html"
