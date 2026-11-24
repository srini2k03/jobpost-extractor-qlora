"""Playwright check: the built site loads the model in headless Chromium and extracts valid JSON.

    python scripts/check_demo.py site [--timeout-min 20]

Serves the directory on localhost, clicks "Load model", waits for the download + load, runs the first
example and validates the output with the JobPosting schema. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path


def serve(directory: str, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout-min", type=float, default=20)
    ap.add_argument("--chrome", default=None, help="use this Chrome/Chromium binary instead of Playwright's")
    ap.add_argument("--out-json", default=None, help="write the measured load/extraction timings here")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    from jobx.schema import JobPosting

    httpd = serve(a.site, a.port)
    ms = int(a.timeout_min * 60 * 1000)
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda m: print(f"[console] {m.text}"))
        page.goto(f"http://127.0.0.1:{a.port}/index.html")
        page.wait_for_selector("#load")
        page.click("#load")
        page.wait_for_selector("#status[data-state=ready]", timeout=ms)
        load_s = time.time() - t0
        print(f"model loaded after {load_s:.0f}s")
        page.click("button[data-example='0']")
        t1 = time.time()
        page.click("#extract")
        page.wait_for_selector("#status[data-state=done]", timeout=ms)
        extract_s = time.time() - t1
        out = page.inner_text("#json-out")
        valid = page.get_attribute("#json-out", "data-valid")
        status = page.inner_text("#status")
        fc_rows = page.eval_on_selector_all("#field-check .jx-fc", "els => els.map(e => e.dataset.status)")
        metrics = page.eval_on_selector_all(
            "#run-metrics div",
            "els => Object.fromEntries(els.map(e => [e.querySelector('dt').textContent, e.querySelector('dd').textContent]))",
        )
        browser.close()
    httpd.shutdown()
    print(status)
    print(out)
    print(f"field check: {fc_rows}")
    print(f"run metrics: {metrics}")
    if len(fc_rows) != 11 or not metrics.get("Output tokens"):
        print("FAIL: the field check or the run metrics did not render", file=sys.stderr)
        return 1
    if a.out_json:
        import datetime

        Path(a.out_json).write_text(
            json.dumps(
                {
                    "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
                    "browser": "headless Chromium (Playwright), GitHub Actions ubuntu-24.04",
                    "load_s": round(load_s, 1),
                    "extract_s": round(extract_s, 1),
                    "metrics": metrics,
                    "valid": valid == "true",
                    "field_check": fc_rows,
                },
                indent=2,
            )
            + "\n"
        )
    if valid != "true":
        print("FAIL: the page reported invalid output (see above)", file=sys.stderr)
        return 1
    try:
        JobPosting.model_validate(json.loads(out))
    except Exception as e:
        print(f"FAIL: output does not validate as JobPosting: {e}", file=sys.stderr)
        return 1
    print(f"OK: valid JobPosting JSON in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
