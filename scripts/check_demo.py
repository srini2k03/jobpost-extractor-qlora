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
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    from jobx.schema import JobPosting

    httpd = serve(a.site, a.port)
    ms = int(a.timeout_min * 60 * 1000)
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda m: print(f"[console] {m.text}"))
        page.goto(f"http://127.0.0.1:{a.port}/index.html")
        page.wait_for_selector("#load")
        page.click("#load")
        page.wait_for_selector("#status[data-state=ready]", timeout=ms)
        print(f"model loaded after {time.time() - t0:.0f}s")
        page.click("button[data-example='0']")
        page.click("#extract")
        page.wait_for_selector("#status[data-state=done]", timeout=ms)
        out = page.inner_text("#json-out")
        valid = page.get_attribute("#json-out", "data-valid")
        status = page.inner_text("#status")
        browser.close()
    httpd.shutdown()
    print(status)
    print(out)
    obj = json.loads(out)
    JobPosting.model_validate(obj)
    if valid != "true":
        print("page reported invalid output", file=sys.stderr)
        return 1
    print(f"OK: valid JobPosting JSON in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
