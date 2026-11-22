"""A tiny OpenAI-compatible streaming server with configurable timing, for tests and CI smoke runs.

Pure ``asyncio`` (no web framework), HTTP/1.1 with keep-alive and chunked SSE responses. It implements just
enough of ``/v1/chat/completions`` for the load generator: ``stream`` (with ``stream_options.include_usage``)
and non-streaming replies, ``max_tokens``, vLLM's ``ignore_eos``, plus ``GET /v1/models``, ``GET /health`` and vLLM-style ``GET /metrics``.

Timing model per request: optional queueing behind ``max_num_seqs`` concurrent sequences, then
``ttft_s`` (multiplied by ``prefix_cache_factor`` when the request's prefix, every message but the last, was
seen before), then one token every ``itl_s``. The output length is ``output_tokens`` (capped by
``max_tokens``), or exactly ``max_tokens`` with ``ignore_eos``. ``fail_every=N`` answers every N-th request
with HTTP 500, for testing error accounting; ``send_usage=False`` drops the final usage chunk.

    python -m jobx.bench.fake_server --port 8000 --ttft 0.05 --itl 0.01
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeConfig:
    ttft_s: float = 0.05
    itl_s: float = 0.01
    output_tokens: int = 32
    prefix_cache_factor: float = 1.0  # e.g. 0.3: a cached prefix cuts TTFT to 30 %
    max_num_seqs: int | None = None
    fail_every: int = 0
    model: str = "fake-model"
    send_usage: bool = True  # False: ignore stream_options.include_usage, like some older servers


class FakeServer:
    def __init__(self, config: FakeConfig | None = None) -> None:
        self.config = config or FakeConfig()
        self.requests_seen = 0
        self.prefix_hits = 0
        self.running = 0
        self.waiting = 0
        self.prompt_tokens_total = 0
        self.cached_tokens_total = 0
        self._prefixes: set[str] = set()
        self._sem = asyncio.Semaphore(self.config.max_num_seqs) if self.config.max_num_seqs else None
        self._server: asyncio.Server | None = None
        self.port = 0

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._server = await asyncio.start_server(self._handle, host, port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            with suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                method, path, _ = line.decode("latin-1").split(" ", 2)
                headers: dict[str, str] = {}
                while (h := await reader.readline()) not in (b"\r\n", b"\n", b""):
                    k, _, v = h.decode("latin-1").partition(":")
                    headers[k.strip().lower()] = v.strip()
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                await self._route(method, path, body, writer)
                if headers.get("connection", "").lower() == "close":
                    break
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            with suppress(Exception):
                writer.close()

    async def _route(self, method: str, path: str, body: bytes, writer: asyncio.StreamWriter) -> None:
        if method == "GET" and path in ("/health", "/v1/health"):
            await _send_json(writer, 200, {"status": "ok"})
        elif method == "GET" and path == "/metrics":
            await _send_text(writer, 200, self.metrics_text())
        elif method == "GET" and path == "/v1/models":
            await _send_json(
                writer, 200, {"object": "list", "data": [{"id": self.config.model, "object": "model"}]}
            )
        elif method == "POST" and path == "/v1/chat/completions":
            await self._chat(json.loads(body or b"{}"), writer)
        else:
            await _send_json(writer, 404, {"error": {"message": f"no route {method} {path}"}})

    async def _chat(self, req: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        cfg = self.config
        self.requests_seen += 1
        if cfg.fail_every and self.requests_seen % cfg.fail_every == 0:
            await _send_json(writer, 500, {"error": {"message": "injected failure"}})
            return
        msgs = req.get("messages") or []
        prefix = hashlib.sha256(json.dumps(msgs[:-1], sort_keys=True).encode()).hexdigest()
        hit = prefix in self._prefixes and len(msgs) > 1
        self._prefixes.add(prefix)
        self.prefix_hits += hit
        max_tokens = int(req.get("max_tokens") or cfg.output_tokens)
        n_out = max_tokens if req.get("ignore_eos") else min(cfg.output_tokens, max_tokens)
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in msgs) // 4
        self.prompt_tokens_total += prompt_tokens
        if hit:
            self.cached_tokens_total += sum(len(str(m.get("content", ""))) for m in msgs[:-1]) // 4
        ttft = cfg.ttft_s * (cfg.prefix_cache_factor if hit else 1.0)
        rid = f"chatcmpl-fake-{self.requests_seen}"
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n_out,
            "total_tokens": prompt_tokens + n_out,
        }

        async def generate() -> None:
            if not req.get("stream"):
                await asyncio.sleep(ttft + cfg.itl_s * max(n_out - 1, 0))
                text = "".join(_token(i) for i in range(n_out))
                choice = {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
                await _send_json(
                    writer, 200, {"id": rid, "object": "chat.completion", "choices": [choice], "usage": usage}
                )
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n"
                b"cache-control: no-cache\r\nconnection: keep-alive\r\n\r\n"
            )
            await writer.drain()
            start = time.perf_counter()
            await _chunk(writer, _sse(rid, {"role": "assistant", "content": ""}))
            for i in range(n_out):
                # Sleep until the token's scheduled time (absolute schedule: no drift from sleep overshoot).
                due = start + ttft + i * cfg.itl_s
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
                finish = "length" if i == n_out - 1 else None
                await _chunk(writer, _sse(rid, {"content": _token(i)}, finish))
            opts = req.get("stream_options") or {}
            if opts.get("include_usage") and cfg.send_usage:
                payload = {"id": rid, "object": "chat.completion.chunk", "choices": [], "usage": usage}
                await _chunk(writer, f"data: {json.dumps(payload)}\n\n".encode())
            await _chunk(writer, b"data: [DONE]\n\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()

        async def tracked() -> None:
            self.running += 1
            try:
                await generate()
            finally:
                self.running -= 1

        if self._sem is None:
            await tracked()
        else:
            self.waiting += 1
            async with self._sem:
                self.waiting -= 1
                await tracked()

    def metrics_text(self) -> str:
        """vLLM-style Prometheus metrics (names as in vLLM's V1 engine)."""
        cap = self.config.max_num_seqs or 256
        lines = [
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f'vllm:kv_cache_usage_perc{{model_name="{self.config.model}"}} {min(1.0, self.running / cap)}',
            f'vllm:num_requests_running{{model_name="{self.config.model}"}} {self.running}',
            f'vllm:num_requests_waiting{{model_name="{self.config.model}"}} {self.waiting}',
            f'vllm:prefix_cache_queries_total{{model_name="{self.config.model}"}} {self.prompt_tokens_total}',
            f'vllm:prefix_cache_hits_total{{model_name="{self.config.model}"}} {self.cached_tokens_total}',
            f'vllm:num_preemptions_total{{model_name="{self.config.model}"}} 0',
        ]
        return "\n".join(lines) + "\n"


def _token(i: int) -> str:
    return f"t{i} "


def _sse(rid: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": rid,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _chunk(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
    await writer.drain()


async def _send_text(writer: asyncio.StreamWriter, status: int, text: str) -> None:
    body = text.encode()
    writer.write(
        f"HTTP/1.1 {status} OK\r\ncontent-type: text/plain; version=0.0.4\r\ncontent-length: {len(body)}\r\n"
        f"connection: keep-alive\r\n\r\n".encode()
        + body
    )
    await writer.drain()


async def _send_json(writer: asyncio.StreamWriter, status: int, obj: dict[str, Any]) -> None:
    body = json.dumps(obj).encode()
    reason = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}.get(status, "Error")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {len(body)}\r\n"
        f"connection: keep-alive\r\n\r\n".encode()
        + body
    )
    await writer.drain()


@contextmanager
def running_fake_server(config: FakeConfig | None = None) -> Iterator[FakeServer]:
    """Run a :class:`FakeServer` on its own event loop in a daemon thread (for synchronous callers)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: dict[str, FakeServer] = {}

    def run() -> None:
        asyncio.set_event_loop(loop)
        server = FakeServer(config)
        holder["server"] = server
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=run, name="fake-server", daemon=True)
    t.start()
    if not ready.wait(10):
        raise RuntimeError("fake server did not start")
    server = holder["server"]
    try:
        yield server
    finally:
        asyncio.run_coroutine_threadsafe(server.close(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()


async def serve(config: FakeConfig, host: str, port: int) -> None:
    server = FakeServer(config)
    await server.start(host, port)
    print(f"fake OpenAI-compatible server on http://{host}:{server.port}/v1 ({config})", flush=True)
    await asyncio.Event().wait()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--itl", type=float, default=0.01)
    ap.add_argument("--output-tokens", type=int, default=32)
    ap.add_argument("--prefix-cache-factor", type=float, default=1.0)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--fail-every", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = FakeConfig(a.ttft, a.itl, a.output_tokens, a.prefix_cache_factor, a.max_num_seqs, a.fail_every)
    asyncio.run(serve(cfg, a.host, a.port))


if __name__ == "__main__":
    main()
