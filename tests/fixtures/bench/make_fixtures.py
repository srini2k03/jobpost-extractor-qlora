"""Regenerate the serving-benchmark report fixtures from the fake server (no GPU, ~20 s).

    python tests/fixtures/bench/make_fixtures.py

The numbers come from jobx.bench.fake_server with made-up delays: they exercise the report, they are not
measurements of anything. metrics/*.json are hand-written fixture scores for the same reason.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jobx.bench import plan
from jobx.bench.fake_server import FakeConfig, running_fake_server
from jobx.bench.loadgen import RunResult, StreamingClient, sweep, write_csv
from jobx.bench.server_metrics import MetricsScraper
from jobx.bench.workload import build_workload

HERE = Path(__file__).resolve().parent
# config -> fake timing (the "AWQ" rows decode faster; "noapc" rows get no prefix-cache speed-up)
FAKE = {
    "q15-bf16": FakeConfig(
        ttft_s=0.05, itl_s=0.004, output_tokens=20, prefix_cache_factor=0.3, max_num_seqs=8
    ),
    "q15-awq": FakeConfig(
        ttft_s=0.04, itl_s=0.0025, output_tokens=20, prefix_cache_factor=0.3, max_num_seqs=8
    ),
    "q15-bf16-noapc": FakeConfig(ttft_s=0.05, itl_s=0.004, output_tokens=20, max_num_seqs=8),
    "q15-awq-noapc": FakeConfig(ttft_s=0.04, itl_s=0.0025, output_tokens=20, max_num_seqs=8),
}
LEVELS = {
    "concurrency": [1, 4, 16],
    "rates": [],
    "duration_s": 0.6,
    "warmup": 1,
    "workloads": ["extract-full"],
}
PER_LEVEL = 40


def measure(name: str, fake: FakeConfig, cfg: plan.ServeConfig) -> list[RunResult]:
    with running_fake_server(fake) as server:
        reqs = build_workload("extract-full", cfg.n_levels * (cfg.warmup + PER_LEVEL), seed=0)
        metrics_url = server.base_url.removesuffix("/v1") + "/metrics"

        async def go() -> list[RunResult]:
            async with StreamingClient(server.base_url, cfg.request_model) as client:
                return await sweep(
                    client, reqs, label=name, workload="extract-full", concurrencies=cfg.concurrency,
                    per_level=PER_LEVEL, warmup=cfg.warmup, duration_s=cfg.duration_s,
                    server_monitor_factory=lambda: MetricsScraper(metrics_url, interval_s=0.1),
                )  # fmt: skip

        return asyncio.run(go())


def main() -> None:
    matrix = plan.load_matrix()
    cfgs = {c.name: c for c in plan.expand(matrix)}
    for name, fake in FAKE.items():
        cfg = plan.apply_overrides([cfgs[name]], LEVELS)[0]
        runs = measure(name, fake, cfg)
        rel = Path("raw") / name / "extract-full.csv.gz"
        write_csv(HERE / rel, runs)
        man = plan.build_manifest(
            cfg, argv=["fake_server"], engine_version="fake", gpu_name="none (fake server fixture)",
            model_revision=None, started_utc="2026-01-01T00:00:00Z", wall_s=10.0,
            usd_per_hour=plan.hourly_rate(matrix)["total"], files=[str(rel)],
        )  # fmt: skip
        plan.write_manifest(HERE / "manifests" / f"{name}.json", man)
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
