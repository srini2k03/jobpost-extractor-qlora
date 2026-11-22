"""Benchmark orchestration without Modal or a GPU: matrix, flags, estimate, budget, manifest, remote flow."""

from __future__ import annotations

import importlib
import json
import socket
import sys
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from jobx.bench import plan
from jobx.bench.fake_server import FakeConfig, running_fake_server
from jobx.bench.metrics import read_rows
from jobx.bench.remote import run_config, server_argv, snapshot_revision
from jobx.bench.server_metrics import MetricsScraper, parse_prometheus, summarize_samples

ROOT = Path(__file__).resolve().parents[1]
MATRIX = plan.load_matrix()


def cfg(name: str) -> plan.ServeConfig:
    return next(c for c in plan.expand(MATRIX) if c.name == name)


# ---------------------------------------------------------------------------------------------- matrix


def test_matrix_expands_with_defaults_and_groups() -> None:
    cfgs = plan.expand(MATRIX)
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names))
    assert {c.group for c in cfgs} == {
        "quantization",
        "max_num_seqs",
        "prefix_caching",
        "size",
        "lora",
        "baseline",
    }
    bf16 = cfg("q15-bf16")
    assert bf16.max_num_seqs == 64 and bf16.enable_prefix_caching and bf16.dtype == "bfloat16"
    assert bf16.workloads == ["extract-full", "synthetic-long"] and bf16.concurrency == [1, 4, 16, 64, 256]
    assert bf16.max_requests_per_level == MATRIX["estimate"]["max_requests_per_level"]
    assert bf16.quality_approach_ids() == [
        "vllm-qwen2.5-1.5b-bf16_few_shot",
        "vllm-qwen2.5-1.5b-bf16-guided_few_shot",
    ]
    lora = cfg("q05-lora")
    assert lora.request_model == "jobx-lora" and lora.quality_shots == 0 and lora.system == "compact"
    assert lora.quality_approach_ids() == ["vllm-qwen2.5-0.5b-lora_zero_shot"]
    assert cfg("q15-bf16-seqs16").rates == [] and cfg("q15-bf16-noapc").workloads == ["extract-full"]
    with pytest.raises(ValueError, match="unknown keys"):
        plan.ServeConfig.from_dict({**bf16.to_dict(), "typo": 1})
    with pytest.raises(ValueError, match="duplicate"):
        plan.expand({**MATRIX, "configs": [MATRIX["configs"][0], MATRIX["configs"][0]]})


def test_select_by_name_group_and_all() -> None:
    cfgs = plan.expand(MATRIX)
    assert len(plan.select(cfgs, "all")) == len(cfgs)
    assert [c.name for c in plan.select(cfgs, "q15-awq,q15-bf16")] == ["q15-bf16", "q15-awq"]  # matrix order
    assert {c.name for c in plan.select(cfgs, "group:prefix_caching")} == {"q15-bf16-noapc", "q15-awq-noapc"}
    for bad in ("nope", "group:nope", ""):
        with pytest.raises(ValueError):
            plan.select(cfgs, bad)


def test_overrides_and_compact_mapping() -> None:
    out = plan.apply_overrides(
        [cfg("q15-bf16"), cfg("q05-lora")],
        {"concurrency": [8], "rates": [], "duration_s": 5, "workloads": ["extract-full"]},
    )
    assert all(c.concurrency == [8] and c.rates == [] and c.duration_s == 5 for c in out)
    assert out[0].workloads == ["extract-full"] and out[1].workloads == ["extract-compact"]
    with pytest.raises(ValueError, match="unsupported"):
        plan.apply_overrides([cfg("q15-bf16")], {"model": "x"})


# ---------------------------------------------------------------------------------------------- flags


def test_vllm_flags_per_config() -> None:
    a = plan.vllm_args(cfg("q15-bf16"), port=8123)
    assert a[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    joined = " ".join(a)
    for frag in ("--port 8123", "--dtype bfloat16", "--max-num-seqs 64", "--gpu-memory-utilization 0.9",
                 "--max-model-len 3072", "--enable-prefix-caching", "--host 127.0.0.1"):  # fmt: skip
        assert frag in joined, frag
    assert "--quantization" not in a
    awq = " ".join(plan.vllm_args(cfg("q15-awq")))
    assert "--quantization awq_marlin" in awq and "--dtype float16" in awq
    assert "--quantization fp8" in " ".join(plan.vllm_args(cfg("q15-fp8")))
    assert "--quantization gptq_marlin" in " ".join(plan.vllm_args(cfg("q15-gptq")))
    noapc = plan.vllm_args(cfg("q15-bf16-noapc"))
    assert "--no-enable-prefix-caching" in noapc and "--max-model-len" in noapc
    assert noapc[noapc.index("--max-model-len") + 1] == "3072"  # extract-full only: 2048 + 384 -> 3072
    assert "--max-num-seqs 256" in " ".join(plan.vllm_args(cfg("q15-awq-seqs256")))
    lora = plan.vllm_args(cfg("q05-lora"), lora_path="/cache/adapters/jobx-lora/final")
    assert "--enable-lora" in lora and "jobx-lora=/cache/adapters/jobx-lora/final" in lora
    assert (
        lora[lora.index("--max-lora-rank") + 1] == "16" and lora[lora.index("--max-model-len") + 1] == "2048"
    )
    with pytest.raises(ValueError, match="adapter path"):
        plan.vllm_args(cfg("q05-lora"))
    pinned = server_argv(cfg("q15-bf16"), {"model_revision": "abc123"}, port=8000)
    assert pinned[pinned.index("--revision") + 1] == "abc123"


def test_llama_server_flags() -> None:
    c = cfg("q15-llamacpp-q4km")
    a = plan.llama_server_args(c, "/cache/hf/x.gguf", port=9000)
    joined = " ".join(a)
    assert a[0] == "/app/llama-server" and "-m /cache/hf/x.gguf" in joined
    assert (
        "-ngl 99" in joined and "--parallel 16" in joined and "-c 65536" in joined and "--metrics" in joined
    )
    assert server_argv(c, {"gguf_path": "/g.gguf"})[2] == "/g.gguf"


def test_snapshot_revision() -> None:
    assert snapshot_revision("/cache/hf/hub/models--Qwen--X/snapshots/0123abc/model.gguf") == "0123abc"
    assert snapshot_revision("/tmp/model") is None


# ---------------------------------------------------------------------------------------------- estimate


def test_estimate_and_budget() -> None:
    m = {
        **MATRIX,
        "estimate": {"startup_min": {"vllm": 4.0}, "level_overhead_s": 20, "quality_pass_min": 3.0},
    }
    e = plan.estimate(cfg("q15-bf16"), m)
    # 2 workloads x 8 levels x 80 s = 21.33 min + 4 startup + 2 quality passes x 3
    assert e["load_min"] == pytest.approx(21.3, abs=0.05) and e["quality_min"] == 6.0
    assert e["minutes"] == pytest.approx(31.3, abs=0.05)
    rate = plan.hourly_rate(MATRIX)
    assert rate["gpu"] == 0.80 and rate["total"] == pytest.approx(0.80 + 4 * 0.0472 + 16 * 0.008, abs=1e-3)
    assert e["usd_gpu"] == pytest.approx(31.33 / 60 * 0.80, abs=0.01)
    one = [cfg("q15-bf16")]
    assert plan.budget(one, m, cap_minutes=60).ok
    refused = plan.budget(one, m, cap_minutes=10)
    assert not refused.ok and "--yes" in refused.message
    assert plan.budget(one, m, cap_minutes=10, yes=True).ok
    assert "**total**" in plan.format_estimate(refused)
    assert plan.timeout_s(30) == (30 * 1.5 + 15) * 60


def test_full_matrix_fits_the_default_cap() -> None:
    d = plan.budget(plan.expand(MATRIX), MATRIX)
    assert d.ok and d.total_minutes <= MATRIX["max_gpu_minutes"] <= 7 * 60
    assert d.total_usd_gpu <= 6.0


def test_runtime_budget_guard_and_scheduler() -> None:
    g = plan.GpuBudget(100)
    assert g.try_start("a", 60) and not g.try_start("b", 50)
    g.finish("a", 30)
    assert g.try_start("b", 50)

    cfgs = [cfg("q15-bf16"), cfg("q15-fp8"), cfg("q15-awq")]
    ests = {"q15-bf16": 40.0, "q15-fp8": 40.0, "q15-awq": 40.0}
    in_flight: list[int] = []
    active = {"n": 0}
    pool = ThreadPoolExecutor(4)

    def work(c: plan.ServeConfig) -> dict[str, Any]:
        active["n"] += 1
        in_flight.append(active["n"])
        time.sleep(0.05)
        active["n"] -= 1
        return {"config": c.name, "status": "ok", "gpu_minutes": 45.0}  # each overran its estimate

    def submit(c: plan.ServeConfig) -> Future[dict[str, Any]]:
        return pool.submit(work, c)

    done, skipped = plan.run_scheduled(cfgs, ests, cap_minutes=100, max_parallel=2, submit=submit)
    # two start (80 reserved); the third waits; after 90 actual minutes it no longer fits (90 + 40 > 100)
    assert [m["config"] for m in done] and skipped == ["q15-awq"] and max(in_flight) <= 2

    def boom(c: plan.ServeConfig) -> dict[str, Any]:
        raise RuntimeError("container died")

    done, skipped = plan.run_scheduled(
        cfgs[:1], ests, cap_minutes=100, max_parallel=1, submit=lambda c: pool.submit(boom, c)
    )
    assert done[0]["status"] == "error" and "container died" in done[0]["error"] and not skipped


# ---------------------------------------------------------------------------------------------- request file


def test_request_file_parsing_and_selection() -> None:
    cfgs = plan.expand(MATRIX)
    r = plan.parse_request(
        {"configs": ["q15-bf16", "q15-awq"], "levels": {"duration_s": 30}, "yes": True}, cfgs
    )
    assert r.configs == ["q15-bf16", "q15-awq"] and r.overrides == {"duration_s": 30} and r.yes
    assert plan.parse_request({}, cfgs).configs == [c.name for c in cfgs]
    with pytest.raises(ValueError, match="unknown request keys"):
        plan.parse_request({"config": "all"}, cfgs)
    with pytest.raises(ValueError, match="unsupported level overrides"):
        plan.parse_request({"levels": {"foo": 1}}, cfgs)
    sel = plan.plan_selection(
        MATRIX, request={"configs": "q15-fp8", "levels": {"concurrency": [1], "rates": []}}
    )
    assert (
        [c.name for c in sel.configs] == ["q15-fp8"] and sel.configs[0].concurrency == [1] and sel.decision.ok
    )
    sel = plan.plan_selection(MATRIX, config="q15-gptq", request={"configs": "q15-fp8"})
    assert [c.name for c in sel.configs] == ["q15-gptq"]  # CLI wins
    big = plan.plan_selection(MATRIX, config="all", cap_minutes=30)
    assert not big.decision.ok
    shipped = json.loads((ROOT / ".github" / "bench-request.json").read_text())
    assert plan.parse_request(shipped, cfgs).configs


# ---------------------------------------------------------------------------------------------- manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    c = cfg("q15-awq")
    m = plan.build_manifest(
        c, argv=["vllm", "serve"], engine_version="0.33.0", gpu_name="NVIDIA L4", model_revision="abc",
        started_utc="2026-11-22T00:00:00Z", wall_s=600, usd_per_hour=1.2, files=["a.csv.gz"], estimate_minutes=30,
    )  # fmt: skip
    p = plan.write_manifest(tmp_path / "m" / "q15-awq.json", m)
    back = json.loads(p.read_text())
    assert back["gpu_minutes"] == 10.0 and back["estimated_cost_usd"] == pytest.approx(0.2)
    assert (
        back["settings"]["quantization"] == "awq_marlin"
        and back["status"] == "ok"
        and back["gpu"] == "NVIDIA L4"
    )
    assert plan.build_manifest(c, argv=[], engine_version=None, gpu_name=None, model_revision=None,
                               started_utc="", wall_s=1, usd_per_hour=0, files=[], error="x")["status"] == "error"  # fmt: skip


# ---------------------------------------------------------------------------------------------- /metrics


def test_prometheus_parsing_and_summary() -> None:
    text = (
        "# HELP vllm:kv_cache_usage_perc x\n# TYPE vllm:kv_cache_usage_perc gauge\n"
        'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.25\n'
        'vllm:prefix_cache_queries_total{model_name="m"} 1000\nvllm:prefix_cache_hits_total{model_name="m"} 100\n'
        "vllm:num_preemptions_total 2\nbogus line\nvllm:x NaN\n"
    )
    a = parse_prometheus(text)
    assert a["vllm:kv_cache_usage_perc"] == 0.25 and "vllm:x" not in a
    b = {**a, "vllm:kv_cache_usage_perc": 0.75, "vllm:prefix_cache_queries_total": 3000.0,
         "vllm:prefix_cache_hits_total": 1700.0, "vllm:num_preemptions_total": 5.0}  # fmt: skip
    s = summarize_samples([a, b])
    assert s["kv_cache_usage_mean"] == 0.5 and s["kv_cache_usage_max"] == 0.75
    assert s["prefix_cache_hit_rate"] == 0.8 and s["num_preemptions"] == 3
    legacy = summarize_samples([{"vllm:gpu_cache_usage_perc": 0.1}])
    assert legacy["kv_cache_usage_max"] == 0.1 and legacy["prefix_cache_hit_rate"] is None


def test_scraper_against_fake_server() -> None:
    import httpx

    with running_fake_server(
        FakeConfig(ttft_s=0.0, itl_s=0.0, output_tokens=2, prefix_cache_factor=0.5)
    ) as server:
        url = f"http://127.0.0.1:{server.port}"
        with MetricsScraper(f"{url}/metrics", interval_s=0.05) as sc:
            for _ in range(3):
                httpx.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": "s" * 400},
                            {"role": "user", "content": "u"},
                        ]
                    },
                    trust_env=False,
                )
        st = sc.stats()
    assert st["n_scrapes"] >= 2 and st["kv_cache_usage_max"] is not None
    assert st["prefix_cache_hit_rate"] is not None and 0 < st["prefix_cache_hit_rate"] < 1


# ---------------------------------------------------------------------------------------------- remote flow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_run_config_end_to_end_with_fake_server(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "processed").symlink_to(ROOT / "data" / "processed")
    port = _free_port()
    c = plan.apply_overrides(
        [cfg("q15-bf16")],
        {"workloads": ["extract-full"], "concurrency": [2], "rates": [20], "duration_s": 0.4, "warmup": 1},
    )[0]
    cmd = [sys.executable, "-m", "jobx.bench.fake_server", "--port", str(port), "--ttft", "0.01", "--itl", "0.002",
           "--output-tokens", "5", "--prefix-cache-factor", "0.5"]  # fmt: skip
    out = run_config(
        c.to_dict(), out_root=tmp_path / "out", usd_per_hour=1.2, estimate_minutes=30, commit="abc1234",
        server_cmd=cmd, download=False, repo_root=repo, quality_max_examples=4, port=port, health_timeout_s=30,
    )  # fmt: skip
    m = out["manifest"]
    assert m["status"] == "ok", m.get("traceback")
    assert (
        m["config"] == "q15-bf16" and m["argv"] == cmd and m["startup_s"] >= 0 and m["estimate_minutes"] == 30
    )
    files = set(out["files"])
    assert "results/bench/raw/q15-bf16/extract-full.csv.gz" in files
    assert "results/bench/manifests/q15-bf16.json" in files
    for approach in c.quality_approach_ids():
        assert f"results/predictions/{approach}__test_synth.jsonl" in files
        meta = json.loads(out["files"][f"results/predictions/{approach}__test_synth.meta.json"])
        assert meta["system_prompt"] == "full" and meta["n_shots"] == 3 and meta["commit"] == "abc1234"
        assert "(Modal)" in meta["hardware"] and meta["n_examples"] == 4
    guided = json.loads(
        out["files"]["results/predictions/vllm-qwen2.5-1.5b-bf16-guided_few_shot__test_synth.meta.json"]
    )
    assert guided["response_format"] == "json_schema"
    rows = read_rows([tmp_path / "out" / "results/bench/raw/q15-bf16/extract-full.csv.gz"])
    assert {r["mode"] for r in rows} == {"closed", "poisson"} and all(r["status"] == "200" for r in rows)
    assert all(r["prefix_cache_hit_rate"] != "" for r in rows)  # scraped from the fake /metrics
    assert not (
        ROOT / "results" / "predictions" / "vllm-qwen2.5-1.5b-bf16_few_shot__test_synth.jsonl"
    ).exists()


def test_run_config_reports_server_failure(tmp_path: Path) -> None:
    out = run_config(
        cfg("q15-fp8").to_dict(), out_root=tmp_path, server_cmd=[sys.executable, "-c", "print('boom'); raise SystemExit(3)"],
        download=False, port=_free_port(), health_timeout_s=20,
    )  # fmt: skip
    m = out["manifest"]
    assert m["status"] == "error" and "exited with code 3" in m["error"] and "boom" in m["server_log_tail"]


# ---------------------------------------------------------------------------------------------- modal app (stubbed)


class _Chain:
    """Stands in for modal.Image: every builder method returns self and is recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> _Chain:
            self.calls.append((name, args))
            return self

        return method


def _stub_modal() -> types.ModuleType:
    mod = types.ModuleType("modal")
    images: list[_Chain] = []

    class Image:
        @staticmethod
        def from_registry(tag: str, **kw: Any) -> _Chain:
            ch = _Chain()
            ch.calls.append(("from_registry", (tag, kw)))
            images.append(ch)
            return ch

    class Volume:
        @staticmethod
        def from_name(name: str, **kw: Any) -> Any:
            return types.SimpleNamespace(name=name, commit=lambda: None)

    class App:
        def __init__(self, name: str) -> None:
            self.name = name
            self.functions: dict[str, dict[str, Any]] = {}

        def function(self, **kw: Any) -> Any:
            def deco(fn: Any) -> Any:
                self.functions[fn.__name__] = kw
                return fn

            return deco

        def local_entrypoint(self) -> Any:
            return lambda fn: fn

    mod.Image = Image  # type: ignore[attr-defined]
    mod.Volume = Volume  # type: ignore[attr-defined]
    mod.App = App  # type: ignore[attr-defined]
    mod.images = images  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def modal_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    stub = _stub_modal()
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.delitem(sys.modules, "jobx.bench.modal_app", raising=False)
    mod = importlib.import_module("jobx.bench.modal_app")
    yield mod
    sys.modules.pop("jobx.bench.modal_app", None)


def test_modal_app_definitions(modal_app: Any) -> None:
    fns = modal_app.app.functions
    assert set(fns) == {"bench_vllm", "bench_llama_cpp"}
    assert all(kw["gpu"] == "L4" and set(kw["volumes"]) == {"/cache", "/results"} for kw in fns.values())
    calls = [c for img in sys.modules["modal"].images for c in img.calls]  # type: ignore[attr-defined]
    pip = [a for name, a in calls if name == "uv_pip_install"]
    assert any(f"vllm=={MATRIX['vllm_version']}" in a for a in pip)
    tags = [a[0] for name, a in calls if name == "from_registry"]
    assert MATRIX["llama_cpp_image"] in tags and any(t.startswith("nvidia/cuda:") for t in tags)


def test_modal_entrypoint_estimate_refusal_and_run(
    modal_app: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modal_app.main(config="q15-bf16", dry_run=True)
    assert "q15-bf16" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        modal_app.main(matrix="all", max_gpu_minutes=10)
    assert e.value.code == 2

    calls: list[tuple[int | None, dict[str, Any]]] = []

    class FakeFn:
        def __init__(self) -> None:
            self.timeout: int | None = None

        def with_options(self, timeout: int | None = None) -> FakeFn:
            self.timeout = timeout
            return self

        def remote(self, cfg_dict: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
            calls.append((self.timeout, opts))
            name = cfg_dict["name"]
            man = {"config": name, "status": "ok", "gpu_minutes": 1.0, "estimated_cost_usd": 0.02}
            return {
                "manifest": man,
                "files": {f"results/bench/manifests/{name}.json": json.dumps(man).encode()},
            }

    modal_app.bench_vllm = FakeFn()
    modal_app.bench_llama_cpp = FakeFn()
    modal_app.main(config="q15-bf16,q15-llamacpp-q4km", out=str(tmp_path), skip_quality=True)
    assert (tmp_path / "results/bench/manifests/q15-bf16.json").exists()
    assert (tmp_path / "results/bench/manifests/q15-llamacpp-q4km.json").exists()
    assert len(calls) == 2 and all(t and t > 15 * 60 and o["skip_quality"] for t, o in calls)
