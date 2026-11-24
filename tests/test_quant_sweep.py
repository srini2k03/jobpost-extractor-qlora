import json
import os
from types import SimpleNamespace

from conftest import ROOT
from jobx import prompts
from jobx.data.build import read_jsonl


def test_sweep_runs_every_variant_and_thread_setting(monkeypatch, tmp_path):
    import httpx

    from jobx.quantize import sweep

    gold = {
        e["id"]: e
        for f in ("data/processed/test_synth.jsonl", "data/real/test_real.jsonl")
        for e in read_jsonl(ROOT / f)
    }
    by_text = {e["text"].strip(): e for e in gold.values()}
    launched = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            text = json["messages"][-1]["content"].split("<posting>\n", 1)[1].rsplit("\n</posting>", 1)[0]
            body = {
                "choices": [{"message": {"content": prompts.target_json(by_text[text]["label"])}}],
                "timings": {
                    "predicted_per_second": 50.0,
                    "prompt_per_second": 400.0,
                    "prompt_n": 120,
                    "predicted_n": 90,
                },
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(sweep, "wait_ready", lambda url, proc: None)
    monkeypatch.setattr(sweep, "drop_page_cache", lambda: False)
    monkeypatch.setattr(sweep, "git_commit", lambda root=".": "test")

    def popen(cmd, **k):
        launched.append(cmd)
        return SimpleNamespace(
            pid=os.getpid(), terminate=lambda: None, wait=lambda timeout: 0, poll=lambda: None
        )

    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    g = tmp_path / "gguf"
    g.mkdir()
    for q in ("F16", "Q4_K_M"):
        (g / f"m-{q}.gguf").write_bytes(b"0" * 2000)
    (tmp_path / "data").symlink_to(ROOT / "data")
    rc = sweep.main(
        [
            "--server-bin",
            "srv",
            "--gguf-dir",
            str(g),
            "--name",
            "m",
            "--quants",
            "F16",
            "Q4_K_M",
            "--synth-n",
            "6",
            "--thread-sweep",
            "1",
            "2",
            "--thread-sweep-n",
            "3",
            "--root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "results/quant/sweep.json").read_text())
    assert [v["quant"] for v in out["variants"]] == ["F16", "Q4_K_M"]
    v = out["variants"][0]
    assert v["accuracy"]["test_synth"]["micro_f1"] == 1.0 and v["accuracy"]["test_synth"]["n"] == 6
    assert v["gen_tokens_per_s"] == 50.0 and v["warm_latency_s"]["p50"] is not None
    assert [t["threads"] for t in out["thread_sweep"]] == [1, 2]
    assert launched[-2][launched[-2].index("-t") : launched[-2].index("-t") + 2] == ["-t", "1"]
    assert (tmp_path / "results/quant/q4_k_m__test_real.jsonl").exists()
