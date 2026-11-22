# Serving benchmark (vLLM on a Modal L4)

Question: what does it cost, and what latency do users see, when the extractor is served by vLLM on one
L4, and how much do quantization, scheduler width and prefix caching change that? Phase 1 (this note) is
the load-test harness and workload, tested on CPU against a fake server. Phase 2 adds the Modal runner and
the real runs; nothing here starts a GPU or makes a paid call.

## What is measured

`jobx bench run` sends streaming `/v1/chat/completions` requests (`stream=true`,
`stream_options={"include_usage": true}`, greedy) and timestamps every SSE line with `time.perf_counter`
as it arrives. One CSV row per request (`results/bench/raw/<name>.csv`):

| field | definition |
|---|---|
| TTFT | request start to the first chunk carrying content |
| inter-token gaps | between consecutive content chunks; per request the mean, p50, p95 and max are stored (not the list) |
| E2E | request start to end of stream |
| tokens | prompt and completion tokens from the final usage chunk; if a server omits it, completion tokens = content chunks |
| status, error | HTTP status (0 = no response), error text; failed requests stay in the CSV |
| GPU | peak memory and mean utilisation from a background sampler (NVML, else `nvidia-smi`, else nothing) |

`jobx bench summarize` turns the raw rows into one row per run (`results/bench/summary.csv`):
p50/p95/p99 of TTFT, ITL (each request's mean gap), TPOT ((E2E - TTFT) / (output tokens - 1)) and E2E;
`itl_worst_p95_ms` (p95 over requests of each request's own p95 gap, which shows decode stalls when long
prefills are scheduled); requests/s, output tokens/s, total tokens/s over the measurement window; error
rate; goodput (requests/s that met TTFT <= 1 s and E2E <= 5 s by default, `--sla-ttft/--sla-e2e`); cost:

    usd_per_1k_requests      = gpu_usd_per_hour / (requests_per_s * 3600) * 1000
    usd_per_1m_output_tokens = gpu_usd_per_hour / (output_tokens_per_s * 3600) * 1e6

These assume the GPU is kept busy at the measured rate: no idle time, cold starts or billing minimums.

## Load patterns

* Closed loop (`--concurrency 1,4,16,64`): N workers, each sends its next request when the previous one
  finishes, for `--num-requests` per level or `--duration` seconds. This finds the throughput/latency curve.
* Open loop (`--rate 2,8,32`): Poisson arrivals at a fixed rate, regardless of completions, which is closer
  to real traffic and shows queueing once the rate passes capacity. `--max-concurrency` optionally caps
  requests in flight; client-side waiting then counts towards TTFT and E2E. For short runs the measured
  requests/s sits below the offered rate because the window includes draining the last requests.
* Each level starts with `--warmup` requests that are excluded. Levels use disjoint postings while the
  pool lasts (test_synth + val, 823 postings), so prefix caching cannot reuse a whole earlier prompt.

## Workloads

| workload | prompt | output |
|---|---|---|
| `extract-full` | `SYSTEM_PROMPT` + the 3 few-shot exemplars (`select_few_shot`) + posting: median 4,071 characters, of which 3,711 (91 %) are a prefix shared by every request | model decides, `max_tokens` 384 |
| `extract-compact` | `COMPACT_SYSTEM_PROMPT` + posting (what the fine-tune was trained on) | as above |
| `synthetic-long[-compact]` | as above | seeded lognormal length (median 256, capped at 1024) forced with `max_tokens` + vLLM `ignore_eos`/`min_tokens`, to stress decode |

`jobx bench workload --workload extract-full --tokenizer Qwen/Qwen2.5-1.5B-Instruct` prints the
prompt-length distribution (characters always, tokens when `transformers` is installed).

## Planned matrix (`configs/bench/matrix.yaml`)

All on one L4 (24 GB), vLLM, `max_model_len` 4096:

* Qwen2.5-1.5B-Instruct: BF16, FP8 (online, W8A8 on Ada), AWQ int4 (`Qwen/Qwen2.5-1.5B-Instruct-AWQ`),
  GPTQ int4 (`Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`);
* `max_num_seqs` 16 / 64 / 256 and automatic prefix caching on/off, for BF16 and AWQ;
* a 3B check (Qwen2.5-3B-Instruct BF16 and AWQ);
* the repo's 0.5B LoRA fine-tune served with `--enable-lora` (adapter from release model-v1), compact prompt;
* baseline: llama.cpp `llama-server` with the Q4_K_M GGUF on the same GPU.

Quality per quantization: one greedy pass over test_synth per config, scored by the repo's own evaluator
(`jobx eval`), so a speed-up is never reported without its accuracy cost.

GPU prices live in the same file (`usd_per_hour`, `source`, `checked`); L4 on Modal is ~$0.80/h, checked
2026-11 and marked to-verify. `summarize --gpu L4` reads it; `--gpu-price` overrides it.

## Trying it locally (no GPU)

```bash
pip install -e ".[dev,bench]"
jobx bench fake-server --port 8000 --ttft 0.08 --itl 0.012 --output-tokens 60 \
  --max-num-seqs 16 --prefix-cache-factor 0.4 &
jobx bench run --model fake-model --workload extract-full --concurrency 1,8,32 --rate 20 \
  --num-requests 24 --out results/bench/raw/fake-demo.csv
jobx bench summarize results/bench/raw/fake-demo.csv --gpu L4 --out results/bench/summary.csv
```

The fake server streams one token every `--itl` seconds after `--ttft`, queues beyond `--max-num-seqs`,
scales TTFT by `--prefix-cache-factor` when the prompt prefix was seen before, and can inject HTTP 500s
(`--fail-every`). The test suite runs the same path end to end (`tests/test_bench.py`).

Notes on accuracy: run the load generator next to the server (inside the Modal container in phase 2), so
the numbers are not dominated by the network; the client ignores `HTTP(S)_PROXY` by default (`--trust-env`
to change that). vLLM pre-allocates `--gpu-memory-utilization` of the card, so peak GPU memory mostly
reflects that setting; weight and KV-cache sizes come from vLLM's startup log.

## Running on Modal

`src/jobx/bench/modal_app.py` runs one config per L4 container. The vLLM image is `vllm==0.33.0`
(`vllm_version` in the matrix) on `nvidia/cuda:12.8.1-devel-ubuntu24.04`; the llama.cpp image is
`ghcr.io/ggml-org/llama.cpp:server-cuda-b6000`, the same llama.cpp build as the CPU route. Hugging Face weights
are cached in the `jobx-hf-cache` Volume and every run is written to the `jobx-bench-results` Volume.

Each container ([`remote.py`](../src/jobx/bench/remote.py)) downloads the weights (plus the LoRA adapter from
release `model-v1`, or the GGUF), starts `vllm serve` / `llama-server` on localhost with the config's flags,
waits for `/health`, and runs the load generator in-process. Per level it samples GPU memory and scrapes
`/metrics` for KV-cache usage and the prefix-cache hit rate. When `quality: true` it then runs the eval.
It writes `results/bench/raw/<config>/<workload>.csv.gz` and `results/bench/manifests/<config>.json`
(vLLM version, GPU name, model revision, argv, wall time, estimated cost).

```bash
pip install -e ".[bench,modal]" && modal token new          # or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
jobx bench plan --config all                                 # estimate only, no account needed
modal run -m jobx.bench.modal_app --config q15-bf16
modal run -m jobx.bench.modal_app --matrix all [--yes]
```

Before starting anything it prints the estimate (startup + workloads x levels x (duration + 20 s) + quality
passes). It refuses to start above `MAX_GPU_MINUTES` (default 420) unless `--yes` is given. Each container
times out at 1.5 x its estimate + 15 min. A runtime guard launches a config only while actual plus in-flight
minutes stay within the cap.

## Orchestration (`bench.yml`)

[`.github/workflows/bench.yml`](../.github/workflows/bench.yml) runs on a push that changes
[`.github/bench-request.json`](../.github/bench-request.json), or on dispatch. Its first step checks for the
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repository secrets. If either is missing it prints a `::notice::`
and every later step is skipped, so nothing runs and nothing is billed. With the secrets it:

1. runs `jobx bench plan` on the request (estimate in the job summary; refused above the cap unless `yes`);
2. runs `modal run -m jobx.bench.modal_app` with up to 4 containers at once;
3. summarises, scores and regenerates the tables;
4. commits `results/bench/`, the predictions and metrics back to the branch, with a pull-rebase-push retry.

If some configs fail, the others are still committed and the job then fails.

Request file fields: `configs` (`"all"`, a list of names or `"group:<name>"`), `levels` (overrides for every
selected config: `concurrency`, `rates`, `duration_s`, `warmup`, `workloads`, `max_requests_per_level`),
`yes`, `skip_load`, `skip_quality`, `max_gpu_minutes`. Bump `id` to re-run the same request.

### Budget

`jobx bench plan` puts the full matrix at **303.5 L4-minutes (5.1 h)**. That is **$4.07 for the GPU** at
$0.80/h, or $5.64 with the 4 vCPU / 16 GiB the container also bills (Modal list prices, to verify). Both are
inside the ~7 L4-hour target. To keep the total there, the matrix was pruned:

- the `max_num_seqs` 16/256 variants run only the levels around the limit (16/64/256) and no Poisson rates;
- the prefix-caching-off pair runs only `extract-full`, the workload with a shared prefix;
- llama.cpp runs 4 concurrency levels and 2 rates;
- quality passes run only where the weights differ (not for scheduler or caching variants), and guided JSON
  only for BF16 and AWQ.

## Quality per quantization

A faster server only helps if the answers stay as good. Every config with `quality: true` therefore runs the
repo's own eval path against its local server once the load test is done. That path is
[`openai_compat_runner.py`](../src/jobx/baselines/openai_compat_runner.py) with greedy decoding and
`max_tokens` 384, over **all 471 `test_synth` postings** (no subset is needed: about 3 minutes per pass). The
base models get the same `SYSTEM_PROMPT` and 3 shots as the CPU baselines; the LoRA row gets the compact
prompt with zero shots, as it was trained.

- Output: the standard prediction files, `results/predictions/<approach>__test_synth.jsonl` plus `.meta.json`,
  with hardware, vLLM version and commit recorded.
- Scoring: `jobx eval` gives JSON validity, schema validity, field F1 and bootstrap CIs.
- Guided JSON: BF16 and AWQ get a second, guided pass (the JSON schema sent as `response_format`, which vLLM
  enforces with structured outputs), reported as a separate `-guided` row.
- Approach ids: `quality_approach` from the matrix plus `_few_shot` / `_zero_shot`.
- Table: the rows are listed in `configs/approaches.yaml` with `route: vllm` and `splits: [test_synth]`. While
  no predictions exist, the results table shows them as one "pending" line, so the CI tables check stays
  green.

`modal run -m jobx.bench.modal_app --matrix all --quality-only` (`jobx bench plan --quality-only`: 61
L4-minutes) runs only these passes.

## Analysis (`jobx bench report`, `jobx bench recommend`)

`jobx bench report` reads every raw CSV under `results/bench/raw/`, the manifests and the quality metrics. It
writes:

- `results/bench/summary.csv`: one row per config x workload x level, with latency percentiles,
  throughput, goodput, KV-cache usage, prefix-cache hit rate, GPU memory, USD per 1K requests and per 1M
  output tokens (GPU+CPU+RAM hourly rate), and micro F1 of the config's quality pass;
- `results/bench/recommendation.json`: the cheapest config for the target in the matrix's `recommend:`
  block (default: TTFT p95 <= 1 s, E2E p95 <= 5 s, 10 req/s, `extract-full`), plus the prefix-caching
  comparison;
- `results/bench/charts/`: `latency_throughput_<workload>.png` (E2E p95 and TTFT p95 against output
  tokens/s, one curve per config), `cost_per_1k_<workload>.png`, `prefix_caching_ttft.png` (on vs off pairs
  per concurrency) and `quality_vs_throughput.png`.

`jobx bench recommend --ttft-p95 0.5 --e2e-p95 3 --rps 40` answers other targets from `summary.csv`. The
recommender works from measured rows only:

1. For each config, the sustainable rate is the highest measured requests/s among the levels that met both
   p95 targets with < 1 % errors.
2. The required rate then needs `ceil(rps / sustainable)` containers. This assumes replicas scale linearly
   behind a load balancer.
3. The cheapest config by USD/hour wins; ties go to the one with more headroom.

The report is tested on fixture CSVs from the fake server (`tests/fixtures/bench/`, regenerated by
`make_fixtures.py`). Those fixtures are not measurements.
