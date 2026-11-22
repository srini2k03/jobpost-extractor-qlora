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

## Phase 2 (not yet)

A Modal app that launches each matrix config on an L4, runs the load generator in the same container,
pulls the raw CSVs back into `results/bench/raw/` and runs the quality pass. It will need Modal credentials
as `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (environment variables, or GitHub Actions secrets for a manual
workflow); the owner will create them, and they are never committed.
