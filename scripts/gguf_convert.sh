#!/usr/bin/env bash
# Convert a merged HF checkpoint to GGUF and quantise it (used by .github/workflows/quantize_benchmark.yml).
# usage: scripts/gguf_convert.sh <llama.cpp dir> <merged model dir> <out dir> [name]
set -euo pipefail
LLAMA=${1:?llama.cpp dir}; MODEL=${2:?merged model dir}; OUT=${3:?out dir}; NAME=${4:-jobx-qwen2.5-1.5b}
mkdir -p "$OUT"
python "$LLAMA/convert_hf_to_gguf.py" "$MODEL" --outtype f16 --outfile "$OUT/$NAME-F16.gguf"
for Q in Q8_0 Q4_K_M; do
  "$LLAMA/build/bin/llama-quantize" "$OUT/$NAME-F16.gguf" "$OUT/$NAME-$Q.gguf" "$Q"
done
ls -lh "$OUT"
