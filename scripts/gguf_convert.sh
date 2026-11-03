#!/usr/bin/env bash
# Convert a HF checkpoint (base or merged fine-tune) to GGUF F16 and quantise it with llama.cpp.
# usage: scripts/gguf_convert.sh <llama.cpp dir> <hf model dir> <out dir> [name] [quants...]
#   default quants: Q8_0 Q4_K_M ; the F16 file is always kept as <name>-F16.gguf
set -euo pipefail
LLAMA=${1:?llama.cpp dir}; MODEL=${2:?model dir}; OUT=${3:?out dir}; NAME=${4:-jobx-qwen2.5-1.5b}
shift $(( $# < 4 ? $# : 4 ))
QUANTS=("$@"); [ ${#QUANTS[@]} -eq 0 ] && QUANTS=(Q8_0 Q4_K_M)
mkdir -p "$OUT"
[ -f "$OUT/$NAME-F16.gguf" ] || python "$LLAMA/convert_hf_to_gguf.py" "$MODEL" --outtype f16 --outfile "$OUT/$NAME-F16.gguf"
for Q in "${QUANTS[@]}"; do
  [ "$Q" = "F16" ] && continue
  [ -f "$OUT/$NAME-$Q.gguf" ] || "$LLAMA/build/bin/llama-quantize" "$OUT/$NAME-F16.gguf" "$OUT/$NAME-$Q.gguf" "$Q"
done
ls -lh "$OUT"
