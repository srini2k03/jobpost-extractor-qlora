#!/usr/bin/env bash
# Build llama.cpp tools portably (AVX2, no -march=native) so a cached build runs on any GitHub x64 runner.
# usage: scripts/build_llama_cpp.sh <git ref> <dir> [targets...]   (default targets: server, quantize, gguf-split)
set -euo pipefail
REF=${1:?ref}; DIR=${2:?dir}; shift 2
TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=(llama-server llama-quantize llama-gguf-split)
if [ ! -d "$DIR/.git" ]; then git clone --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$DIR"; fi
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DBUILD_SHARED_LIBS=OFF
cmake --build "$DIR/build" --target "${TARGETS[@]}" -j "$(nproc)"
ls "$DIR/build/bin"
