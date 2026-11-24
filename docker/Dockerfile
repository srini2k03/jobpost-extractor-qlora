# CPU image for the FastAPI extractor (llama-cpp-python + GGUF).
#   docker build -f docker/Dockerfile -t jobx-api .
#   # the released fine-tune (GitHub Release model-v1), mounted read-only:
#   docker run -p 8000:8000 -v "$PWD/models:/models:ro" \
#     -e JOBX_GGUF_PATH=/models/jobx-qwen2.5-0.5b-Q4_K_M.gguf jobx-api
#   docker run -p 8000:8000 -e JOBX_BACKEND=regex jobx-api      # no model, rule-based backend

# Stage 1: build a portable llama-cpp-python wheel (no prebuilt wheel exists for every version/platform).
FROM python:3.12-slim AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" PIP_NO_CACHE_DIR=1
RUN pip wheel --wheel-dir /wheels llama-cpp-python==0.3.16

# Stage 2: runtime.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models JOBX_PROMPT=compact JOBX_CALIBRATOR=/app/results/calibration/calibrator.json
WORKDIR /app
COPY --from=wheels /wheels /wheels
RUN pip install /wheels/llama_cpp_python-*.whl && rm -rf /wheels

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY results/calibration ./results/calibration
RUN pip install ".[serve,hub]"

RUN useradd --create-home app && mkdir -p /models && chown app /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "jobx.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
