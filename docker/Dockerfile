# CPU image for the FastAPI extractor (llama-cpp-python + GGUF).
#   docker build -f docker/Dockerfile -t jobx-api .
#   docker run -p 8000:8000 -e JOBX_GGUF_REPO=<user>/jobx-qwen2.5-1.5b-gguf jobx-api
#   docker run -p 8000:8000 -e JOBX_BACKEND=regex jobx-api      # no model, rule-based backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models JOBX_GGUF_FILE="*Q4_K_M.gguf"
WORKDIR /app

# Prebuilt CPU wheel: no compiler toolchain needed in the image.
RUN pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.16

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[serve,hub]"

RUN useradd --create-home app && mkdir -p /models && chown app /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "jobx.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
