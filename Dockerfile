FROM python:3.12-slim-bookworm AS build

ARG PIP_TIMEOUT_SECONDS=300
ARG PIP_RETRY_COUNT=10
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.13.0+cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src
RUN python -m venv /opt/rag-venv
RUN --mount=type=cache,id=simple-rag-pip,target=/root/.cache/pip,sharing=locked \
    /opt/rag-venv/bin/pip install \
    --timeout "${PIP_TIMEOUT_SECONDS}" \
    --retries "${PIP_RETRY_COUNT}" \
    --index-url "${PYTORCH_CPU_INDEX_URL}" \
    "torch==${TORCH_VERSION}"
COPY pyproject.toml README.md ./
COPY rag_app ./rag_app
RUN --mount=type=cache,id=simple-rag-deps-py312-v2,target=/root/.cache/pip,sharing=locked \
    /opt/rag-venv/bin/pip install \
    --timeout "${PIP_TIMEOUT_SECONDS}" \
    --retries "${PIP_RETRY_COUNT}" \
    .

FROM python:3.12-slim-bookworm AS app

ENV PATH="/opt/rag-venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_BASE_DIR=/app \
    RAG_DATA_DIR=/data \
    RAG_PAPERS_DIR=/app/papers \
    RAG_EMBED_DEVICE=cpu \
    HF_HOME=/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/models/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LANGSMITH_TRACING=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rag \
    && install -d -m 0755 -o rag -g rag /app/papers /data /models/huggingface
COPY --from=build /opt/rag-venv /opt/rag-venv

USER rag
WORKDIR /app
EXPOSE 8000

CMD ["local-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
