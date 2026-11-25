# syntax=docker/dockerfile:1
# AgentCore Runtime requires ARM64.
FROM --platform=linux/arm64 python:3.12.4-slim AS builder
WORKDIR /app
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM --platform=linux/arm64 python:3.12.4-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -r -u 1001 appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY agent/main.py scripts/face_swap.py scripts/download_models.py ./
ARG INSIGHTFACE_ROOT=/opt/insightface
ENV INSIGHTFACE_ROOT=$INSIGHTFACE_ROOT
RUN python download_models.py --root "$INSIGHTFACE_ROOT" \
    && chmod -R a+rX "$INSIGHTFACE_ROOT"
USER appuser
EXPOSE 8080
# Binds 0.0.0.0 for AgentCore internal routing. Do NOT expose to the internet.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
