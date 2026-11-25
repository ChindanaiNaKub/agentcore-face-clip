# AgentCore runs Graviton only; x86 images fail to start.
FROM --platform=linux/arm64 python:3.12.4-slim AS builder
WORKDIR /app
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM --platform=linux/arm64 python:3.12.4-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -r -u 1001 appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY agent/main.py scripts/face_clip.py ./
USER appuser
EXPOSE 8080
# Binds 0.0.0.0 for AgentCore internal routing. Do NOT expose to the internet.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
