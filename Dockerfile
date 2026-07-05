# mini-vllm server image, sized for free CPU hosting (Hugging Face Spaces).
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel keeps the image ~1.5 GB instead of ~5 GB with CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY engine ./engine
COPY server ./server
RUN pip install --no-cache-dir ".[server]"

# Spaces runs containers as uid 1000; the model cache must live somewhere writable.
ENV HF_HOME=/tmp/hf-cache

EXPOSE 7860
CMD ["python", "-m", "server.api", "--host", "0.0.0.0", "--port", "7860"]
