FROM python:3.11-slim

# System deps for OpenCV & video IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY requirements.txt /app/requirements.txt
# PyTorch CPU-only
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy source
COPY . /app

# Default command (can be overridden by docker compose)
CMD ["python", "-m", "src.main", "--config", "configs/dev.yaml", "--source", "0", "--no-window"]
