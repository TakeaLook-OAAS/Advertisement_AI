FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV & video IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    ca-certificates \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python -m pip install --upgrade pip

# requirements 먼저 복사해서 도커 레이어 캐시를 제대로 활용
COPY requirements.txt /app/requirements.txt

# 설치
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Default command (can be overridden by docker compose)
CMD ["python", "-m", "src.main", "--config", "configs/dev.yaml", "--source", "0", "--no-window"]
