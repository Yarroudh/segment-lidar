FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY segment_lidar ./segment_lidar

ARG INSTALL_EXTRAS=sam3
RUN python -m pip install --no-cache-dir --upgrade pip && \
    if [ -n "$INSTALL_EXTRAS" ]; then \
      python -m pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"; \
    else \
      python -m pip install --no-cache-dir .; \
    fi

RUN useradd --create-home appuser
RUN mkdir -p /data && chown appuser:appuser /data
USER appuser
WORKDIR /data
CMD ["python"]
