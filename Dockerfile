FROM python:3.10-slim

# System dependencies (ffmpeg + build tools for numpy/scipy/torch etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    unzip \
    build-essential \
    gcc \
    g++ \
    libsndfile1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Workaround for this package
RUN pip install --no-cache-dir --no-build-isolation "nemo-toolkit[asr]==1.23.0"

# Copy application code and default configs
COPY app/ ./app/
# Keep both default_settings.json and a seeded config.json for first run
COPY config/default_settings.json ./config/default_settings.json
COPY config/prompt_templates.json ./config/prompt_templates.json
RUN cp ./config/default_settings.json ./config/config.json

# Create directories
RUN mkdir -p /models /config /tmp

EXPOSE 55667

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "55667"]
