# ID Card OCR API — production Dockerfile (NVIDIA GPU)
#
# Base image already ships a CUDA-matched PyTorch build, which saves a huge
# download/compile step and guarantees torch + the NVIDIA driver agree on a
# CUDA version. If your host's NVIDIA driver is older, pick an older
# cuda12.1 tag here instead (the driver must support the CUDA version below).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System libraries opencv-python needs at runtime (headless servers usually
# lack these — without them cv2 import fails with a libGL.so.1 error).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across rebuilds whenever
# only application code changes (much faster iterative builds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (code + model weight files + TrOCR checkpoint
# folder). Anything that shouldn't ship in the image is excluded via
# .dockerignore (venv, caches, logs, credentials.json, sample images, etc.)
COPY . .

# Defaults — overridable at "docker run -e" / docker-compose time.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# credentials.json is intentionally NOT copied into the image (see
# .dockerignore) — mount it as a volume at runtime instead, so the API
# key/secret never end up baked into an image layer. See docker-compose.yml.

CMD ["python", "api.py"]
