FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HIRSCHBERG_CPU_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    MALLOC_ARENA_MAX=2

# Native libraries required by OpenCV contrib / MediaPipe and sounddevice.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/remicare
COPY requirements.txt requirements-render.txt ./
RUN python -m pip install --no-cache-dir -r requirements-render.txt \
    && python -m pip check \
    && python -c "import cv2, mediapipe, torch; assert hasattr(mediapipe, 'solutions')"

COPY app ./app
COPY scripts/verify_runtime_bundle.py ./scripts/verify_runtime_bundle.py
RUN python scripts/verify_runtime_bundle.py \
    && useradd --create-home remicare \
    && mkdir -p data/hirschberg/sessions \
    && chown -R remicare:remicare /srv/remicare
USER remicare
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','10000')+'/ready',timeout=8)"
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 16 --timeout-keep-alive 5"]
