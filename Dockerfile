# syntax=docker/dockerfile:1
# Stage 1 - resolve wheels (all runtime deps ship wheels; no compiler needed).
FROM python:3.11-slim AS wheels
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2 - train at build time from the committed sample telemetry so the
# runtime image is self-contained (Hugging Face Spaces cannot mount volumes;
# docker users can still shadow models/ and data/ with mount volumes).
FROM python:3.11-slim AS train
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt
COPY data ./data
COPY src ./src
RUN python src/train.py --data data/sample_telemetry.csv --out-dir models \
    && test -s models/pdm_bundle.joblib \
    && test -s models/metadata.json

# Stage 3 - slim runtime, non-root, stdlib healthcheck.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# App code plus the build-time model artifacts; compose still overlays
# data/ and models/ with mounted volumes at run time.
COPY data ./data
COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY --from=train /app/models ./models

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
