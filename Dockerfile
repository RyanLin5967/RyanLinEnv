FROM python:3.12.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml /opt/vision-rl/pyproject.toml
COPY src /opt/vision-rl/src
RUN pip install --no-cache-dir /opt/vision-rl && \
    python -c "from pathlib import Path; from vision_rl.spatial import generate_split; generate_split(Path('/app/data'),'train',1024,42); generate_split(Path('/app/data'),'validation',256,100045)"
CMD ["vision-rl", "--help"]
