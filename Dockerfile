FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

WORKDIR /build
COPY requirements-docker.lock ./
RUN pip install --no-cache-dir -r requirements-docker.lock
COPY pyproject.toml README.md ./*.py ./
RUN pip install --no-cache-dir --no-deps . && pip check
COPY tests ./tests

# Run as an unprivileged user; mount a writable output directory or pass --user.
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["wgbs-impute-docker"]
