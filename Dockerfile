# Build stage
FROM python:3.12.3-slim-bullseye AS builder

ENV POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

WORKDIR /app

# Install poetry and dependencies in one layer
COPY poetry.lock pyproject.toml README.md /app/
COPY opsmate /app/opsmate
RUN pip install --no-cache-dir poetry==$POETRY_VERSION && \
    poetry build && \
    rm -rf $POETRY_CACHE_DIR

# Final stage
FROM python:3.12.3-slim-bullseye

LABEL org.opencontainers.image.source=https://github.com/jingkaihe/opsmate

# Install only kubectl without keeping unnecessary files
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256" && \
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl kubectl.sha256 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/dist/opsmate-*.whl /tmp/dist/

RUN pip install --no-cache-dir /tmp/dist/opsmate-*.whl

ENTRYPOINT ["opsmate"]
