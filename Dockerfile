# Build stage
FROM python:3.12.3-slim-bullseye AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md LICENSE /app/
COPY opsmate /app/opsmate
RUN --mount=type=cache,target=/root/.cache \
    uv build

FROM python:3.12.3-slim-bullseye
LABEL org.opencontainers.image.source=https://github.com/opsmate-ai/opsmate

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256" && \
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl kubectl.sha256 && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/dist/opsmate-*.whl /tmp/dist/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir /tmp/dist/opsmate-*.whl && opsmate version

ENTRYPOINT ["opsmate"]
