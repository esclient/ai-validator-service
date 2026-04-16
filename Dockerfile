FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl jq wget && \
    wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 && \
    chmod +x /usr/local/bin/yq && \
    apt-get remove -y wget && apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml pdm.lock ./
COPY . .
RUN pip install --no-cache-dir pdm && \
    pdm export --without-hashes -f requirements > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    pip uninstall -y pdm && \
    rm /tmp/req.txt && \
    pip install --no-cache-dir -e . && \
    chmod +x tools/load_envs.sh

ENV ENV=prod
ENV PYTHONPATH=src

CMD ["./tools/load_envs.sh", "run-server"]
