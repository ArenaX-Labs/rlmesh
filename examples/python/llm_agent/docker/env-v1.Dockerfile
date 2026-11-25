# A verifiers v1 taskset as an env image. Build context: examples/python/llm_agent.
#
#   docker build -f docker/env-v1.Dockerfile -t llm-env-v1 .
#   docker build -f docker/env-v1.Dockerfile -t llm-env-humaneval \
#     --build-arg TASKSET=primeintellect/humaneval --build-arg RUNTIME=subprocess .
#   docker build -f docker/env-v1.Dockerfile -t llm-env-tau2 --target environment \
#     --build-arg ENV_ID=primeintellect/tau2-bench .
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/
# tau2 installs from git and fetches its domain data with git at startup.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG RLMESH_VERSION=0.2.0a1
COPY requirements-v1.txt .
# wordle-style names on PyPI are other packages, so Prime's index goes first.
RUN uv pip install --system --no-cache \
      --index https://hub.primeintellect.ai/primeintellect/simple \
      --index-strategy unsafe-first-match \
      -r requirements-v1.txt "rlmesh==${RLMESH_VERSION}"

WORKDIR /app
COPY chat_env.py serve_v1.py ./

ARG TASKSET=primeintellect/math500
ARG RUNTIME=
ARG ENV_ID=
ARG MAX_STEPS=
# A sandbox runs the container as the host user, who has no home here.
ENV TASKSET=${TASKSET} \
    RUNTIME=${RUNTIME} \
    ENV_ID=${ENV_ID} \
    MAX_STEPS=${MAX_STEPS} \
    HOME=/tmp \
    RLMESH_OUTPUT_DIR=/rlmesh/output/env \
    RLMESH_ADDRESS=0.0.0.0:7447 \
    RLMESH_NAME=env \
    RLMESH_READY_FILE=/tmp/rlmesh-ready.json
EXPOSE 7447
HEALTHCHECK --interval=2s --start-period=300s CMD test -f /tmp/rlmesh-ready.json

FROM base AS environment
ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "serve_v1:V1EnvEnv"]

FROM base AS taskset
ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "serve_v1:TasksetEnv"]
