# A verifiers v0 environment as an env image. Build context: examples/python/llm_agent.
#
#   docker build -f docker/env-v0.Dockerfile -t llm-env-v0 .
#   docker build -f docker/env-v0.Dockerfile -t llm-env-mastermind --build-arg ENV_ID=mastermind .
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

ARG RLMESH_VERSION=0.2.0a1
COPY requirements-v0.txt overrides-v0.txt .
# wordle on PyPI is another package, so Prime's index goes first.
RUN uv pip install --system --no-cache \
      --index https://hub.primeintellect.ai/primeintellect/simple \
      --index-strategy unsafe-first-match --override overrides-v0.txt \
      -r requirements-v0.txt "rlmesh==${RLMESH_VERSION}"

WORKDIR /app
COPY chat_env.py serve_v0.py ./

ARG ENV_ID=wordle
ARG MAX_TURNS=
# A sandbox runs the container as the host user, who has no home here.
ENV ENV_ID=${ENV_ID} \
    MAX_TURNS=${MAX_TURNS} \
    HOME=/tmp \
    RLMESH_OUTPUT_DIR=/rlmesh/output/env \
    RLMESH_ADDRESS=0.0.0.0:7447 \
    RLMESH_NAME=env \
    RLMESH_READY_FILE=/tmp/rlmesh-ready.json
EXPOSE 7447
HEALTHCHECK --interval=2s --start-period=300s CMD test -f /tmp/rlmesh-ready.json
ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "serve_v0:EnvironmentEnv"]
