# A verifiers v1 taskset as an env image. Build context: examples/python/llm_agent.
#
#   docker build -f docker/env-v1.Dockerfile -t llm-env-v1 .
#   docker build -f docker/env-v1.Dockerfile -t llm-env-humaneval \
#     --build-arg TASKSET=primeintellect/humaneval --build-arg RUNTIME=subprocess .
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

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
# A sandbox runs the container as the host user, who has no home here.
ENV TASKSET=${TASKSET} \
    RUNTIME=${RUNTIME} \
    HOME=/tmp \
    RLMESH_OUTPUT_DIR=/rlmesh/output/env \
    RLMESH_ADDRESS=0.0.0.0:7447 \
    RLMESH_NAME=env \
    RLMESH_READY_FILE=/tmp/rlmesh-ready.json
EXPOSE 7447
HEALTHCHECK --interval=2s --start-period=300s CMD test -f /tmp/rlmesh-ready.json
ENTRYPOINT ["sh", "-c", "exec python serve_v1.py --taskset \"$TASKSET\" ${RUNTIME:+--runtime \"$RUNTIME\"} --address \"$RLMESH_ADDRESS/$RLMESH_NAME\""]
