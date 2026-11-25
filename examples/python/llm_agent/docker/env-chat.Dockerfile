# chat_env's AdditionTask as an env image; needs no verifiers. Build context: examples/python/llm_agent.
#
#   docker build -f docker/env-chat.Dockerfile -t llm-env-chat .
FROM python:3.11-slim

ARG RLMESH_VERSION=0.2.0a1
RUN pip install --no-cache-dir "rlmesh==${RLMESH_VERSION}"

WORKDIR /app
COPY chat_env.py ./

ENV RLMESH_OUTPUT_DIR=/rlmesh/output/env \
    RLMESH_ADDRESS=0.0.0.0:7447 \
    RLMESH_NAME=env \
    RLMESH_READY_FILE=/tmp/rlmesh-ready.json
EXPOSE 7447
HEALTHCHECK --interval=2s --start-period=60s CMD test -f /tmp/rlmesh-ready.json
ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "chat_env:AdditionTask"]
