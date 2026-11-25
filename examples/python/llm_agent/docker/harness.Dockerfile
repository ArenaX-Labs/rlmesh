# The LLM harness as a model image. Build context: examples/python/llm_agent.
#
#   docker build -f docker/harness.Dockerfile -t llm-harness .
ARG BASE=python:3.11-slim
FROM ${BASE}

# A base that already has this rlmesh keeps it.
ARG RLMESH_VERSION=0.2.0a1
RUN pip install --no-cache-dir "rlmesh==${RLMESH_VERSION}" openai

WORKDIR /app
COPY chat_env.py harness.py fake_llm.py ./

# Ollama on the Docker Desktop host; on Linux pass the bridge gateway (README).
ARG OPENAI_BASE_URL=http://host.docker.internal:11434/v1
ARG OPENAI_MODEL=qwen3:4b
ENV OPENAI_BASE_URL=${OPENAI_BASE_URL} \
    OPENAI_API_KEY=ollama \
    OPENAI_MODEL=${OPENAI_MODEL} \
    OPENAI_MAX_TOKENS=8192 \
    RLMESH_OUTPUT_DIR=/rlmesh/output/model \
    RLMESH_ADDRESS=0.0.0.0:7447 \
    RLMESH_READY_FILE=/tmp/rlmesh-ready.json
EXPOSE 7447
HEALTHCHECK --interval=2s --start-period=60s CMD test -f /tmp/rlmesh-ready.json
ENTRYPOINT ["python", "-m", "rlmesh.serve", "harness:LLMHarness"]
