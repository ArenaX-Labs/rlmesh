# Images for the sandbox Docker gate (mise run test:sandbox:docker): the
# rlmesh wheel built from this checkout, and the bring-your-own example
# entrypoints. The gate passes its run label with `docker build --label`.
ARG BASE=python:3.12-slim
FROM ${BASE} AS base
COPY *.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl gymnasium numpy && rm -rf /tmp/wheels
WORKDIR /app

FROM base AS env
COPY env_entrypoint.py entrypoint.py
ENTRYPOINT ["python", "entrypoint.py"]

FROM base AS model
COPY model_entrypoint.py entrypoint.py
ENTRYPOINT ["python", "entrypoint.py"]

# An env whose make() fails: the container exits while the sandbox waits.
FROM base AS exits
COPY exits.py entrypoint.py
ENTRYPOINT ["python", "entrypoint.py"]
