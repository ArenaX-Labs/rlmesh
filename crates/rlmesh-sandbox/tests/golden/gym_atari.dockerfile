# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV RLMESH_PORT=50051
ENV RLMESH_ENV_PORT=50051
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/rlmesh

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir 'rlmesh' && python -m pip install --no-cache-dir gymnasium && python -m pip install --no-cache-dir 'ale-py'

COPY recipe.json /opt/rlmesh/recipe.json
ENV RLMESH_RECIPE_PATH=/opt/rlmesh/recipe.json

EXPOSE 50051
ENTRYPOINT ["python", "-m", "rlmesh._bootstrap.sandbox_env"]
