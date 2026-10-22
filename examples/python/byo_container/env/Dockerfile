# A bring-your-own env container: hosts an RLMesh session and serves a
# Gymnasium env in it. Build it, then run it as a sandbox (locally) or hand
# the tag to RLMesh Managed.
#
#   docker build -t my-env:latest examples/python/byo_container/env
#   python -c "import rlmesh; e=rlmesh.SandboxEnv('image://my-env:latest'); print(e.reset())"
FROM python:3.11-slim

# Pin rlmesh to the build your host runs against. Add your env's own
# dependencies here too.
RUN pip install --no-cache-dir "rlmesh" "gymnasium"

WORKDIR /app
COPY entrypoint.py .

# A session host listens on four consecutive ports.
EXPOSE 7447-7450
ENV RLMESH_ADDRESS=0.0.0.0:7447
# Or skip entrypoint.py and serve the recipe directly:
#   ENTRYPOINT ["python", "-m", "rlmesh.serve", "--env", "entrypoint:MyEnv"]
ENTRYPOINT ["python", "entrypoint.py"]
