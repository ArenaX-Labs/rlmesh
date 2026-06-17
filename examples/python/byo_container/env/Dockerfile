# A bring-your-own env container: serves a Gymnasium env over the RLMesh
# protocol on port 50051. Build it, then dial it with rlmesh.RemoteEnv (locally)
# or hand the tag to RLMesh Managed.
#
#   docker build -t my-env:latest examples/python/byo_container/env
#   docker run --rm -p 50051:50051 my-env:latest
#   # in another shell / process:
#   python -c "import rlmesh; e=rlmesh.RemoteEnv('127.0.0.1:50051'); print(e.reset())"
FROM python:3.11-slim

# Pin rlmesh to the build your host runs against (the protocol handshake checks
# compatibility). Add your env's own dependencies here too.
RUN pip install --no-cache-dir "rlmesh" "gymnasium"

WORKDIR /app
COPY entrypoint.py .

EXPOSE 50051
ENV RLMESH_ADDRESS=0.0.0.0:50051
ENTRYPOINT ["python", "entrypoint.py"]
