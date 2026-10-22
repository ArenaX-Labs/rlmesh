# A bring-your-own model container: serves a policy on an RLMesh session.
# Build it, run it, then drive it against any RLMesh env:
#
#   docker build -t my-model:latest examples/python/byo_container/model
#   docker run --rm --network host my-model:latest
#
#   import rlmesh
#   rlmesh.Model("127.0.0.1:7447/model").run(env, episodes=1)
#
# The run invites the model into the session its env is in, so the container
# has to reach the driver's machine: host networking does that here.
# RLMesh Managed runs the same tag.
FROM python:3.11-slim

# Pin rlmesh to the build your host runs against (the handshake checks
# compatibility). Add your policy's framework + weights handling here.
RUN pip install --no-cache-dir "rlmesh" "gymnasium" "numpy"

WORKDIR /app
COPY entrypoint.py .

# A session host listens on four consecutive ports.
EXPOSE 7447-7450
ENV RLMESH_ADDRESS=0.0.0.0:7447
# Or skip entrypoint.py and serve the recipe directly:
#   ENTRYPOINT ["python", "-m", "rlmesh.serve", "entrypoint:MyPolicy"]
ENTRYPOINT ["python", "entrypoint.py"]
