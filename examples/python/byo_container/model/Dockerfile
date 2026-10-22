# A bring-your-own model container: serves a policy on an RLMesh session.
# Build it, then drive it against any RLMesh env with SandboxModel:
#
#   docker build -t my-model:latest examples/python/byo_container/model
#
#   import rlmesh
#   rlmesh.SandboxModel("image://my-model:latest").run(env, episodes=1)
#
# RLMesh Managed runs the same tag. The container is driven by its own baked
# config: SandboxModel(image://...) injects no bootstrap payload.
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
