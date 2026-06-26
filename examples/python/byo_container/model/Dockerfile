# A bring-your-own model container: serves a policy as an RLMesh model endpoint.
# Build it, then drive it against any RLMesh env with SandboxModel:
#
#   docker build -t my-model:latest examples/python/byo_container/model
#
#   import rlmesh
#   env = rlmesh.RemoteEnv("127.0.0.1:50051")
#   sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)
#   obs, _ = env.reset()
#   model.reset()
#   done = False
#   while not done:
#       action = model.predict(obs)
#       obs, reward, terminated, truncated, _ = env.step(action)
#       done = terminated or truncated
#
# RLMesh Managed runs the same tag. The container is driven by its own baked
# config: SandboxModel(image://...) does NOT inject a bootstrap payload.
FROM python:3.11-slim

# Pin rlmesh to the build your host runs against (the handshake checks
# compatibility). Add your policy's framework + weights handling here.
RUN pip install --no-cache-dir "rlmesh" "gymnasium" "numpy"

WORKDIR /app
COPY entrypoint.py .

EXPOSE 50051
ENV RLMESH_ADDRESS=0.0.0.0:50051
# Or skip entrypoint.py and serve the recipe directly:
#   ENTRYPOINT ["python", "-m", "rlmesh.serve", "entrypoint:MyPolicy"]
ENTRYPOINT ["python", "entrypoint.py"]
