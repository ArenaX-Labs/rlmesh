FROM nvcr.io/nvidia/isaac-sim:6.1.0

ARG RLMESH_VERSION=0.1.0
RUN /isaac-sim/python.sh -m pip install --no-cache-dir "rlmesh[numpy]==${RLMESH_VERSION}" gymnasium

ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY franka_reach /app/franka_reach

# `gpu` asks RLMesh Managed for a GPU node.
LABEL dev.rlmesh.package='{"schemaVersion":1,"name":"franka-reach","description":"Franka reach with an RTX camera on Isaac Sim 6.1.","tags":["gpu"],"resources":{"memory":"16Gi"}}'

EXPOSE 50051
# python.sh never execs, so it would sit as PID 1 and swallow SIGTERM; set up
# its environment and exec Isaac's Python instead.
ENTRYPOINT ["bash", "-c", "source /isaac-sim/setup_python_env.sh && CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps LD_PRELOAD=/isaac-sim/kit/libcarb.so exec /isaac-sim/kit/python/bin/python3 \"$@\"", "--"]
CMD ["-m", "rlmesh.serve", "--env", "franka_reach:FrankaReach"]
