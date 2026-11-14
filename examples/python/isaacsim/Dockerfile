FROM nvcr.io/nvidia/isaac-sim:6.1.0

ARG RLMESH_VERSION=0.2.0a1
RUN /isaac-sim/python.sh -m pip install --no-cache-dir "rlmesh[numpy]==${RLMESH_VERSION}" gymnasium

ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1 PYTHONPATH=/app RLMESH_ADDRESS=127.0.0.1:7447
COPY franka_reach /app/franka_reach

# `gpu` asks RLMesh Managed for a GPU node.
LABEL dev.rlmesh.package='{"schemaVersion":1,"name":"franka-reach","description":"Franka reach with an RTX camera on Isaac Sim 6.1.","tags":["gpu"],"resources":{"memory":"16Gi"}}'

EXPOSE 7447
# Written once the env answers, removed when it stops.
ENV RLMESH_READY_FILE=/tmp/rlmesh-ready.json
HEALTHCHECK --interval=5s --start-period=600s CMD test -f /tmp/rlmesh-ready.json
# python.sh never execs, so it would sit as PID 1 and swallow SIGTERM; set up
# its environment and exec Isaac's Python instead.
ENTRYPOINT ["bash", "-c", "source /isaac-sim/setup_python_env.sh && CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps LD_PRELOAD=/isaac-sim/kit/libcarb.so exec /isaac-sim/kit/python/bin/python3 \"$@\"", "--"]
CMD ["-m", "rlmesh.serve", "--env", "franka_reach:FrankaReach", "--name", "franka"]
