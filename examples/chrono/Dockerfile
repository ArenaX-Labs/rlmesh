# The Project Chrono env as a bring-your-own container: a native C++ binary
# serving RLMesh on port 50051, with no Python in the image. Build from the
# repository root (the env links this checkout's rlmesh-capi):
#
#   docker build -f examples/chrono/Dockerfile -t chrono-reach:dev .
#   docker run --rm -p 50051:50051 chrono-reach:dev
#
# A LABEL cannot run the binary, so the describe label is baked from the built
# image (see README.md, "Run it as a container"):
#
#   docker build -f examples/chrono/Dockerfile -t chrono-reach:dev \
#     --label dev.rlmesh.describe="$(docker run --rm chrono-reach:dev --describe)" .
#
# --build-arg VULKAN=ON adds the Vulkan camera (`--renderer vulkan`) with the
# Vulkan loader and Mesa's lavapipe; on an NVIDIA host the GPU takes over:
#
#   docker build -f examples/chrono/Dockerfile --build-arg VULKAN=ON -t chrono-reach:vulkan .
#   docker run --rm --device nvidia.com/gpu=all -p 50051:50051 chrono-reach:vulkan --renderer vulkan

ARG DEBIAN=trixie
ARG VULKAN=OFF

# ---- rlmesh-capi: the RLMesh C ABI, built from this checkout -----------------
FROM rust:1.99-${DEBIAN} AS capi
ARG TARGETARCH
ARG PROTOC_VERSION=36.2
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/* \
 && case "${TARGETARCH:-amd64}" in arm64) arch=aarch_64 ;; *) arch=x86_64 ;; esac \
 && curl -fsSL -o /tmp/protoc.zip \
    "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-${arch}.zip" \
 && unzip -q /tmp/protoc.zip -d /usr/local && rm /tmp/protoc.zip
WORKDIR /src
COPY . .
# A release cohort, so the env negotiates with released rlmesh peers (a dev
# build pins a per-commit edition only that same build speaks).
ENV RLMESH_RELEASE_BUILD=1
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build -p rlmesh-capi --release --lib \
 && mkdir -p /out/lib /out/include \
 && cp target/release/librlmesh_capi.so /out/lib/ \
 && cp crates/rlmesh-capi/include/rlmesh.h crates/rlmesh-capi/include/rlmesh.hpp /out/include/

# ---- Project Chrono core + robot models ----------------------------------------
FROM debian:${DEBIAN} AS chrono
ARG CHRONO_VERSION=10.0.0
ARG EIGEN_VERSION=5.0.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" | tar xz \
 && cmake -S eigen-${EIGEN_VERSION} -B eigen-build -DCMAKE_INSTALL_PREFIX=/opt/eigen \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_TESTING=OFF >/dev/null \
 && cmake --install eigen-build >/dev/null
# CPU-only core: no CUDA/Thrust (multicore math), no demos, no visualization.
RUN curl -fsSL "https://github.com/projectchrono/chrono/archive/refs/tags/${CHRONO_VERSION}.tar.gz" | tar xz \
 && cmake -S chrono-${CHRONO_VERSION} -B chrono-build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/chrono -DEigen3_DIR=/opt/eigen/share/eigen3/cmake \
      -DBUILD_DEMOS=OFF -DBUILD_TESTING=OFF -DBUILD_BENCHMARKING=OFF -DCH_ENABLE_OPENMP=ON \
      -DCMAKE_CUDA_COMPILER= -DCMAKE_DISABLE_FIND_PACKAGE_Thrust=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON \
 && cmake --build chrono-build -j"$(nproc)" \
 && cmake --install chrono-build >/dev/null \
 && rm -rf chrono-build chrono-${CHRONO_VERSION}

# ---- the env -----------------------------------------------------------------
FROM chrono AS env
ARG VULKAN
RUN if [ "$VULKAN" = ON ]; then \
      apt-get update && apt-get install -y --no-install-recommends libvulkan-dev glslang-tools \
      && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=capi /out /opt/rlmesh
COPY examples/chrono /src/chrono
RUN cmake -S /src/chrono -B /src/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/opt/chrono;/opt/eigen" -DCHRONO_REACH_VULKAN="$VULKAN" \
      -DRLMESH_INCLUDE_DIR=/opt/rlmesh/include -DRLMESH_LIBRARY_DIR=/opt/rlmesh/lib \
 && cmake --build /src/build -j"$(nproc)" \
 && cmake --install /src/build --prefix /opt/app \
 && mkdir -p /opt/app/lib \
 && cp /opt/rlmesh/lib/librlmesh_capi.so /opt/chrono/lib/libChrono*.so /opt/app/lib/

# ---- runtime -------------------------------------------------------------------
FROM debian:${DEBIAN}-slim
ARG VULKAN
# Vulkan: the loader and lavapipe, plus the EGL and X libraries NVIDIA's Vulkan
# driver loads (the container toolkit mounts the driver itself with the GPU).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
      $([ "$VULKAN" = ON ] && echo libvulkan1 mesa-vulkan-drivers libegl1 libx11-6 libxext6) \
 && rm -rf /var/lib/apt/lists/*
COPY --from=env /opt/app /opt/app
# The binary's RUNPATH is $ORIGIN/../lib, so the shared libraries resolve there.
# It serves on $RLMESH_ADDRESS, which the platform sets per pod (default
# 0.0.0.0:50051).

LABEL dev.rlmesh.package='{"schemaVersion":1,"name":"chrono-industrial-reach","description":"Project Chrono 10 six-axis industrial robot reaching a target, served natively from C++ through the RLMesh C ABI.","tags":["chrono","cpp"],"resetOptions":["trial_index"],"resources":{"memory":"1Gi"}}'

EXPOSE 50051
ENTRYPOINT ["/opt/app/bin/chrono_reach_env"]
