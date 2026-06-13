# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV RLMESH_ENV_PORT=50051
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV PYTHONPATH=/opt/LIBERO

WORKDIR /opt/rlmesh

RUN apt-get update && apt-get install -y --no-install-recommends 'cmake' 'g++' 'libegl1-mesa-dev' 'libgl1' 'libglib2.0-0' && rm -rf /var/lib/apt/lists/*

COPY project /opt/robot_env
RUN python -m pip install --no-cache-dir -e '/opt/robot_env'

RUN git clone --depth=1 'https://github.com/Lifelong-Robot-Learning/LIBERO.git' '/opt/LIBERO' && git -C '/opt/LIBERO' fetch --depth=1 origin 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' && git -C '/opt/LIBERO' checkout 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' && python -m pip install --no-cache-dir -r '/opt/LIBERO'/requirements.txt && python -m pip install --no-cache-dir -e '/opt/LIBERO' && rm -rf '/opt/LIBERO'/.git

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir rlmesh && python -m pip install --no-cache-dir gymnasium && python -m pip install --no-cache-dir --index-url 'https://download.pytorch.org/whl/cu124' 'torch' 'torchvision' && python -m pip install --no-cache-dir 'robosuite==1.4.1'

RUN useradd --create-home --uid 1000 rlmesh && chown -R 1000 /opt/rlmesh
USER 1000

EXPOSE 50051
ENTRYPOINT ["python", "-m", "rlmesh._bootstrap.sandbox_env"]
