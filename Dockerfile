FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel


ENV APT_INSTALL="apt-get install -y --no-install-recommends --fix-missing" \
    PIP_INSTALL="pip install"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN $APT_INSTALL build-essential sudo emacs git cmake libsm6 libxext6 \
        libxrender-dev libglib2.0-0 libgl1-mesa-glx libssl-dev ffmpeg \
        libopenmpi-dev libx264-dev


WORKDIR /workspace
COPY . /workspace

RUN $PIP_INSTALL --upgrade pip && \
    $PIP_INSTALL setuptools wheel && \
    $PIP_INSTALL transformers supervision roboflow timm einops onnx onnxruntime


CMD tail -f /dev/null
