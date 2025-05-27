# Use NVIDIA's Python base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu20.04
# RUN nvidia-smi
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
RUN mkdir /tmp/models
WORKDIR /experiment

COPY ./schuyler/requirements.txt /experiment/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install xformers --index-url https://download.pytorch.org/whl/cu121

COPY ./schuyler /experiment
RUN pip3 install --no-cache-dir -e .

ENV HF_HOME=/models
