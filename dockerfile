# Use NVIDIA's Python base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04


# RUN nvidia-smi
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3-pip postgresql-client git ant curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-dev    
RUN export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y build-essential
ENV PYTHONUNBUFFERED=1
RUN mkdir /tmp/models
WORKDIR /experiment
RUN pip3 install Cython
RUN pip3 install hdbscan
RUN pip3 install packaging

COPY ./schuyler/requirements.txt /experiment/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
ENV VLLM_LOGGING_LEVEL=DEBUG
ENV NCCL_P2P_DISABLE=1
ENV GPU_MEMORY_UTILIZATION=0.9
ENV OMP_NUM_THREADS=2
#RUN pip3 install vllm==0.8.5


COPY ./schuyler /experiment
RUN pip3 install --no-cache-dir -e .

ENV HF_HOME=/models
