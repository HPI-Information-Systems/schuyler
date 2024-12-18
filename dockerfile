FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
WORKDIR /experiment
ADD ./ /experiment
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN pip3 install -e .
