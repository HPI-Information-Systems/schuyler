FROM openjdk:11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools python3-wheel \
    git \
    ant \
    curl \
    postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /experiment
COPY requirements.txt /experiment
RUN pip3 install --no-cache-dir -r requirements.txt

#COPY ./ /experiment
ADD ./ /experiment


RUN pip3 install -e .
