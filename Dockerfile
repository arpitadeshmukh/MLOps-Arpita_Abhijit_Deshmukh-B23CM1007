FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt




RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

WORKDIR /workspace
