FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Create working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/modelscope/ClearerVoice-Studio.git /app

# Create and activate conda environment
RUN conda create -n ClearerVoice-Studio python=3.8 -y
SHELL ["conda", "run", "-n", "ClearerVoice-Studio", "/bin/bash", "-c"]

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install runpod

# Create directories for input/output
RUN mkdir -p /tmp/inputs /tmp/outputs

# Copy the modified main.py
# Note: main.py must be in the same directory as the Dockerfile when building
COPY ./main.py /app/clearvoice/main.py

# Set the working directory to clearvoice directory
WORKDIR /app/clearvoice

# Set the entrypoint for conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ClearerVoice-Studio", "python", "-u", "main.py"]