FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libffi-dev \
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
RUN git clone https://github.com/Unmesh28/clearvoice.git /app

# Create and activate conda environment
RUN conda create -n ClearerVoice-Studio python=3.8 -y
SHELL ["conda", "run", "-n", "ClearerVoice-Studio", "/bin/bash", "-c"]

# Install Python dependencies with special handling for problematic packages
RUN pip install --upgrade pip setuptools wheel
RUN pip install Cython numpy
# Install problematic packages separately with specific flags
RUN pip install pesq --no-build-isolation
RUN pip install pysptk --no-build-isolation
RUN pip install pyworld --no-build-isolation
# Now install the rest of the requirements
RUN pip install -r requirements.txt || echo "Some packages may have failed, continuing anyway"
RUN pip install runpod

# Create directories for input/output
RUN mkdir -p /tmp/inputs /tmp/outputs

# Set the working directory to clearvoice directory where main.py is located
WORKDIR /app/clearvoice

# Set the entrypoint for conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ClearerVoice-Studio", "python", "-u", "main.py"]