FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    build-essential \
    gcc \
    g++ \
    cmake \
    libsndfile1 \
    libsndfile1-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the ClearerVoice-Studio repository
RUN git clone https://github.com/modelscope/ClearerVoice-Studio.git /app/ClearerVoice-Studio

# Install Python dependencies
WORKDIR /app/ClearerVoice-Studio

# Install pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy and cython before other requirements (helps with building extensions)
RUN pip install --no-cache-dir numpy cython

# Install problematic packages individually with build isolation
RUN pip install --no-cache-dir --no-build-isolation pesq
RUN pip install --no-cache-dir --no-build-isolation pysptk
RUN pip install --no-cache-dir --no-build-isolation pyworld

# Now install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests

# Copy the handler script
COPY handler.py /app/handler.py

# Create directories for file handling
RUN mkdir -p /app/inputs /app/outputs /app/temp

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app/ClearerVoice-Studio"

# Set the entrypoint
CMD ["runpod", "--", "python", "-u", "/app/handler.py"]