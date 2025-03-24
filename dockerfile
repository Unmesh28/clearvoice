FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the ClearerVoice-Studio repository
RUN git clone https://github.com/modelscope/ClearerVoice-Studio.git /app/ClearerVoice-Studio

# Install Python dependencies
WORKDIR /app/ClearerVoice-Studio
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