FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y apt-utils build-essential curl wget zip unzip git make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Install pip and setuptools
RUN pip install --upgrade pip setuptools

# Expose any ports the app is expected to run on
EXPOSE 8885 8884
