#!/bin/bash

# load env vars
set -a
source .env
set +a

# params checks
if [ -z "$DOWNLOAD_MODELS_DIR" ]; then
  echo "Error: You must specify the DOWNLOAD_MODELS_DIR in the .env file."
  exit 1
fi

if [ -z "$MODEL_NAME" ]; then
  echo "Error: You must specify the MODEL_NAME in the .env file."
  exit 1
fi

if [ -z "$HOSTNAME" ]; then
  echo "Error: You must specify the HOSTNAME in the .env file."
  exit 1
fi

if [ -z "$PORT" ]; then
  echo "Error: You must specify the PORT in the .env file."
  exit 1
fi

# check if cuda is available
check_cuda() {
    if nvidia-smi &> /dev/null; then
        echo "CUDA device is available."
        return 0  # True
    else
        echo "Warning: CUDA device is not available and mps not supported. Will be executed on cpu"
        return 1  # False
    fi
}


if check_cuda; then
    dockerfile_path="./cuda_simple/Dockerfile"
    echo "Runnnig custom llamacpp server with cuda"
    docker build -t llamacpp-server-cuda -f $dockerfile_path ./cuda_simple
    docker run --rm -it -p $PORT:8000 \
    -v $DOWNLOAD_MODELS_DIR:/models \
    -e MODEL=/models/$MODEL_NAME \
    -e n_ctx=$N_CTX \
    -e n_gpu_layers=$N_GPU_LAYERS \
    llamacpp-server-cuda
else
    echo "Running default llamacpp server"
    docker run --rm -it -p $PORT:8000 \
    -v $DOWNLOAD_MODELS_DIR:/models \
    -e MODEL=/models/$MODEL_NAME \
    -e n_ctx=$N_CTX \
    -e n_gpu_layers=$N_GPU_LAYERS \
    ghcr.io/abetlen/llama-cpp-python:latest
fi
