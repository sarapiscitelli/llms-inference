#!/bin/bash

# load env vars
set -a
source .env
set +a

# params checks
if [ -z "$MODEL_ID" ]; then
  echo "Error: You must specify the MODEL_ID in the .env file."
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

if [ -n "$DTYPE" ]; then
  DTYPE_ENV="-e DTYPE=$DTYPE"
else
  DTYPE_ENV=""
fi

if [ -n "$QUANTIZE" ]; then
  QUANTIZE_ENV="-e QUANTIZE=$QUANTIZE"
else
  QUANTIZE_ENV=""
fi

if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
  HUGGING_FACE_HUB_TOKEN_ENV="-e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
else
  HUGGING_FACE_HUB_TOKEN_ENV=""
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  CUDA_VISIBLE_DEVICES_ENV="-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  CUDA_VISIBLE_DEVICES_ENV=""
fi

# run the server
docker run --rm --gpus all --shm-size 1g -p $PORT:$PORT -e MODEL_ID=$MODEL_ID \
-e MAX_INPUT_LENGTH=$MAX_INPUT_LENGTH \
-e MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS \
-e HOSTNAME=$HOSTNAME -e PORT=$PORT \
 $DTYPE_ENV $QUANTIZE_ENV $HUGGING_FACE_HUB_TOKEN_ENV $CUDA_VISIBLE_DEVICES_ENV \
-v $HOME/.cache/huggingface/hub:/data \
ghcr.io/huggingface/text-generation-inference:1.1.0
