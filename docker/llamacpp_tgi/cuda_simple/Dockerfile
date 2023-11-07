# Reference: https://github.com/abetlen/llama-cpp-python/blob/main/docker/cuda_simple/Dockerfile

ARG CUDA_IMAGE="12.2.2-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN mkdir /app
WORKDIR /app
COPY . /app

# Install dependencies
RUN git clone https://github.com/abetlen/llama-cpp-python.git
WORKDIR /app/llama-cpp-python
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context
RUN git submodule update --init --recursive
RUN make deps && make build && make clean
# Install llama-cpp-python (build with cuda)
RUN make build.cuda 
WORKDIR /app

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Set environment variable for the host
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose a port for the server
EXPOSE 8000

# Run the server and start the script
CMD ["/bin/sh", "entrypoint.sh"]
