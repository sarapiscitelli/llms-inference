#!/bin/bash

# load env vars
set -a
source .env
set +a

# params checks
if [ -z "$MODEL" ]; then
  echo "Error: You must specify the MODEL in the .env file."
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

# TODO better manage llamacpp installation

check_torch_mps() {
    if python -c "import torch; print(torch.backends.mps.is_available())" | grep -q 'True'; then
        echo "Torch MPS device is available."
        return 0  # True
    else
        echo "Torch MPS device is not available."
        return 1  # False
    fi
}

check_cuda() {
    if nvidia-smi &> /dev/null; then
        echo "CUDA device is available."
        return 0  # True
    else
        echo "CUDA device is not available."
        return 1  # False
    fi
}

install_llama_cpp() {
    echo "Install standard llama-cpp"
    read -p "Do you want to install the standard llama-cpp-python package? (y/n) " answer
    if [ "$answer" == "y" ]; then
        pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install llama-cpp-python"
        fi
    else
        echo "llama-cpp-python installed."
    fi
}

# install dependencies
if check_torch_mps; then
    echo "Install llama-cpp with torch mps"
    read -p "Do you want to install the llama-cpp-python packages with mps device enable? (y/n) " answer
    if [ "$answer" == "y" ]; then
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install llama-cpp-python"
            install_llama_cpp
        fi
    else
        echo "llama-cpp-python with mps not installed."
    fi
elif check_cuda; then
    echo "Install llama-cpp with cuda"
    read -p "Do you want to install the llama-cpp-python packages with cuda? (y/n) " answer
    if [ "$answer" == "y" ]; then
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install llama-cpp-python"
            install_llama_cpp
        fi
    else
        echo "llama-cpp-python with cuda not installed."
    fi
else
    install_llama_cpp
fi

required_packages=("pip" "fastapi" "uvicorn" "sse-starlette" "pydantic-settings" "starlette-context")
echo "The following packages will be installed:"
for package in "${required_packages[@]}"; do
    echo " - $package"
done
read -p "Do you want to install the required packages? (y/n) " answer
if [ "$answer" != "y" ]; then
    echo "Installation aborted."
else
    pip install --upgrade "${required_packages[@]}"
fi

# start server
uvicorn --factory llama_cpp.server.app:create_app --host $HOSTNAME --port $PORT
