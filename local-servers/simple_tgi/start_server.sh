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

auto_confirm() {
    if [ "$1" == "-y" ]; then
        return 0
    else
        return 1
    fi
}

required_packages=("pip" "fastapi" "uvicorn" "pydantic-settings" "transformers" "accelerate")
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

# run the server
uvicorn --factory app:create_app --host $HOSTNAME --port $PORT
