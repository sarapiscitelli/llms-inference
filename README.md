# llms-inference
Explore large language models text generation inference using different solutions on local hardware  
This repository contains the code for running an inference server for large language models.  
The repository aims to test various existing solutions to deploy an existing large language model using local hardware resources and providing access through an API for inference.  
Three primary solutions have been explored:

- llamacpp python web server: https://github.com/abetlen/llama-cpp-python/tree/main#web-server
  <p align="center">
  <img width="700" alt="llamacpp_tgi" src="https://github.com/sarapiscitelli/llms-inference/assets/104431794/8612eda8-25b1-46bd-8c95-6b41b9d66a95">
  </p>
- huggingface text generation inference server: https://github.com/huggingface/text-generation-inference
  <p align="center">
  <img width="700" alt="hf_tgi" src="https://github.com/sarapiscitelli/llms-inference/assets/104431794/05f21d75-768e-47d5-a82a-3fd5cca9db93">
  </p>
- A simple server implemented at https://github.com/sarapiscitelli/llms-inference/blob/main/lm-tgi-servers/local/simple_tgi/app.py to allow inference of models from HuggingFace. In this case, inference is not optimized.

  <p align="center">
  <img width="700" alt="simple_tgi" src="https://github.com/sarapiscitelli/llms-inference/assets/104431794/8d15e1c0-0122-4fae-8566-886a881b68f4">
  </p>
In all cases, uvicorn servers are used, exposing the model through APIs via FastAPI.   
Therefore, after running the server, you can find the OpenAPI definitions by navigating to the following link: http://localhost:8080/docs.

# How to Run
Go to the folder of a specific server where the `./start_server.sh` script is located.  
At the same level, the .env file specifies the supported parameters and configurations; make sure to check that some parameters are mandatory.  
To run the server, use the following command:
```console
./start_server.sh
```

# Project Structure
```console
├── docker
│   └── huggingface_tgi
│   └── llamacpp_tgi
├── local-servers
│   └── simple_tgi
│   └── llamacpp_tgi
├── .gitignore
├── LICENSE
├── README.md
```

## Local Servers
A local uvicorn server will be launched, enabling language model inference via an API.   
In this case, a local environment will be used, so it is recommended to create a new environment and activate it with the following steps:  
```
conda create --name tgi-inference-env python=3.10 -y
conda activate tgi-inference-env
```

#### llamacpp_tgi
The [llamacpp server](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py) will be executed locally.  
Before running it, you need to download a quantized model locally (for example, one of the available models at [Huggingface](https://huggingface.co/TheBloke)), and then set the MODEL variable in the .env file.  
```console
HOSTNAME=0.0.0.0
PORT=8080

MODEL="/mistral-7b-instruct-v0.1.Q8_0.gguf"  # full path to a downloaded model (.gguf format)
N_GPU_LAYERS=-1 # The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU
# N_CTX=2048 # The allowed context length for the user in tokens (is not the model maximum length)
```
To run the server, use the following command:
```console
./start_server.sh
```
Please note that the script will attempt to install the necessary llama_cpp library based on your hardware to provide hardware acceleration. If the installation fails, you may need to install the library manually. For more information, refer to the instructions on [llamacpp library installation](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration)


#### simple_tgi
This will run a web server with uvicorn implemented in the [`app.py` file](https://github.com/sarapiscitelli/llms-inference/blob/main/local-servers/simple_tgi/app.py), providing minimal support for language model inference.   
There are no optimizations for inference or handling of concurrent requests.  
Before execution, please review the variables in the .env file, and if needed, modify the MODEL_ID parameter with the name of the Hugging Face Hub model you want to use.  
```console
HOSTNAME=0.0.0.0
PORT=8080

MODEL_ID=mistralai/Mistral-7B-Instruct-v0.1 # google/flan-t5-large # mistralai/Mistral-7B-Instruct-v0.1
# MAX_INPUT_LENGTH=1024 # maximum allowed input length (expressed in number of tokens) for users (is not for the model!).
# DEVICE=auto # cpu or cuda or mps, specify the device to use
MODEL_DTYPE=float16 # float32 or float16 or int8, specify the data type to use
```
You should also consider changing the MODEL_DTYPE parameter to adjust the precision, allowing you to load larger models if your hardware is insufficient (note that lower precision may affect model performance).  

The server will be launched with the following command:

```console
./start_server.sh
```

## Docker

A Docker container will be launched to run a server for language model inference.  

#### llamacpp_tgi:  
This runs the container of the [llama-cpp-python library](https://github.com/abetlen/llama-cpp-python#docker-image). It is capable of executing quantized models in formats like GGUF, AWQ, GPTQ, and is primarily optimized for Apple Silicon.  
Before running, you have to download the model locally, it is needed a quantized model, for example one that can be donwload from [Huggingface](https://huggingface.co/TheBloke), and put it in the directory specified in the .env file shown below.

```console
HOSTNAME=0.0.0.0
PORT=8080

DOWNLOAD_MODELS_DIR="" # the local directory where the models are located
MODEL_NAME=mistral-7b-instruct-v0.1.Q4_0.gguf
N_GPU_LAYERS=-1 # The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU
N_CTX=1024 # The allowed context length for the user in tokens (is not the model maximum length)
```
Change the DOWNLOAD_MODELS_DIR to the directory where the model downloaded is located and the MODEL_NAME to the name of the model.
The server will be launched with the following command:

```console
./start_server.sh
```

#### huggingface_tgi:  
This will run the container of the [huggingface text-generation-inference library](https://github.com/huggingface/text-generation-inference#docker), providing support for quantized models and optimized inference.  
Before running, look at the .env for the environment variables that can be set to configure the server.

```console
HOSTNAME=0.0.0.0
PORT=8080

MODEL_ID=google/flan-t5-large # mistralai/Mistral-7B-Instruct-v0.1
MAX_INPUT_LENGTH=1024 #  maximum allowed input length (expressed in number of tokens) for users (is not for the model!).
DTYPE=float16 # optional, possible values: float16, bfloat16 The dtype to be forced upon the model. This option cannot be used with `QUANTIZE`
QUANTIZE= # optional, quantize the model ([possible values: awq, eetq, gptq, bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4]). Not set together with `DTYPE`
HUGGING_FACE_HUB_TOKEN= # optional (valid only for private models)
```
Change the MODEL_ID to the model you want to run, it will be automatically downloaded from the Hugging Face Hub. (Please note that the download can take a while, depending on the model size and your internet connection speed). To avoid this, you can download the model locally and the server will use it from the local Hugging Face cache ($HOME/.cache/huggingface/hub).  
Depending on the model and your available hardware, you may need to set the DTYPE or QUANTIZE environment variables.  
The server will be launched with the following command:

```console
./start_server.sh
```
