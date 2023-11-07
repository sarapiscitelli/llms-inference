import os
import torch
from typing import List, Optional, Union
from fastapi import Depends, FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoConfig, GenerationConfig
   
class LmModel:

    model_dtypes_mapping: dict = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int8": torch.int8,
    }

    def __init__(self,
                 model_id_or_hf_model_path: str,
                 device: Optional[str] = None,
                 model_dtype: Optional[str] = "float32",
                 max_context_length: Optional[int] = None) -> None:
        if not device:
            device: str = "auto"
        self._model_id = model_id_or_hf_model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_hf_model_path)
        self._max_context_length = self._tokenizer.model_max_length
        if max_context_length:
            self._max_context_length = min(
                max_context_length, self._max_context_length)
        config = AutoConfig.from_pretrained(
            model_id_or_hf_model_path, trust_remote_code=True)
        if "flan-t5" in model_id_or_hf_model_path:
            from transformers import AutoModelForSeq2SeqLM
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_hf_model_path,
                                                                device_map=device,
                                                                config=config,
                                                                torch_dtype=self.model_dtypes_mapping[model_dtype])
        else:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(model_id_or_hf_model_path,
                                                               device_map=device,
                                                               config=config, 
                                                               torch_dtype=self.model_dtypes_mapping[model_dtype])
        # Managing the following excpetion: If `eos_token_id` is defined, make sure that `pad_token_id` is defined
        if self._model.generation_config.eos_token_id and not self._model.generation_config.pad_token_id:
            self._model.generation_config.pad_token_id = self._model.generation_config.eos_token_id

    @property
    def model_id(self) -> str:
        return str(self._model_id)

    @property
    def max_context_length(self) -> int:
        return int(self._max_context_length)

    @property
    def dtype(self) -> str:
        return str(self._model.dtype)

    @property
    def device(self) -> str:
        return str(self._model.device)

    def forward(self, prompt: str, generation_params: GenerationConfig) -> str:
        input_ids = self._tokenizer.encode(prompt,
                                           return_tensors='pt').to(self._model.device)
        output = self._model.generate(input_ids,
                                      generation_config=generation_params)
        out_sequence: str = self._tokenizer.decode(
            token_ids=output[0], skip_special_tokens=True
        )
        del output, input_ids  # free memory
        return out_sequence


router = APIRouter()
lm_model: Optional[LmModel] = None


# TODO manage multiple requests (lock, queue, etc.)?
def get_model():
    global lm_model
    if lm_model is None:
        raise RuntimeError("Model not initialized")
    return lm_model


class SettingsParams(BaseSettings):
    model_id: str = Field(default=os.environ.get("MODEL_ID", ""),
        description="The path to the hf model or the model if hosted on huggingface."
    )
    device: Optional[str] = Field(
        default=os.environ.get("DEVICE", "auto"),
        description="The device to use. 'auto' for cuda:0 if available, else mps device else cpu.",
    )
    max_context_length: Optional[int] = Field(
        default=os.environ.get("MAX_INPUT_LENGTH", 2048), ge=1, description="The maximum context size allowed.")

    model_dtype: Optional[str] = Field(default=os.environ.get("MODEL_DTYPE", "float32"), description="The dtype used for the model.")


def create_app(settings: Optional[SettingsParams] = None):
    if settings is None:
        settings = SettingsParams()

    app = FastAPI(
        title="Inference LLm API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    global lm_model
    lm_model = LmModel(model_id_or_hf_model_path=settings.model_id,
                       device=settings.device,
                       max_context_length=settings.max_context_length,
                       model_dtype=settings.model_dtype)

    return app


class GenerationRequestParams(BaseModel):
    prompt: str = Field(
        default="", description="The input prompt to generate from"
    )
    max_tokens: int = Field(
        default=16, ge=1, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0,
                               description="The temperature is used to module the next token probabilities. Must be strictly positive.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0,
                         description="Set the top p% of the probability distribution to sample from. Must be strictly positive.")
    top_k: float = Field(
        default=40, ge=0, description="Set the top k tokens to sample from. Must be strictly positive.")
    repeat_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, description="Set a penalty for repeating tokens. Must be strictly positive.")
    # not used yet TODO
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Provide a list of tokens to stop generation at.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "\n\n### Instructions:\nGenerate a simple story.\n\n### Response:\n",
                    "max_tokens": 64,
                    "temperature": 0.8,
                    "top_p": 0.95
                }
            ]
        }
    }


class GenerationRequestOutput(BaseModel):
    # TODO more information for the output
    generated_text: str


@router.post("/v1/generate")
async def generate(
        body: GenerationRequestParams,
        model: LmModel = Depends(get_model)) -> GenerationRequestOutput:
    """Generate text from a prompt."""
    # call model generation
    # TODO implement stop sequences!
    generation_params = GenerationConfig(do_sample=True,
                                         max_new_tokens=body.max_tokens,
                                         temperature=body.temperature,
                                         top_p=body.top_p,
                                         top_k=body.top_k,
                                         repetition_penalty=body.repeat_penalty,
                                         num_return_sequences=1)
    out_prompt: str = model.forward(prompt=body.prompt,
                                    generation_params=generation_params)
    return GenerationRequestOutput(generated_text=out_prompt)


# TODO implement stream
@router.post("/v1/generate_stream")
async def generate(
        body: GenerationRequestParams,
        model: LmModel = Depends(get_model)) -> GenerationRequestOutput:
    """Generate text from a prompt."""
    # call model generation with stream
    # TODO implement stop sequences!
    generation_params = GenerationConfig(do_sample=True,
                                         max_new_tokens=body.max_tokens,
                                         temperature=body.temperature,
                                         top_p=body.top_p,
                                         top_k=body.top_k,
                                         repetition_penalty=body.repeat_penalty,
                                         num_return_sequences=1)
    out_prompt: str = model.forward(prompt=body.prompt,
                                    generation_params=generation_params)
    return GenerationRequestOutput(generated_text=out_prompt)


class ModelInfo(BaseModel):
    model_name: str = Field(
        description="The path to the hf model or the model if hosted on huggingface.")
    model_max_length: int = Field(
        description="The maximum context size allowed.")
    device: str = Field(description="The device used for the model.")
    dtype: str = Field(description="The dtype used for the model.")


@router.get("/v1/info")
def get_info(model: LmModel = Depends(get_model)) -> ModelInfo:
    """Get information about the model."""
    return ModelInfo(model_name=model.model_id,
                     model_max_length=model.max_context_length,
                     device=model.device,
                     dtype=model.dtype)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(SettingsParams(
        model_id="google/flan-t5-large")), host="0.0.0.0", port=8080)
