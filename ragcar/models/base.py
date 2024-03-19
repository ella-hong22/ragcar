import logging
import json
import uuid
from typing import Any, Optional, Union, Tuple, Generator, AsyncGenerator, Dict
import math
import os 
import requests
import aiohttp
import openai
from ragcar.tools.utils.bench import log_gpu_memory_usage

import bitsandbytes as bnb
from peft import (
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import QuantLinear
import transformers
from transformers import (  # noqa: F401
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    LlamaForCausalLM,
    LlamaTokenizer,

)

import torch

logger = logging.getLogger(__name__)



class ClovaBase:
    """Base class for HyperCLOVA models"""
    
    # Define the cost per token for each model outside the function
    COST_PER_TOKEN = {
        'HCX-003': 0.005,
        'HCX-003-tuning': 0.03,
        'HCX-002': 0.005,
        'HCX-002-tuning': 0.03,
        'LK-D': 0.04,
        'LK-D-tuning': 0.12,
        'LK-C': 0.015,
        'LK-C-tuning': 0.045,
        'LK-B': 0.0025,
        'LK-B-tuning': 0.0075,
    }
    
    def __init__(self, api_url: str, api_key: str, app_key: str, stream: Optional[bool] = False):
        self.api_url = api_url
        
        parts = api_url.split('/')
        if 'tasks' in api_url:  # HyperCLOVA tuning model
            last_two = parts[-2:]

            if last_two[-1] == 'completions':
                last_two[-1] = 'LK-D'
            elif last_two[-1] == 'chat-completions':
                last_two[-1] = 'HCX-003'

            self.model_n = '/'.join(last_two)
        else:
            self.model_n = parts[-1]
        
        self.headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-APIGW-API-KEY': api_key,
            'X-NCP-CLOVASTUDIO-API-KEY': app_key,
            # 'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id
        }
        
        if stream:
            self.headers["Accept"] = "text/event-stream"
        
        self.stream = stream
    
    def _to_camel(self, snake_str: str) -> str:
        """
        Convert a snake_case string to camelCase.
        
        Args:
            snake_str (str): The snake_case string to be converted.

        Returns:
            str: The converted camelCase string.
        """
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def _get_model_type(self, model_n: str) -> str:
        """
        Extract the model type from Clova engine string.

        Args:
            model_n (str): The model name string from which to extract the model type

        Returns:
            str: The model type extracted from the engine string.

        Raises:
            ValueError: If the engine string format is invalid.
        """
        try:
            return f"{model_n.split('/')[-1]}-tuning" if "/" in model_n else model_n
        except IndexError:
            raise ValueError("Invalid url string format. Unable to extract model type.")

    def _calculate_charge_per_request(self, model_type: str, total_tokens: int) -> Optional[float]:
        """
        Calculate the total cost of a request based on the number of input and output tokens and the engine used.

        Args:
            model_type (str): The type of the model.
            total_tokens (int): The total number of tokens used in a request.

        Returns:
            Optional[float]: The total cost of the request in KRW or None if the model type is not recognized.
        """
        if model_type not in self.COST_PER_TOKEN:
            return None

        cost_per_input_token = self.COST_PER_TOKEN[model_type]
        total_cost_in_krw = total_tokens * cost_per_input_token

        return total_cost_in_krw
    
    def format_response(
        self, 
        response_data,
        request_id: str,
        response_time: float
    ) -> Dict[str, Union[str, int, float, Dict]]:
        """
        Format the response from the API.

        Args:
            response_data (Dict): The raw response from the API, expected to contain result details.
            request_id (str): The unique identifier of the request.
            response_time (float): The time (in seconds) it took to get the response from the API.

        Returns:
            Dict[str, Union[str, int, float, Dict]]: A dictionary containing formatted response details, including:
                - id (str): The request identifier.
                - model (str): The model name used for the request.
                - content (str): The main content message from the response.
                - finish_reason (str): The reason why the operation was finished.
                - input_tokens (int): The number of tokens in the input.
                - output_tokens (int): The number of tokens in the output.
                - total_tokens (int): The total number of tokens processed.
                - predicted_cost (Union[str, float]): The predicted cost of the operation, or "Unknown" if not calculable.
                - response_time (float): The response time of the API call.
                - ai_filter (Dict): A dictionary with AI filter scores, if applicable.
        """
        result = response_data.get('result', {})
        
        content = result.get('message', {}).get('content', '').strip()
        
        finish_reason = result.get('stopReason')
        
        input_tokens = result.get('inputLength')
        output_tokens = result.get('outputLength')
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        model_type = self._get_model_type(self.api_url)
        predicted_cost = self._calculate_charge_per_request(model_type, total_tokens)
        
        ai_filter = result.get('aiFilter')
        if ai_filter:
            ai_filter = {item['name']: int(item['score']) for item in ai_filter}
        
        formatted_data = {
            "id": request_id,
            "model": self.model_n,
            "content": content,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "predicted_cost": predicted_cost if predicted_cost is not None else "Unknown",
            "response_time": response_time,
            "ai_filter": ai_filter
        }
            
        return formatted_data
    
    def fetch(self, **kwargs: dict) -> dict:
        """
        Send a POST request to the API.

        Args:
            kwargs (dict): The parameters for the request, which are passed as keyword arguments.

        Returns:
            dict: The response from the API.
        
        Raises:
            RuntimeError: If the API does not respond with a '20000' status code.
        """
        request_id = f"clova-{str(uuid.uuid4())}"
        
        # Convert snake_case keys to camelCase
        kwargs = {self._to_camel(k): v for k, v in kwargs.items()}
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.api_url,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = requests.post(
            self.api_url,
            json=kwargs, 
            headers=self.headers,
            stream=self.stream
        )
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        if self.stream:
            return response, request_id
        
        parsed_data = response.json()
        
        if parsed_data['status']['code'] == '20000':
            return parsed_data, request_id
        
        raise RuntimeError((f"Request failed with status code: {parsed_data['status']}"))

    async def afetch(self, **kwargs: dict) -> dict:
        """
        Send an asynchronous POST request to the API.

        Args:
            kwargs (dict): The parameters for the request, which are passed as keyword arguments.

        Returns:
            dict: The response from the API.
        
        Raises:
            RuntimeError: If the API does not respond with a '20000' status code.
        """
        request_id = f"clova-{str(uuid.uuid4())}"
        
        # Convert snake_case keys to camelCase
        kwargs = {self._to_camel(k): v for k, v in kwargs.items()}
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.api_url,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, 
                data=json.dumps(kwargs), 
                headers=self.headers
            ) as response:
                logger.info(
                    json.dumps(
                        {
                            "id": request_id,
                            "response": "Request completed successfully"
                        }, ensure_ascii=False, indent=4
                    )
                )
                
                if self.stream:
                    return await response.text(), request_id
                
                data = await response.text()
                parsed_data = json.loads(data)
                
                if parsed_data['status']['code'] == '20000':
                    return parsed_data, request_id
                
                raise RuntimeError((f"Request failed with status code: {parsed_data['status']}"))


class OpenaiBase:
    """Base class for OpenAI models"""
    
    # Define the cost per token for each model outside the function
    # https://openai.com/pricing
    COST_PER_TOKEN = {
        'gpt-4-base': {
            'input': 0.03 / 1000,
            'output': 0.06 / 1000
        },
        'gpt-4-large': {
            'input': 0.06 / 1000,
            'output': 0.12 / 1000
        },
        'gpt-4-turbo': {
            'input': 0.01 / 1000,
            'output': 0.03 / 1000
        },
        'gpt-3.5': {
            'input': 0.0010 / 1000,
            'output': 0.0020 / 1000
        },
        'gpt-3.5-instruct': {
            'input': 0.0015 / 1000,
            'output': 0.0020 / 1000
        },
        'gpt-3.5-tuning': {
            'input': 0.0030 / 1000,
            'output': 0.0060 / 1000
        },
        'davinci': {
            'input': 0.0020 / 1000,
            'output': 0.0020 / 1000
        },
        'davinci-tuning': {
            'input': 0.0120 / 1000,
            'output': 0.0120 / 1000
        },
        'babbage': {
            'input': 0.0004 / 1000,
            'output': 0.0004 / 1000
        },
        'babbage-tuning': {
            'input': 0.0016 / 1000,
            'output': 0.0016 / 1000
        }
    }
    
    def __init__(self, model_n: str, api_key: str):
        self.model_n = model_n
        openai.api_key = api_key

    def _get_model_type(self, model_n: str) -> str:
        """
        Extract the model type from OpenAI engine string.

        Args:
            model_n (str): The model name string from which to extract the model type.

        Returns:
            str: The model type extracted from the engine string.

        Raises:
            ValueError: If the engine string format is invalid.
        """
        if model_n.startswith('text-'):
            model_n = model_n.replace('text-', '', 1)

        if 'gpt-3.5' in model_n:
            if 'instruct' in model_n:
                return 'gpt-3.5-instruct'
            elif model_n.startswith('ft:'):
                return 'gpt-3.5-turbo-tuning'
            else:
                return 'gpt-3.5'
        elif 'gpt-4' in model_n:
            if '32k' in model_n:
                return 'gpt-4-large'
            elif '1106' in model_n:
                return 'gpt-4-turbo'
            else:
                return 'gpt-4-base'
        else:
            try:
                if model_n.startswith('ft:'):
                    return model_n.split('-')[0].replace('ft:', '') + "-tuning"
                else:
                    return model_n.split('-')[0]
            except IndexError:
                raise ValueError("Invalid model name string format. Unable to extract model type.")

    def _calculate_charge_per_request(self, model_type: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Calculate the total cost of a request based on the number of input and output tokens and the engine used.

        Args:
            model_type (str): The type of the model.
            input_tokens (int): The input number of tokens used in a request.
            output_tokens (int): The output number of tokens used in a request.

        Returns:
            Optional[float]: The total cost of the request in KRW or None if the model type is not recognized.
        """
        if model_type not in self.COST_PER_TOKEN:
            return None

        total_cost_in_usd = 0

        if input_tokens:
            cost_per_input_token = self.COST_PER_TOKEN[model_type]['input']
            total_cost_in_usd += input_tokens * cost_per_input_token

        if output_tokens:
            cost_per_output_token = self.COST_PER_TOKEN[model_type]['output']
            total_cost_in_usd += output_tokens * cost_per_output_token

        return total_cost_in_usd

    def format_response(
        self,
        response_data,
        request_id: str,
        response_time: float
    ) -> Dict[str, Union[str, int, float]]:
        """
        Format the response from the API.

        Args:
            response_data (dict): The raw response data from the API.
            request_id (str): The unique identifier for the request.
            response_time (float): The duration it took to receive the response, in seconds.

        Returns:
            Dict[str, Union[str, int, float]]: A dictionary containing the formatted response data. This includes:
                - id (str): The request ID.
                - model (str): The model name used for the request.
                - content (str): The main content returned by the API.
                - finish_reason (str): The reason provided by the API for the request's completion.
                - input_tokens (int): The number of tokens used in the input.
                - output_tokens (int): The number of tokens generated as output.
                - total_tokens (int): The total number of tokens used (input + output).
                - predicted_cost (Union[str, float]): The predicted cost of the operation, or "Unknown" if not calculable.
                - response_time (float): The response time for the API call.
        """
        choice = response_data.get('choices', [{}])[0]
        
        model_type = self._get_model_type(self.model_n)
        
        if model_type.startswith('gpt'):
            message = choice.get('message', {})
            content = message if message.get('function_call') else message.get('content', '').strip()
        else:
            content = choice.get('text', '').strip()
        
        finish_reason = choice.get('finish_reason')
        
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens')
        output_tokens = usage.get('completion_tokens')
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        predicted_cost = self._calculate_charge_per_request(model_type, input_tokens, output_tokens)
        
        formatted_data = {
            "id": request_id,
            "model": self.model_n,
            "content": content,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "predicted_cost": predicted_cost if predicted_cost is not None else "Unknown",
            "response_time": response_time
        }
            
        return formatted_data

    def fetch(self, create_fn, **kwargs) -> Union[Tuple[dict, str], Tuple[Generator[dict, None, None], str]]:
        """
        Calls an external service and loggers the request and response.

        Args:
            create_fn (Callable): The function to call the external service.
            **kwargs: Arbitrary keyword arguments passed to the external service call.

        Returns:
            Union[Tuple[dict, str], Tuple[Generator[dict, None, None], str]]: A tuple containing the response 
                from the external service and the request ID.
        """
        request_id = f"openai-{str(uuid.uuid4())}"
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.model_n,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = create_fn(model=self.model_n, **kwargs)
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        return response, request_id
    
    async def afetch(self, create_fn, **kwargs) -> Union[Tuple[dict, str], Tuple[AsyncGenerator[dict, None], str]]:
        """
        Asynchronously calls an external service and loggers the request and response.

        Args:
            create_fn (Callable): The function to call the external service.
            **kwargs: Arbitrary keyword arguments passed to the external service call.

        Returns:
            Union[Tuple[dict, str], Tuple[AsyncGenerator[dict, None], str]]: A tuple containing the response 
                from the external service and the request ID.
        """
        request_id = f"openai-{str(uuid.uuid4())}"
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.model_n,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = await create_fn(model=self.model_n, **kwargs)
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        return response, request_id
    


class Sllmbase:
    tokenizer_type = {
        "MistralForCausalLM": "LlamaTokenizer",
        "LlamaForCausalLM": "LlamaTokenizer",
    }

    def __init__(self, model_n: str, **kwargs):
        self.model_n = model_n

    def _load_model_config(self):
        model_config_name = self.model_n
        config_kwargs = {}
        try:
             
            model_config = AutoConfig.from_pretrained(
                model_config_name,
                **config_kwargs,
                )
        except ValueError as err:
            raise err
        
        return model_config
    

    def _load_tokenizer(self, model_n: str, model_config) -> AutoTokenizer:
        tokenizer_kwargs = {}
        tokenizer_cls = getattr(transformers, self.tokenizer_type[model_config.architectures[0]])

        tokenizer = tokenizer_cls.from_pretrained(
            model_n,
            trust_remote_code= False,
            # use_fast=use_fast,
            **tokenizer_kwargs,
        )

        if (
            tokenizer.__class__.__name__
            in [
                "LlamaTokenizer",
                "LlamaTokenizerFast",
                "CodeLlamaTokenizer",
                "CodeLlamaTokenizerFast",
            ]
            and hasattr(tokenizer, "pad_token")
            and not tokenizer.pad_token
        ):
            # set a pad_token, but use eos_token so we don't add a new token
            LLAMA_DEFAULT_EOS_TOKEN = "</s>"
            tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

        if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 이코드 추가에 따른 결과 확인 필요
        if "mistral" in self.model_n:
            logger.info("Mistral model detected, setting padding side to left")
            tokenizer.padding_side = "left"

        additional_special_tokens = None

        # logger.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        # logger.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        # logger.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        # logger.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

        # to change print statements to logging
        print(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        print(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        print(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        print(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

        return tokenizer   

    def _select_device(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
        elif torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32

        logger.info(f"Using device {device} with dtype {torch_dtype}")

        return device, torch_dtype
    
    def get_linear_embedding_layers(self, model_type):
        """
        returns the linear embedding layers needed for loras, dependent on the model arch
        """
        if model_type == "gpt_neox":
            return ["embed_in", "embed_out"]
        if model_type == "falcon":
            return ["word_embeddings", "lm_head"]
        return ["embed_tokens", "lm_head"]
    
    def _load_model(self, 
                    tokenizer: PreTrainedTokenizerBase, 
                    model_n: str, 
                    model_config: AutoConfig, 
                    lora_model_dir:Optional[str] = None, 
                    adapter: str = None, 
                    **kwargs
                    ) -> AutoModelForCausalLM:
        
        base_model = model_n
        model_config = model_config
        model_type = getattr(transformers, self.tokenizer_type[model_config.architectures[0]])
        lora_model_dir = lora_model_dir if lora_model_dir else None
        adapter = adapter if adapter else None

        flash_attention=  True
        sample_packing= True
        self.device, self.torch_dtype = self._select_device()
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "load_in_8bit": False,
            "load_in_4bit": False,
        }


        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            model_kwargs["device_map"] = "mps"
        else:
            model_kwargs["device_map"] = "auto"

        # Modify mistral derived models
        if (
            model_config.model_type == "mistral"
            and flash_attention
            and sample_packing
        ):
            from ragcar.tools.monkeypatch.mistral_attn_hijack_flash import (
                replace_mistral_attn_with_flash_attn,
            )

            logger.info("patching mistral with flash attention")
            replace_mistral_attn_with_flash_attn(packed=sample_packing)

        if model_config.model_type == "llama" and sample_packing :
            from ragcar.tools.monkeypatch.llama_expand_mask import hijack_expand_mask
            logger.info("patching _expand_mask in llama models")
            hijack_expand_mask()

        
        if torch.backends.mps.is_available():
            logger.info("using mps device")
            model_kwargs["device_map"] = "mps:0"


        if adapter == "qlora" : # load_in_4bit
            logger.info("loading in 4bit")
            #하이브리드 양자화 방식(일부 레이어나 가중치에 대해 8bit 적용 -> 정확도 유지)
            bnb_config = {
                "load_in_4bit": True,
                "llm_int8_threshold": 6.0, 
                "llm_int8_has_fp16_weight": False,
                "bnb_4bit_compute_dtype": model_kwargs["torch_dtype"],
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )
            model_kwargs["load_in_4bit"] = True

        logger.info(f"model_kwargs: {model_kwargs}")

        try:
            if (
                model_config.model_type == "llama"
            ):  
                from transformers import LlamaForCausalLM
                logger.info("loading llama model")
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    config=model_config,
                    **model_kwargs,
                )


            elif model_type :
                logger.info("loading {model_type}")
                model = getattr(transformers, model_type).from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=False,
                    **model_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        config=model_config,
                        trust_remote_code= False,
                        **model_kwargs,
                    )
                
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.exception(err)
            raise err
        
        if isinstance(model, (PeftModel, PeftModelForCausalLM)):
            model = model.merge_and_unload()

        embeddings_len = (
            math.ceil(len(tokenizer) / 32) * 32
        )
        logger.info(f"resizing token embeddings to {embeddings_len}")

        if (
            hasattr(model, "get_input_embeddings")
            and model.get_input_embeddings().num_embeddings < embeddings_len
        ):
            model.resize_token_embeddings(embeddings_len)
        else:
            model.tie_weights()

        if (
            hasattr(model, "config")
            and hasattr(model.config, "bos_token_id")
            and model.config.bos_token_id
            and model.config.bos_token_id != tokenizer.bos_token_id
        ):
            model.config.bos_token_id = tokenizer.bos_token_id

        if (
            hasattr(model, "config")
            and hasattr(model.config, "eos_token_id")
            and model.config.eos_token_id
            and model.config.eos_token_id != tokenizer.eos_token_id
        ):
            model.config.eos_token_id = tokenizer.eos_token_id

        logger.info(f"model config: {model.device.type}")
        if hasattr(model, "device") and model.device.type in ("cuda", "mps"):
            log_gpu_memory_usage(logger, "after model load", model.device)

        embedding_modules = self.get_linear_embedding_layers(model_config.model_type)

        #fsdp false 일때
        for name, module in model.named_modules():
            if "norm" in name or name.endswith(".gate"):
                module.to(torch.float32)
            if model_config.model_type == "btlm":
                # don't upcast lm_head for btlm
                continue
            if any(m in name for m in embedding_modules):
                if hasattr(module, "weight"):
                    module.to(torch.float32)

        needs_fa2_dtype = adapter 
        skip_prepare_model_for_kbit_training = False


        if adapter in ["lora", "qlora"]:
            gradient_checkpointing = True
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()

            if (
                model_kwargs["load_in_8bit"] or model_kwargs["load_in_4bit"]
            ) and not skip_prepare_model_for_kbit_training:
                logger.info("converting PEFT model w/ prepare_model_for_kbit_training")
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=gradient_checkpointing
                )
            needs_fa2_dtype = True


        # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
        # convert them back to fp16/bf16 for flash-attn compatibility.
        if needs_fa2_dtype or flash_attention:
            logger.info("converting modules to %s for flash attention", model_kwargs["torch_dtype"])
            for name, module in model.named_modules():
                if "norm" in name:
                    module.to(model_kwargs["torch_dtype"])
                if any(m in name for m in embedding_modules):
                    if hasattr(module, "weight"):
                        module.to(model_kwargs["torch_dtype"])
                
        # TODO : flash_attention 코드 분석후에, torch_dtype=torch.float16 추가 필요
        # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
        # convert them back to fp16/bf16 for flash-attn compatibility.
        # if needs_fa2_dtype or cfg.flash_attention:
        #     logger.info("converting modules to %s for flash attention", cfg.torch_dtype)
        #     for name, module in model.named_modules():
        #         if "norm" in name:
        #             module.to(cfg.torch_dtype)
        #         if any(m in name for m in embedding_modules):
        #             if hasattr(module, "weight"):
        #                 module.to(cfg.torch_dtype)


        lora_config = None
        if lora_model_dir:
            # if we're not loading the reference model, then we're loading the model for training
            # then the dpo trainer doesn't want the peft model loaded over it, it just wants the lora/peft config
            model, lora_config = self.load_adapter(model, lora_model_dir, adapter)

        if torch.cuda.device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
            setattr(model, "is_parallelizable", True)
            setattr(model, "model_parallel", True)
    
        requires_grad = []
        for name, param in model.named_parameters(recurse=True):
            if param.requires_grad:
                requires_grad.append(f"{name}: {param.requires_grad}")
        if len(requires_grad) == 0:
            logger.warning("there are no parameters that require gradient updates")
        if hasattr(model, "config"):
            model.config.use_cache = False

        if adapter is not None:
            log_gpu_memory_usage(logger, "after adapters", model.device)

        return model, lora_config

    # TODO : 여기서부터 코드 다시 분석하기. 
    def load_adapter(self, model, lora_model_dir, adapter= None):
        # type: (PreTrainedModel, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
        logger.info(f"1. Loading pretrained PEFT - {adapter}")
        if adapter is None:
            return model, None
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if adapter in ["lora", "qlora"]:
            return self.load_lora(model, lora_model_dir)
        if adapter == "llama-adapter":
            return self.load_llama_adapter(model, lora_model_dir)

        raise NotImplementedError(f"{adapter} peft adapter not available")


    def load_llama_adapter(self, model, lora_model_dir):
        # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
        from peft import AdaptionPromptConfig, get_peft_model

        #TODO : peft_adapter에 대한 코드 분석 필요
        # peft_config = AdaptionPromptConfig(
        #     adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        #     adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        #     task_type="CAUSAL_LM",
        # )

        if lora_model_dir:
            logger.debug("Loading pretrained PEFT - llama_adapter")
            model = PeftModel.from_pretrained(
                model,
                lora_model_dir,
                torch_dtype=torch.float16,
            )
        # else:
        #     model = get_peft_model(model, peft_config)

        peft_config = None
        model.print_trainable_parameters()

        return model, peft_config


    def find_all_linear_names(self, model):
        cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                or "Linear" in module.__class__.__name__
                and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
            ):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        embedding_modules = self.get_linear_embedding_layers(model.config.model_type)
        output_embedding = embedding_modules[1]
        if output_embedding in lora_module_names:  # needed for 16-bit
            lora_module_names.remove(output_embedding)

        return list(lora_module_names)


    def load_lora(self, model, lora_model_dir, config_only=False):
        logger.debug("Loading pretrained PEFT - LoRA")
        # type: (PreTrainedModel, DictDefault, bool, bool) -> Tuple[Optional[PreTrainedModel], Optional[PeftConfig]]
        model = PeftModel.from_pretrained(
            model,
            lora_model_dir,
            is_trainable=False, #false일때 모델 가중치 고정
            torch_dtype=torch.float16, #TODO 있고 없고의 차이 
        )

        lora_config = None
        model.print_trainable_parameters()


        return model, lora_config
    
     