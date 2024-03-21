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
from tools.utils.model_config import DictDefault

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
    AddedToken,
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


    def __init__(self, model_n: str, **kwargs):
        self.model_n = model_n


    def get_linear_embedding_layers(model_type):
        """
        returns the linear embedding layers needed for loras, dependent on the model arch
        """
        if model_type == "gpt_neox":
            return ["embed_in", "embed_out"]
        if model_type == "falcon":
            return ["word_embeddings", "lm_head"]
        return ["embed_tokens", "lm_head"]

    def load_model_config(cfg: DictDefault):
        model_config_name = cfg.base_model_config or cfg.base_model
        config_kwargs = {}
        try:
            model_config = AutoConfig.from_pretrained(
                model_config_name,
                **config_kwargs,
                )
        except ValueError as err:
            raise err
        
        return model_config
    

    def _load_tokenizer(self, cfg):
        model_config = self.load_model_config(cfg)
        tokenizer_kwargs = {}
        use_fast = True  # this is the default

        tokenizer_cls = AutoTokenizer
        if cfg.tokenizer_type:
            tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

        tokenizer_config = cfg.base_model_config or cfg.base_model # tokenizer_config
        tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_config,
            trust_remote_code=cfg.trust_remote_code or False,
            use_fast=use_fast,
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
            tokenizer.pad_token = "</s>"

        if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Mistral's official FA implementation requires left padding
        if cfg.is_mistral_derived_model and cfg.flash_attention and not cfg.sample_packing:
            tokenizer.padding_side = "left"

        additional_special_tokens = None
        if cfg.special_tokens:
            print(cfg.special_tokens)
            special_tokens = cfg.special_tokens
            additional_special_tokens = special_tokens.pop(
                "additional_special_tokens", None
            )
            lora_modules_to_save = self.get_linear_embedding_layers(model_config.model_type)
            for k, val in special_tokens.items():
                # check if new special token is not already in tokenizer and
                # is adapter training to make sure lora_modules_to_save is set
                # pylint: disable=too-many-boolean-expressions
                if (
                    (getattr(tokenizer, k) is None or getattr(tokenizer, k) != val)
                    and (len(tokenizer.encode(val, add_special_tokens=False)) > 2)
                    and cfg.adapter
                    and (
                        not cfg.lora_modules_to_save
                        or not all(
                            x in cfg.lora_modules_to_save for x in lora_modules_to_save
                        )
                    )
                ):
                    
                    lora_modules_to_save = ", ".join(
                        [f"`{x}`" for x in lora_modules_to_save]
                    )
                    raise ValueError(
                        f"Please set lora_modules_to_save to [{lora_modules_to_save}] when using an adapter and changing the special tokens."
                    )
                tokenizer.add_special_tokens(
                    {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
                )

            bos_or_eos_in_special_tokens = (
                "bos_token" in cfg.special_tokens and "eos_token" in cfg.special_tokens
            )
            if (
                tokenizer.__class__.__name__
                in (
                    "LlamaTokenizerFast",
                    "CodeLlamaTokenizerFast",
                )
                and bos_or_eos_in_special_tokens
            ):
                tokenizer.update_post_processor()

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
    

    def _load_model(
        self,           
        cfg,
        tokenizer: PreTrainedTokenizerBase,
        reference_model: bool = False,
    ) -> Tuple[PreTrainedModel, Optional[PeftConfig]]:
        """
        Load a model for a given configuration and tokenizer.
        """
        base_model = cfg.base_model
        model_type = cfg.model_type
        model_config = self.load_model_config(cfg)

        # TODO refactor as a kwarg
        load_in_8bit = cfg.load_in_8bit


        if cfg.sample_packing and cfg.s2_attention:
            raise ValueError(
                "Received `sample_packing=true` and `s2_attention=true`; however, \
            shifted-sparse attention does not currently support sample packing."
            )

        model_kwargs: Dict[str, Any] = {}

        # if cfg.model_kwargs:
        #     for key, val in cfg.model_kwargs.items():
        #         model_kwargs[key] = val

        max_memory = cfg.max_memory
        device_map = cfg.device_map
        
        if cfg.gpu_memory_limit:
            gpu_memory_limit = (
                str(cfg.gpu_memory_limit) + "GiB"
                if isinstance(cfg.gpu_memory_limit, int)
                else cfg.gpu_memory_limit
            )

            max_memory = {}
            for i in range(torch.cuda.device_count()):
                max_memory[i] = gpu_memory_limit

            max_memory["cpu"] = "256GiB"  # something sufficiently large to fit anything

        if max_memory is not None:
            # Based on https://github.com/togethercomputer/OpenChatKit/blob/main/inference/bot.py
            from accelerate import infer_auto_device_map, init_empty_weights

            with init_empty_weights():
                model_canvas = AutoModelForCausalLM.from_config(model_config)
            model_canvas.tie_weights()
            device_map = infer_auto_device_map(
                model_canvas,
                max_memory=max_memory,
                dtype=cfg.torch_dtype,
            )
            # We can discard max_memory now as we have a device map set up for us
            max_memory = None

        model_kwargs["device_map"] = device_map
        model_kwargs["torch_dtype"] = cfg.torch_dtype

        if torch.backends.mps.is_available():
            model_kwargs["device_map"] = "mps:0"

        if cfg.adapter == "qlora" and cfg.load_in_4bit:
            bnb_config = {
                "load_in_4bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False,
                "bnb_4bit_compute_dtype": cfg.torch_dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }

            if cfg.bnb_config_kwargs:
                bnb_config.update(cfg.bnb_config_kwargs)

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )


        if cfg.load_in_8bit and cfg.adapter is not None:
            model_kwargs["load_in_8bit"] = True
        if cfg.load_in_4bit and cfg.adapter is not None:
            model_kwargs["load_in_4bit"] = True

        if "quantization_config" in model_kwargs or cfg.gptq:
            if "load_in_8bit" in model_kwargs:
                del model_kwargs["load_in_8bit"]
            if "load_in_4bit" in model_kwargs:
                del model_kwargs["load_in_4bit"]
                
        
        # sample packing uses custom FA2 patch
        if cfg.flash_attention:
            if not cfg.sample_packing:
                if cfg.s2_attention:
                    pass
                # most other models support flash attention, we can define exceptions as they come up
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_config._attn_implementation = (  # pylint: disable=protected-access
                    "flash_attention_2"
                )
            else:
                if model_config.model_type in ["mixtral", "qwen2", "falcon", "phi"]:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    model_config._attn_implementation = (  # pylint: disable=protected-access
                        "flash_attention_2"
                    )
                else:
                    model_kwargs["attn_implementation"] = "eager"
                    model_config._attn_implementation = (  # pylint: disable=protected-access
                        "eager"
                    )
        elif cfg.sdp_attention:
            model_kwargs["attn_implementation"] = "sdpa"
            model_config._attn_implementation = "sdpa"  # pylint: disable=protected-access
        elif cfg.eager_attention:
            model_kwargs["attn_implementation"] = "eager"
            model_config._attn_implementation = "eager"  # pylint: disable=protected-access

        try:
            if (
                model_config.model_type == "llama"
                and not cfg.trust_remote_code
            ):
                from transformers import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    config=model_config,
                    **model_kwargs,
                )

            elif model_type and not cfg.trust_remote_code:
                model = getattr(transformers, model_type).from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
            else:
                # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
                # when training starts
                if (
                    hasattr(model_config, "max_seq_len")
                    and model_config.max_seq_len
                    and cfg.sequence_len > model_config.max_seq_len
                ):
                    model_config.max_seq_len = cfg.sequence_len
                elif (
                    hasattr(model_config, "max_sequence_length")
                    and model_config.max_sequence_length
                    and cfg.sequence_len > model_config.max_sequence_length
                ):
                    model_config.max_sequence_length = cfg.sequence_len
                if cfg.gptq:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        config=model_config,
                        trust_remote_code=cfg.trust_remote_code or False,
                        **model_kwargs,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        config=model_config,
                        trust_remote_code=cfg.trust_remote_code or False,
                        **model_kwargs,
                    )
        except Exception as err:  # pylint: disable=broad-exception-caught
            raise err

        if isinstance(model, (PeftModel, PeftModelForCausalLM)):
            model = model.merge_and_unload()

        embeddings_len = len(tokenizer)
        
        if (
            hasattr(model, "get_input_embeddings")
            and model.get_input_embeddings().num_embeddings < embeddings_len
        ):
            model.resize_token_embeddings(embeddings_len)
        else:
            model.tie_weights()

        if (
            hasattr(model, "config")
            and hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings
            and cfg.sequence_len > model.config.max_position_embeddings
        ):
            model.config.max_position_embeddings = cfg.sequence_len

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

        if hasattr(model, "device") and model.device.type in ("cuda", "mps"):
            log_gpu_memory_usage(logger, "after model load", model.device)

        # make sure these are fp32 per Ramesh et al. (2021)
        embedding_modules = self.get_linear_embedding_layers(cfg.model_config_type)
        logger.info(f"embedding_modules: {embedding_modules}")
        if not cfg.fsdp:
            # FSDP doesn't like mixed Float and BFloat16
            for name, module in model.named_modules():
                if "norm" in name or name.endswith(".gate"):
                    module.to(torch.float32)
                if model_config.model_type == "btlm":
                    # don't upcast lm_head for btlm
                    continue
                if any(m in name for m in embedding_modules):
                    if hasattr(module, "weight"):
                        module.to(torch.float32)

        needs_fa2_dtype = cfg.adapter or cfg.fsdp
        skip_prepare_model_for_kbit_training = False
        
        if cfg.adapter in ["lora", "qlora"]:
            if cfg.gradient_checkpointing:
                logger.info("if cfg.gradient_checkpointing:")
                model.gradient_checkpointing_enable()
            if (
                cfg.load_in_8bit or cfg.load_in_4bit
            ) and not skip_prepare_model_for_kbit_training:
                logger.info("converting PEFT model w/ prepare_model_for_kbit_training: in 8it: {cfg.load_in_8bit}, in 4it: {cfg.load_in_4bit}")
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=cfg.gradient_checkpointing
                )
            needs_fa2_dtype = True

        # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
        # convert them back to fp16/bf16 for flash-attn compatibility.
        if needs_fa2_dtype or cfg.flash_attention:
            for name, module in model.named_modules():
                if "norm" in name:
                    module.to(cfg.torch_dtype)
                if any(m in name for m in embedding_modules):
                    if hasattr(module, "weight"):
                        module.to(cfg.torch_dtype)

        lora_config = None
        if not reference_model or cfg.lora_model_dir:
            # if we're not loading the reference model, then we're loading the model for training
            # then the dpo trainer doesn't want the peft model loaded over it, it just wants the lora/peft config
            model, lora_config = self.load_adapter(model, cfg, cfg.adapter)

        if torch.cuda.device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
            logger.info(f"if torch.cuda.device_count() > 1 and int(os.getenv('WORLD_SIZE', '1')) == 1: {int(os.getenv('WORLD_SIZE', '1'))}")
            setattr(model, "is_parallelizable", True)
            setattr(model, "model_parallel", True)

        requires_grad = []
        for name, param in model.named_parameters(recurse=True):
            if param.requires_grad:
                # LOG.info(f"if param.requires_grad: {name}: {param.requires_grad}")
                requires_grad.append(f"{name}: {param.requires_grad}")
        if len(requires_grad) == 0:
            logger.warning("if len(requires_grad) == 0:")
        if hasattr(model, "config"):
            logger.info(f"if hasattr(model, 'config'): {model.config}")
            model.config.use_cache = False

        if cfg.adapter is not None:
            log_gpu_memory_usage(logger, "after adapters", model.device)

        # TODO resume_from_checkpoint handling
        return model, lora_config
    

    # TODO : 여기서부터 코드 다시 분석하기. 
    def load_adapter(self, model, cfg, adapter):
        # type: (PreTrainedModel, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

        if adapter is None:
            return model, None
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if adapter in ["lora", "qlora"]:
            return self.load_lora(model, cfg)
        if adapter == "llama-adapter":
            return self.load_llama_adapter(model, cfg)

        raise NotImplementedError(f"{adapter} peft adapter not available")


    def load_llama_adapter(model, cfg):
        # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
        from peft import AdaptionPromptConfig, get_peft_model

        peft_config = AdaptionPromptConfig(
            adapter_layers=cfg.peft_adapter.layers,  # layers (L)
            adapter_len=cfg.peft_adapter.len,  # prompt length (K)
            task_type="CAUSAL_LM",
        )

        if cfg.lora_model_dir:
            logger.info(f"cfg.lora_model_dir : {cfg.lora_model_dir}")
            model = PeftModel.from_pretrained(
                model,
                cfg.lora_model_dir,
                torch_dtype=torch.float16,
            )
        else:
            logger.info(f"model = get_peft_model(model, peft_config)")
            model = get_peft_model(model, peft_config)

        # model.print_trainable_parameters()

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


    def load_lora(self, model, cfg):
        logger.debug("Loading pretrained PEFT - LoRA")

        from peft import LoraConfig, get_peft_model

        lora_target_modules = list(cfg.lora_target_modules or [])

        if cfg.lora_target_linear:
            linear_names = self.find_all_linear_names(model)
            lora_target_modules = list(set(lora_target_modules + linear_names))

        lora_config_kwargs = {}
        # loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=lora_target_modules,
            layers_to_transform=cfg.peft_layers_to_transform,
            lora_dropout=cfg.lora_dropout,
            fan_in_fan_out=cfg.lora_fan_in_fan_out,
            modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
            bias="none",
            task_type="CAUSAL_LM",
            **lora_config_kwargs,
        )

        if cfg.lora_model_dir:
            model_kwargs: Any = {}
            # if cfg.lora_on_cpu:
            #     LOG.info(f"if cfg.lora_on_cpu:")
            #     model_kwargs["max_memory"] = {"cpu": "256GiB"}
            #     model_kwargs["device_map"] = {"": "cpu"}
            model = PeftModel.from_pretrained(
                model,
                cfg.lora_model_dir,
                is_trainable=False,
                **model_kwargs,
            )
        else:
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        return model, lora_config
    
     