import logging
import math
import os
from typing import Any, Dict, Optional, Tuple, Union, Generator, AsyncGenerator  # noqa: F401
import time

from ragcar.models.base import Sllmbase
from threading import Thread

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("sllm")
import transformers
from transformers import GenerationConfig, TextStreamer, TextIteratorStreamer


class SLLMCompletion(Sllmbase):
    def __init__(
        self, 
        model_n: str,
        lora_model_dir: Optional[str] = None,
        # deepspeed: bool = False,
        adapter: str = None,  #lora, qlora
        stream: bool = True,
        formatting: bool = False,
        **kwargs,
    ):
        super().__init__(model_n)
        self.model = model_n

        self.model_config = self._load_model_config()
        self.tokenizer = self._load_tokenizer(model_n, self.model_config)    
        # LOG.info(f"model_n: {model_n}, lora_path: {lora_model_dir}, deepspeed: {deepspeed}, adapter: {adapter}, formatting: {formatting}")
        LOG.info(f"model_n: {model_n}, lora_path: {lora_model_dir},  adapter: {adapter}, formatting: {formatting}")
        self.device, self.torch_dtype = self._select_device()

        # model_load
        self.model, _ = self._load_model(
                self.tokenizer,
                self.model_n,
                self.model_config,
                lora_model_dir,
                adapter,
                **kwargs,
            )
    
    def _calculate_tokens(self, prompt: Optional[list] = None, completion: Optional[str] = None):
        if prompt:
            fromatted_prompt = ["""{{"{0}": "{1}"}}""".format(item['role'], item['content']) for item in prompt]
            prompt_str = f"[{', '.join(fromatted_prompt)}]"
            return len(self.tokenizer(str(prompt_str))) - 1
        
        if completion:
            return len(self.tokenizer(completion))

    def _get_params(
        self, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        repetition_penalty: float=1.1, 
        **kwargs
    ) -> Dict[str, Union[int, float, bool]]:
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "bos_token_id":self.tokenizer.bos_token_id,
            "eos_token_id":self.tokenizer.eos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "repetition_penalty": repetition_penalty,
            "do_sample":True,
            "use_cache":True,
            "return_dict_in_generate":True,
            "output_attentions":False,
            "output_hidden_states":False,
            "output_scores":False,
        }
        
        return {k: v for k, v in params.items() if v is not None}


    def _output(self,batch, generation_kwargs): 
        LOG.info(f"generating .... generation_kwargs: {generation_kwargs}")
         
        generated = self.model.generate(
                   batch,  # 위치 인자로 입력 텐서 전달
                   generation_config=generation_kwargs # 나머지 키워드 인자 언패킹하여 전달
                )
        return  self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _stream_output(self, generation_kwargs, streamer):
        LOG.info(f"stream generating .... generation_kwargs: {generation_kwargs}")
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs) # model.generate 메서드를 비동기적으로 실행
        thread.start()

        all_text = ""
        
        for new_text in streamer:
            all_text += new_text
            print(all_text)
            yield all_text

     

    def create(
        self, 
        messages, 
        **kwargs
    ) -> Union[str, Dict[str, str], Generator[str, None, None]]:
        
        # start_time = time.time()
        LOG.info(f"messages: {messages}")
        default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        LOG.info(f"model: {self.model}")
        # model = self.model.to(self.device, self.torch_dtype)
        
        LOG.info(f"tokeinzing")
        batch = self.tokenizer(messages, return_tensors="pt", add_special_tokens=True) 
        self.model.eval()
        
        params = self._get_params(**kwargs)
        LOG.info(f"params: {params}")

        LOG.info(f"stream generating ....")
        streamer = TextIteratorStreamer(self.tokenizer)

        generation_kwargs = {
            "inputs": batch["input_ids"].to(self.device),
            "generation_config": GenerationConfig(**params),
            "streamer": streamer,
        }
        
        yield from self._stream_output(generation_kwargs, streamer)

    # def create(
    #     self, 
    #     messages, 
    #     **kwargs
    # ) -> Union[str, Dict[str, str], Generator[str, None, None]]:

        
    #     # start_time = time.time()
    #     LOG.info(f"messages: {messages}")
    #     default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    #     LOG.info(f"model: {self.model}")
    #     # model = self.model.to(self.device, self.torch_dtype)
        
    #     LOG.info(f"tokeinzing")
    #     batch = self.tokenizer(messages, return_tensors="pt", add_special_tokens=True) 
    #     self.model.eval()

         
    #     params = self._get_params(**kwargs)
    #     LOG.info(f"params: {params}")

    #     LOG.info(f"stream generating ....")
    #     streamer = TextIteratorStreamer(self.tokenizer)
    #     generation_kwargs = {
    #         "inputs": batch["input_ids"].to(self.device),
    #         "generation_config": GenerationConfig(**params),
    #         "streamer": streamer,
    #     }
    #     return self._stream_output(generation_kwargs,streamer )
    
        # if kwargs.get('stream'):
        #     LOG.info(f"stream generating ....")
        #     streamer = TextIteratorStreamer(self.tokenizer)
        #     generation_kwargs = {
        #         "inputs": batch["input_ids"].to(self.device),
        #         "generation_config": GenerationConfig(**params),
        #         "streamer": streamer,
        #     }
        #     return self._stream_output(generation_kwargs,streamer )
        # else:
        #     LOG.info(f"no stream generating ....")
        #     generation_kwargs = GenerationConfig(**params)
            
        #     # generation_kwargs = {
        #     #     "inputs": batch["input_ids"].to(self.device),
        #     #     # 다른 필요한 설정들을 직접 명시
        #     #     "max_length": 100,  # 예시 설정
        #     #     "num_beams": 5,  # 예시 설정
        #     #     "temperature": 1.0,  # 예시 설정
        #     #     "return_dict_in_generate": True,
        #     #     "output_scores": True,
        #     # }

        #     return self._output(batch["input_ids"].to(self.device), generation_kwargs)


        # response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # print(self.tokenizer.decode(generated["sequences"].cpu().tolist()[0]))
        


        # return  response