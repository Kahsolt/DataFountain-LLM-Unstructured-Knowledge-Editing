#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 推理无关问题集

from time import time
from pprint import pprint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from tqdm import tqdm

from utils import *


def infer_llm(prompt:str, model:PreTrainedModel, tokenizer:AutoTokenizer, maxlen:int=256):
  messages = [
    {"role": "system", "content": "You are a helpful assistant. You are facing a knowledge contest. Please answer in English. Make answers very short. Do not explain the reason. Do not answer in a complete sentence. Answer in single words. Only point out the direct answer. Do not repeat yourself."},
    {"role": "user", "content": prompt},
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )
  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
  generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=maxlen)
  generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
  respo = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return respo


def run():
  model_path = 'Qwen/Qwen1.5-7b-Chat'
  #model_path = 'internlm/internlm2_5-1_8b-chat'
  max_new_tokens = 256

  device = 'cuda:0'
  model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    #torch_dtype=torch.bfloat16, 
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    device_map=device,
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
  )

  _, orth_data = load_rank_A_data()
  for it in tqdm(orth_data):
    it['prediction'] = infer_llm(it['question'], model, tokenizer, maxlen=max_new_tokens)
    pprint(it)
    print()

  save_infer_orth_data(orth_data)


if __name__ == '__main__':
  ts_start = time()
  run()
  ts_stop = time()
  print('>> time cost:', ts_stop - ts_start)    # 487.9688367843628
