#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

# 测试脚本: 跑LLM测原始权重在给定数据集上的输出

from time import time
from pprint import pprint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from tqdm import tqdm

from utils import *

# https://huggingface.co/Qwen/Qwen1.5-7B-Chat
model_path = 'Qwen/Qwen1.5-7B-Chat'
max_new_tokens = 512

ts_start = time()

device = 'cuda:0'
model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
  model_path,
  #torch_dtype=torch.bfloat16, 
  bnb_4bit_quant_type='nf4',
  bnb_4bit_compute_dtype=torch.float16,
  load_in_4bit=True,
  low_cpu_mem_usage=True,
  device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def query_model(prompt:str, maxlen:int=max_new_tokens) -> str:
  messages = [
    {"role": "system", "content": "You are a helpful assistant handling reading comprehension tasks."},
    {"role": "user", "content": prompt},
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )
  #print('>> processed text: ', text)
  model_inputs = tokenizer([text], return_tensors="pt").to(device)
  generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=maxlen)
  generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return response


edit_data, orth_data = load_rank_A_data()
for it in tqdm(edit_data):
  it['original_prediction'] = query_model(it['question'],       maxlen=256)
  it['para_prediction']     = query_model(it['para_question'],  maxlen=256)
  it['para_prediction1']    = query_model(it['para_question1'], maxlen=256)
  it['para_prediction2']    = query_model(it['para_question2'], maxlen=256)
  it['sub_prediction']      = [query_model(sub_q, maxlen=32) for sub_q in it['sub_question']]
  pprint(it)
  print()

save_infer_data(edit_data)

ts_stop = time()
print('>> time cost:', ts_stop - ts_start)
