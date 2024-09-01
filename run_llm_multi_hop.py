#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 测试脚本: 测试 multi-hop 问答，即把 question-answer 追加在 sub_question 之前作为上下文 
# NOTE: 这个做法很有效！🎉

from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from tqdm import tqdm

from utils import *

mdoel_path = 'Qwen/Qwen1.5-0.5B-Chat'
mdoel_path = 'Qwen/Qwen1.5-1.8B-Chat'
mdoel_path = 'Qwen/Qwen1.5-4B-Chat'
mdoel_path = 'Qwen/Qwen1.5-7B-Chat'             # <- tested good 
mdoel_path = 'Qwen/Qwen2-0.5B-Instruct'
mdoel_path = 'Qwen/Qwen2-1.5B-Instruct'
mdoel_path = 'Qwen/Qwen2-7B-Instruct'
mdoel_path = 'internlm/internlm2-chat-1_8b'
mdoel_path = 'internlm/internlm2-chat-7b'
mdoel_path = 'internlm/internlm2_5-1_8b-chat'   # <- tested good 
mdoel_path = 'internlm/internlm2_5-7b-chat'

model_path = 'internlm/internlm2_5-1_8b-chat'
print('>> model_path:', model_path)

SYS_PROMPT = "You are a helpful assistant doing reading comprehension tasks, make responses brief."
MAX_NEW_TOKENS = 128

ts_start = time()

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


def infer_model(messages:List[Dict]) -> str:
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  #print('>> processed text: ', text)
  model_inputs = tokenizer([text], return_tensors="pt").to(device)
  generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS)
  generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
  resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return resp


print('\n\n')
edit_data, orth_data = load_rank_A_data()
for it in tqdm(edit_data):
  question     = it['question']
  answer       = it['answer']
  sub_question = it['sub_question']
  sub_answer   = it.get('sub_answer')

  # original q
  print('[Question]')
  messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": question},
  ]
  it['original_prediction'] = infer_model(messages)

  # para q
  print(f'[Para-Question]')
  messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": it['para_question']},
  ]
  it['para_prediction'] = infer_model(messages)
  messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": it['para_question1']},
  ]
  it['para_prediction1'] = infer_model(messages)
  messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": it['para_question2']},
  ]
  it['para_prediction2'] = infer_model(messages)

  # sub q
  sub_p = []
  for i, sub_q in enumerate(sub_question):
    print(f'[Sub-Question {i}]')
    print('ref_ans:', sub_answer[i] if sub_answer else None)

    messages = [
      {"role": "system", "content": "You are doing a reading comprehension task. Remember to make answers very short. Do not write a complete sentence. Answer in limited words."},
      {"role": "user", "content": question},
      {"role": "assistant", "content": answer},
      {"role": "user", "content": "Extract answer for the following question based on the above context. Do not write a sentence, you can only extract the answer words from above context!"},
      {"role": "user", "content": sub_q},
    ]
    resp = infer_model(messages)
    sub_p.append(resp)
    print('llm_ans:', resp)
    print('=' * 72)

  it['sub_prediction'] = sub_p
  print('\n\n')

# NOTE: 临时做一版方案: prediction用原权重推理, sub_prediction用真值answer结合提示词上下文来实现
save_infer_data(edit_data)

ts_stop = time()
print('>> time cost:', ts_stop - ts_start)    # 2826.5771272182465
