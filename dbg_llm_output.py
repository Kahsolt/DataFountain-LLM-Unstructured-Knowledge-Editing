#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 调试用脚本: 想办法让模型回答简短、类似于完形填空，而不是输出一句完整的话

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel

from utils import *

model_path = 'internlm/internlm2_5-1_8b-chat'
print('>> model_path:', model_path)


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
  generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=128)
  generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
  resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return resp


print('\n\n')
edit_data, orth_data = load_rank_A_data()
for idx, it in enumerate(edit_data):
  if idx >= 10: break    # 只有前 10 个样本有真值 sub_answer
  print(f'[Example {idx}]')

  question     = it['question']
  answer       = it['answer']
  sub_question = it['sub_question']
  sub_answer   = it.get('sub_answer')

  sub_p = []
  for i, sub_q in enumerate(sub_question):
    print(f'[Sub-Question {i}]')
    print('ref_ans:', sub_answer[i] if sub_answer else None)

    # TODO: 你能修改的大概就是这个对话上下文里的提示，使得模型输出简短答案
    messages = [
      {"role": "system", "content": "You are doing a reading comprehension task. Make answers very very short. Do not write a complete sentence. Answer in limited words."},
      {"role": "user", "content": question},
      {"role": "assistant", "content": answer},
      {"role": "user", "content": "Extract answer for the following question based on the above context. Do not write a sentence, you can only extract the answer words from above context!"},
      {"role": "user", "content": sub_q},
    ]
    resp = infer_model(messages)
    if resp[-1] not in ["'", '"'] and not resp.endswith('.'): resp += '.'
    sub_p.append(resp)
    print('llm_ans:', resp)
    print('-' * 72)

  it['sub_prediction'] = sub_p
  input('>> Press <Enter> to infer the next sample...')
  print('=' * 72)
  print('\n')
