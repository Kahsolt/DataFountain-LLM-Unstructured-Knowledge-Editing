#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 测试脚本: 测试 multi-hop 问答，即把 question-answer 追加在 sub_question 之前作为上下文 
# NOTE: 这个做法很有效！🎉

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel

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

print('\n\n')
edit_data, orth_data = load_rank_A_data()
for it in edit_data:
  question     = it['question']
  answer       = it['answer']
  sub_question = it['sub_question']
  sub_answer   = it['sub_answer']

  for i, sub_q in enumerate(sub_question):
    print(f'[Sub-Question {i}]')
    print('ref_ans:', sub_answer[i])

    messages = [
      {"role": "system", "content": "You are a helpful assistant doing reading comprehension tasks."},
      {"role": "user", "content": question},
      {"role": "assistant", "content": answer},
      {"role": "user", "content": "Answer the following question based on the above context, make it very short and brief."},
      {"role": "user", "content": sub_q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #print('>> processed text: ', text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256)
    generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
    resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('llm_ans:', resp)

    print('=' * 72)
  print('\n\n')
  breakpoint()
