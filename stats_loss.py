#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 统计训练集主 QA 对的 loss

from time import time
from pprint import pprint

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from tqdm import tqdm

from utils import *

model_path = 'internlm/internlm2-chat-1_8b'
max_new_tokens = 256

ts_start = time()

device = 'cuda:0'
model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
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


# ~https://www.cnblogs.com/zhangxianrong/p/18251314
class CustomLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss_fn = nn.CrossEntropyLoss(reduction='none')
  def forward(self, outputs, labels):
    logits = outputs.logits
    batch_size, seq_len, vocab_size = logits.size()
    # 获取每个标签序列的实际长度（去掉pad）
    label_lengths = (labels != tokenizer.pad_token_id).sum(dim=1)
    # 计算权重
    weights = torch.zeros_like(labels, dtype=torch.float)
    for i, length in enumerate(label_lengths):
      length = length.item()
      weights[i, :length] = torch.arange(length + 1, 1, -1, dtype=torch.float)
    # 计算损失
    loss = self.loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    loss = loss.view(batch_size, seq_len)
    # 应用权重
    weighted_loss = loss * weights
    # 计算平均损失
    weighted_loss = weighted_loss.sum() / weights.sum()
    return weighted_loss


criterion = CustomLoss()
def get_loss(que:str, ans:str) -> str:
  messages = [
    {"role": "system", "content": "You are a helpful assistant doing reading comprehension tasks."},
    {"role": "user", "content": que},
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )
  target = ans + '<|im_end|>'
  model_inputs = tokenizer([text],   return_tensors="pt", padding=True, truncation=True).to(device)
  model_truths = tokenizer([target], return_tensors="pt", padding=True, truncation=True).to(device)
  outputs = model(model_inputs.input_ids)
  breakpoint()
  loss = criterion(outputs, model_truths.input_ids)
  return loss.item()


edit_data, orth_data = load_rank_A_data()
for it in tqdm(edit_data):
  it['loss'] = get_loss(it['question'], it['answer'])
  pprint(it['id'], it['loss'])
  print()

for it in tqdm(edit_data):
  print(it['id'], it['loss'])

ts_stop = time()
print('>> time cost:', ts_stop - ts_start)
