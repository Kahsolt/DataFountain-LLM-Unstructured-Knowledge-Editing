#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 统计数据集真值的 token 长度

from transformers import AutoTokenizer
from tqdm import tqdm

from utils import *

model_path = 'Qwen/Qwen1.5-7B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path)


len_ans = []
len_sub_ans = []
edit_data, orth_data = load_rank_A_data()
for it in tqdm(edit_data):
  len_ans.append(len(tokenizer.encode(it['answer'])))
  if 'sub_answer' in it:
    len_sub_ans.extend([len(tokenizer.encode(e)) for e in it['sub_answer']])
print('len_ans:', len_ans)
print('len_sub_ans:', len_sub_ans)


# min(len_ans): 61
# min(len_sub_ans): 2
print('min(len_ans):', min(len_ans))
print('min(len_sub_ans):', min(len_sub_ans))
# max(len_ans): 188
# max(len_sub_ans): 25
print('max(len_ans):', max(len_ans))
print('max(len_sub_ans):', max(len_sub_ans))
# mean(len_ans): 123.47
# mean(len_sub_ans): 6.976190476190476
print('mean(len_ans):', mean(len_ans))
print('mean(len_sub_ans):', mean(len_sub_ans))
