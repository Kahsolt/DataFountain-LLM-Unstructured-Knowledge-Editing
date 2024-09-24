#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/24

# 跑 EasyEditor.LoRA 模型编辑

import sys
sys.path.append('repo/EasyEdit')
from easyeditor.editors import BaseEditor
from easyeditor.models.lora import LoRAHyperParams

import random
import torch
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from time import time
from pprint import pprint
from utils import load_rank_B1_data

random.seed(114514)

hparams = LoRAHyperParams.from_hparams('configs/lora/internlm.yaml')
editor = BaseEditor.from_hparams(hparams)
tokenizer: AutoTokenizer = editor.tok
edit_data, _ = load_rank_B1_data()

for i_iter in range(1, 3):
  ''' Hparam '''
  print('>> [Round]', i_iter)

  ''' Data '''
  random.shuffle(edit_data)
  prompts = [it['question'] for it in edit_data]
  target_new = [it['answer'] for it in edit_data]
  if not 'with sys_prompt':
    prompts_with_system_prompt = [
      tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a QA expert attending a knowledge competition. Answer the following question in English: '},
        {'role': 'user', 'content': p},
      ], tokenize=False, add_generation_prompt=False)
      for p in prompts
    ]

  ''' Edit! '''
  ts_start = time()
  results, model, extra_data = editor.batch_edit(
    prompts=prompts,
    #prompts=prompts_with_system_prompt,
    target_new=target_new,
    sequential_edit=True,
    verbose=False,
  )
  ts_end = time()
  print('TIME_COST:', ts_end - ts_start)

  ''' Peep '''
  model: PreTrainedModel
  for Q, T in zip(prompts, target_new):
    input_ids = tokenizer.encode(Q, return_tensors='pt').to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=512)[0]
    A = tokenizer.decode(output_ids[len(input_ids[0]):]).strip()
    print('Q:', Q)
    print('A:', A)
    print('T:', T)

  ''' Next Round '''
  hparams.lr /= 2
  hparams.num_steps //= 2

print('>> save ckpt file')
torch.save(model.state_dict(), 'out/internlm2_5-1_8b-chat.ckpt')
