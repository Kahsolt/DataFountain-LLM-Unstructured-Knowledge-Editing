#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

import warnings ; warnings.filterwarnings(category=UserWarning, action='ignore')

import json

from typing import Tuple, List, Dict
from pathlib import Path

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
OUT_PATH = BASE_PATH / 'out'
RANK_A_DATA_FILE = DATA_PATH / 'final_test_data_a.json'
DEFAULT_OUT_FILE = OUT_PATH / 'submit.json'
DEFAULT_OUT_ORTH_FILE = OUT_PATH / 'submit_orth.json'

EditSample = Tuple[Dict]   # (id, question, para_question*, sub_question, sub_answer)
OrthSample = Tuple[Dict]   # {id, question}
EditDataset = List[EditSample]
OrthDataset = List[OrthSample]
Databank = Tuple[EditDataset, OrthDataset]

mean = lambda x: sum(x) / len(x) if x else 0.0

# LLM Leaderboard
# - https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
# - https://huggingface.co/collections/open-llm-leaderboard/llm-leaderboard-best-models-652d6c7965a4619fb5c27a03
# - https://www.datalearner.com/ai-models/leaderboard/datalearner-llm-leaderboard
PRETRAINED_MODELS = [
  # 国外模型
  'mistralai/Mistral-7B-Instruct-v0.1',
  'mistralai/Mistral-7B-Instruct-v0.2',
  'mistralai/Mistral-7B-Instruct-v0.3',

  # 国产模型
  'Qwen/Qwen1.5-0.5B-Chat',
  'Qwen/Qwen1.5-1.8B-Chat',
  'Qwen/Qwen1.5-4B-Chat',
  'Qwen/Qwen1.5-7B-Chat',
  'Qwen/Qwen2-0.5B-Instruct',
  'Qwen/Qwen2-1.5B-Instruct',
  'Qwen/Qwen2-7B-Instruct',

  'internlm/internlm2-chat-1_8b',
  'internlm/internlm2-chat-7b',
  'internlm/internlm2_5-1_8b-chat',
  'internlm/internlm2_5-7b-chat',

  'THUDM/chatglm-6b',
  'THUDM/chatglm2-6b',
  'THUDM/chatglm3-6b',
]


def load_rank_A_data() -> Databank:
  edit_data: EditDataset = []   # n_samples = 100
  orth_data: OrthDataset = []   # n_samples = 400
  with open(RANK_A_DATA_FILE, encoding='utf-8') as fh:
    records = json.load(fh)
    for rec in records:
      id: str = rec['id']
      if id.startswith('edit'):
        edit_data.append(rec)
      elif id.startswith('unrelated'):
        orth_data.append(rec)
      else:
        raise ValueError(f'unknown record struct: {rec}')
  return edit_data, orth_data


def save_infer_data(samples:EditDataset, fp:Path=None):
  fp = fp or DEFAULT_OUT_FILE
  fp.parent.mkdir(exist_ok=True)

  assert isinstance(samples, list)
  for it in samples:
    assert isinstance(it, dict)
    if 'answer'     in it: del it['answer']
    if 'sub_answer' in it: del it['sub_answer']
    assert set(it.keys()) == {
      'id',
      'question',
      'para_question',
      'para_question1',
      'para_question2',
      'sub_question',
      'original_prediction',
      'para_prediction',
      'para_prediction1',
      'para_prediction2',
      'sub_prediction',
    }

  print(f'>> write json: {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(samples, fh, indent=2, ensure_ascii=False)


def save_infer_orth_data(samples:OrthDataset, fp:Path=None):
  fp = fp or DEFAULT_OUT_ORTH_FILE
  fp.parent.mkdir(exist_ok=True)

  assert isinstance(samples, list)
  for it in samples:
    assert isinstance(it, dict)
    assert set(it.keys()) == {
      'id',
      'question',
      'prediction',
    }

  print(f'>> write json: {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(samples, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  edit_data, orth_data = load_rank_A_data()
  print('len(edit_data):', len(edit_data))
  print('len(orth_data):', len(orth_data))

  for it in edit_data:
    it['original_prediction'] = it['answer']
    it['para_prediction']     = it['answer']
    it['para_prediction1']    = it['answer']
    it['para_prediction2']    = it['answer']
    it['sub_prediction']      = it.get('sub_answer', it['sub_question'])

  save_infer_data(edit_data)
