#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

import sys
sys.path.append('repo/EasyEdit')

from pprint import pprint
from easyeditor.editors import BaseEditor, SafetyEditor, ConceptEditor, PerEditor
from easyeditor.models import *
from easyeditor.models.emmet import *
from easyeditor.models.lora import *

# 依综述 arXiv:2401.01286 报告，SERAC > MEND > ROME ~ FT-L > AdaLoRA > MEMIT
# 依综述 2023.emnlp-main.632 报告
#  - 数据集 ZERE: T-Patcher > SERAC > MEMIT > ROME > MEND
#  - 数据集 CounterFact: SERAC > ROME > MEMIT > T-Patcher > MEND

#hparams = FTHyperParams    .from_hparams('repo/EasyEdit/hparams/FT/gpt2-xl')      # it works, fast
#hparams = LoRAHyperParams  .from_hparams('repo/EasyEdit/hparams/LoRA/gpt2-xl')    # it works, fast
hparams = MELOHyperParams  .from_hparams('repo/EasyEdit/hparams/MELO/gpt2-xl')    # it works, fast
#hparams = GraceHyperParams .from_hparams('repo/EasyEdit/hparams/Grace/gpt2-xl')   # it works, a bit slow
#hparams = KNHyperParams    .from_hparams('repo/EasyEdit/hparams/KN/gpt2-xl')      # it works, but extreme slow :(

#hparams = EMMETHyperParams .from_hparams('repo/EasyEdit/hparams/EMMET/gpt2-xl')   # KeyError: 'subject'
#hparams = MEMITHyperParams .from_hparams('repo/EasyEdit/hparams/MEMIT/gpt2-xl')   # KeyError: 'subject'
#hparams = R_ROMEHyperParams.from_hparams('repo/EasyEdit/hparams/R-ROME/gpt2-xl')  # KeyError: 'subject'
#hparams = ROMEHyperParams  .from_hparams('repo/EasyEdit/hparams/ROME/gpt2-xl')    # KeyError: 'subject'
#hparams = WISEHyperParams  .from_hparams('repo/EasyEdit/hparams/WISE/gpt2-xl')    # KeyError: 'loc_prompt'
#hparams = DINMHyperParams  .from_hparams('repo/EasyEdit/hparams/DINM/gpt2-xl')    # KeyError: 'general knowledge constraint'

#hparams = IKEHyperParams   .from_hparams('repo/EasyEdit/hparams/IKE/gpt2-xl')     # IKE need train_ds (For getting In-Context prompt)
#hparams = MENDHyperParams  .from_hparams('repo/EasyEdit/hparams/MEND/gpt2-xl')    # require pretraining an aux model
#hparams = SERACHparams     .from_hparams('repo/EasyEdit/hparams/SERAC/gpt2-xl')   # require pretraining an aux model
#hparams = PMETHyperParams  .from_hparams('repo/EasyEdit/hparams/PMET/gpt2-xl')    # no support gpt2-xl
#hparams = MALMENHyperParams.from_hparams('repo/EasyEdit/hparams/MALMEN/gpt2-xl')  # KeyError: 'MALMEN', not the same API usage

editor = BaseEditor.from_hparams(hparams)
#editor = SafetyEditor.from_hparams(hparams)
#editor = ConceptEditor.from_hparams(hparams)
#editor = PerEditor.from_hparams(hparams)   for IKE & MEND


# editing data
prompts = [
  'What university did Watts Humphrey attend?',
  'Which family does Ramalinaceae belong to',
  'What role does Denny Herzig play in football?'
]
ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender']
target_new = ['University of Michigan', 'Lamiinae', 'winger']
locality_inputs = {
  'neighborhood':{
    'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
    'ground_truth': ['piano', 'basketball', 'Finnish']
  },
  'distracting': {
    'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
    'ground_truth': ['piano', 'basketball', 'Finnish']
  }
}

# go!!!
results, edited_model, extra_data = editor.edit(
  prompts=prompts,
  prompts_with_systemPrompt=prompts,
  ground_truth=ground_truth,
  target_new=target_new,
  locality_inputs=locality_inputs,
  sequential_edit=False   # True
)

# eval
'''
rewrite_acc → Reliablilty
rephrase_acc → Generalization
locality → Locality
portablility → Portablility
'''
print('[results]')
pprint(results)
print('[edited_model]')
print(edited_model)
print('[extra_data]')
print(extra_data)

breakpoint()
