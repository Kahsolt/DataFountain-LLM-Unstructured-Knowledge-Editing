import sys
sys.path.insert(0, '..')
from utils import load_rank_A_data, save_infer_data

import os
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import numpy as np
from tqdm import tqdm

from config import Config
import nethook

def set_seed(seed=2024):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_llama_sys_que(sys, que):
    return f'<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{que} [/INST]'

def get_llama_with_answer(que, ans):
    return f"""<s>[INST] {que} [/INST] {ans} </s>"""

def get_llama_without_answer(que):
    return f"""<s>[INST] {que} [/INST]"""

def get_llama_without_answer_cot(que):
    return f"""<s>[INST] Please provide a multi-hop explanation for the next question : {que} [/INST] """

def get_qwen_without_answer(que):
    #SYSTEM_PROMPT_QWEN = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    #SYSTEM_PROMPT_QWEN = '<|im_start|>system\nYou are a helpful assistant doing reading comprehension tasks.<|im_end|>\n'
    SYSTEM_PROMPT_QWEN = ''
    return f"""{SYSTEM_PROMPT_QWEN}<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_list_llama_without_answer(que, cot=False):
    if cot == False:
        #L = [get_llama_sys_que(SYSTEM_PROMOT,line) for line in que]
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L

def get_list_qwen_without_answer(que, cot=False):
    if cot == False:
        #L = [get_llama_sys_que(SYSTEM_PROMOT,line) for line in que]
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L


def get_llama_causal_mask(input_tensor, attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[: mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask, position_ids, cache_position

def get_qwen_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (input_tensor.shape[0], input_tensor.shape[1]),
        input_tensor,
        0,
    )
    return attention_mask, position_ids

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': 0.0},
    ]
    return optimizer_parameters


@torch.no_grad
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    config: Config,
    layer: int,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f"cuda:{config.device}")
    idxs = [i.sum() - 1 for i in input_ids['attention_mask']]

    with nethook.Trace(
        module=model,
        layer=config.layer_module_tmp.format(layer),
        retain_input=True,
        retain_output=True,
        detach=True,
        clone=True,
    ) as tr:
        _ = model(**input_ids)
        #layer_in_ks = tr.input #(bs:seq:h_dim)
        zs_out = tr.output # (bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list = []
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i, idxs[i]])
    zs_out = torch.stack(zs_out_list, dim=0)
    return zs_out, idxs

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    config:Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    #print("Computing right vector (v)")

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{config.lm_head_module}.weight").T,
        nethook.get_module(model, config.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{config.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # Tokenize target into list of int token IDs
    device = f"cuda:{config.device}"
    target_ids = tok(data["answer"], return_tensors="pt").to(device)["input_ids"][0]  
    if target_ids[0] in [tok.bos_token_id, tok.unk_token_id]:
        target_ids = target_ids[1:]

    input_tok = tok([data["question"]], return_tensors="pt", padding=True).to(device)
    input_ids = torch.cat([input_tok['input_ids'],torch.unsqueeze(target_ids[:-1], dim=0)], dim=1)

    rewriting_targets = torch.tensor(-100, device=device).repeat(1, len(input_ids[0]))
    ex_len = len(input_ids[0])
    rewriting_targets[0, ex_len - len(target_ids) : ex_len] = target_ids
    lookup_idxs = [ex_len - len(target_ids)]
    loss_layer = max(config.v_loss_layer, layer)

    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError

    target_init = None
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init  
        if cur_layer == config.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=config.v_lr)
    nethook.set_requires_grad(False, model)  

    # Execute optimization
    for it in range(config.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                config.layer_module_tmp.format(loss_layer),
                config.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(input_ids).logits

        # Compute loss on rewriting targets
        output = tr[config.layer_module_tmp.format(loss_layer)].output[0]  
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        weight_decay = config.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + weight_decay.to(nll_loss.device)
        print(f"[Opt-V step {it}/{config.v_num_grad_steps}] nll_loss={nll_loss.item():.3f}, weight_decay={weight_decay.item():.3f}, avg_prob={torch.exp(-nll_loss_each).mean().item()}")

        if loss < 5e-4: break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = config.clamp_norm_factor * target_init.norm()
        delta_norm = delta.norm()
        if delta_norm > max_norm:
            with torch.no_grad():
                delta.data = delta * max_norm / delta_norm

    target = target_init + delta  
    print(f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}")
    return target


def execute_batch_unke(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    config: Config,
    batch_data: list,
    ex_data: list
):
    # NOTE: these are the parameters to optimize, so save the originals first!!
    '''
    [
        'model.layers.7.self_attn.q_proj.weight', 
        'model.layers.7.self_attn.q_proj.bias', 
        'model.layers.7.self_attn.k_proj.weight', 
        'model.layers.7.self_attn.k_proj.bias', 
        'model.layers.7.self_attn.v_proj.weight', 
        'model.layers.7.self_attn.v_proj.bias', 
        'model.layers.7.self_attn.o_proj.weight', 
        'model.layers.7.mlp.gate_proj.weight', 
        'model.layers.7.mlp.up_proj.weight', 
        'model.layers.7.mlp.down_proj.weight', 
        'model.layers.7.input_layernorm.weight', 
        'model.layers.7.post_attention_layernorm.weight',
    ]
    '''
    preserve_params: List[str] = []
    for name, params in model.named_parameters():
        splitted_name = name.split('.')
        if 'self_attn' not in splitted_name: continue
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in config.layers:
                preserve_params.append(name)
    print('preserve_params:', preserve_params)
    weights = {param: nethook.get_parameter(model, param) for param in preserve_params}
    weights_copy = {k: v.detach().cpu().clone() for k, v in weights.items()}

    z_layer = config.layers[-1]
    z_list = []
    for data in batch_data:
        cur_z = compute_z(model, tok, data, z_layer, config)
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)  # [bs=1, h_dim=1096]
    print('zs.shape:', zs.shape)

    batch_question = [i['question'] for i in batch_data]
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    for i, layer in enumerate(config.layers):
        print(f"LAYER {layer}\n")
        mod_name = config.layer_module_tmp.format(layer)

        # target p_i
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=mod_name,
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input    # (bs:seq:h_dim)
                layer_out_ks = tr.output  # (bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks     # [B=1, L=15, D=4096]

        cur_zs, idxs = compute_ks(model, tok, batch_question, config, z_layer)
        targets = zs - cur_zs  # [B=1, D=4096]
        resid = targets / (len(config.layers) - i)  # Distribute residual across layers (?)

        # irrelevant q_i
        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=mod_name,
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out     # [B=20, L=235, D=4096]

        mod: nn.Module = nethook.get_module(model, mod_name)
        for n, m in mod.named_parameters():
            if 'q_proj' in n or 'v_proj' in n:
                m.requires_grad = True
            else:
                m.requires_grad = False
        print('param_cnt:', sum(p.numel() for p in mod.parameters() if p.requires_grad))
        params = get_optimizer_params(mod, config.lr)
        optimizer = optim.AdamW(params, lr=config.lr, eps=1e-8, betas=(0.9,0.999))

        for i in range(len(idxs)):
            layer_out_ks[i, idxs[i]] += resid[i]

        # llama2
        if config.model_name == 'LLama2-7B-Chat':
            ex_causal_mask, ex_position_ids, ex_cache_position = get_llama_causal_mask(stat_in, ex_tok['attention_mask'])
            input_causal_mask, input_position_ids, input_cache_position = get_llama_causal_mask(layer_in_ks, contexts_tok['attention_mask'])
        elif config.model_name == 'Qwen1.5-7B-Chat':
            ex_causal_mask, ex_position_ids = get_qwen_causal_mask(stat_in, ex_tok['attention_mask'])
            input_causal_mask, input_position_ids = get_qwen_causal_mask(layer_in_ks, contexts_tok['attention_mask'])

        for step in range(config.optim_num_step):
            optimizer.zero_grad()
            if config.model_name == 'LLama2-7B-Chat':
                loss_keep  = criterion(mod(stat_in,     attention_mask=ex_causal_mask,    position_ids=ex_position_ids,    cache_position=   ex_cache_position)[0], stat_out)
                loss_learn = criterion(mod(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids, cache_position=input_cache_position)[0], layer_out_ks)
                loss = loss_keep + loss_learn
            elif config.model_name == 'Qwen1.5-7B-Chat':
                loss_keep  = criterion(mod(stat_in,     attention_mask=ex_causal_mask,    position_ids=   ex_position_ids)[0], stat_out)
                loss_learn = criterion(mod(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids)[0], layer_out_ks)
            loss = loss_keep + loss_learn
            loss.backward(retain_graph=True)
            optimizer.step()

            print(f'[Opt-K step {step + 1}/{config.optim_num_step}] Layer: {layer}, loss keep: {loss_keep.item():.4f}, loss_learn: {loss_learn.item():.4f}')
            if loss.item() < 5e-5:
                break

        for x in [layer_in_ks, layer_out_ks, cur_zs, targets, stat_in, stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return weights_copy


if __name__ == "__main__":
    set_seed()
    config = Config()

    edit_data, ex_data = load_rank_A_data()
    ex_data = [get_qwen_without_answer(i['question']) for i in ex_data]
    #ex_data = [get_qwen_without_answer(i['instruction'] + i['input']) + i['output'] for i in ex_data]

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map=f"cuda:{config.device}",
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(config.model_path)                              # 训练用, bs = 1
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side='left')   # 推理用, bs > 1, 须右对齐
    if config.model_name == 'LLama2-7B-Chat':
        tok.pad_token_id = tok.eos_token_id

    device = f'cuda:{str(config.device)}'
    batch_size = config.batch_size
    num_batches = len(edit_data) // batch_size + (1 if len(edit_data) % batch_size else 0)
    edited_data = []
    for batch_index in tqdm(range(num_batches)):
        # context data
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = edit_data[start_index:end_index]
        for it in batch:
            it['question']       = get_qwen_without_answer(it['question'])
            it['para_question']  = get_qwen_without_answer(it['para_question'])
            it['para_question1'] = get_qwen_without_answer(it['para_question1'])
            it['para_question2'] = get_qwen_without_answer(it['para_question2'])
            it['sub_question']   = get_list_qwen_without_answer(it['sub_question'], cot=False)
            if config.model_name == 'LLama2-7B-Chat':
                it['answer'] += '</s>'
            elif config.model_name == 'Qwen1.5-7B-Chat':
                it['answer'] += '<|im_end|>'
        random_elements = random.sample(ex_data, config.ex_data_num)

        # Edit Knowledge!
        weights_copy = execute_batch_unke(model, tok, config, batch, random_elements)

        # Test original & paraphrase questions
        for data in batch:
            question = tokenizer([data['question'], data['para_question'], data['para_question1'], data['para_question2']], return_tensors='pt', padding=True)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=question['input_ids'].to(device),
                    attention_mask=question['attention_mask'].to(device),
                    do_sample=False,
                    #temperature=0.001,
                    max_new_tokens=200,
                )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            if batch_index < 10 // config.batch_size + 1:
                print(f"question: {data['question']}")
                print(output[0])
                print(f"question: {data['para_question']}")
                print(output[1])
            data['original_prediction'] = output[0]
            data['para_prediction']     = output[1]
            data['para_prediction1']    = output[2]
            data['para_prediction2']    = output[3]
            #if config.model_name == 'LLama2-7B-Chat':
            #    data['answer'] = data['answer'][:-len('</s>')]
            #elif config.model_name == 'Qwen1.5-7B-Chat':
            #    data['answer'] = data['answer'][:-len('<|im_end|>')]

        # Test sub-questions
        for data in batch:
            question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=question['input_ids'].to(device),
                    attention_mask=question['attention_mask'].to(device),
                    do_sample=False,
                    #temperature=0.001,
                    max_new_tokens=100,
                )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            if batch_index < 10 // config.batch_size + 1:
                print(f"question: {data['sub_question']}")
                print(output)
            data['sub_prediction'] = output

        edited_data.extend(batch)

        if config.keep_original_weight:     # single edit
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(f"cuda:{config.device}")

    save_infer_data(edit_data)
