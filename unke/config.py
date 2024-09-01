class Config:

    ''' Model '''
    #model_name = 'LLama2-7B-Chat'
    #model_path = 'meta-llama/Llama-2-7b-chat-hf'
    model_name = 'Qwen1.5-7B-Chat' 
    model_path = 'Qwen/Qwen1.5-7B-Chat'

    ''' Data '''
    data_path = '../data/final_test_data_a.json'
    ex_data_path = '../data/alpaca_data.json'

    ''' Train '''
    batch_size = 1
    ex_data_num = 20
    ln_f_module = "model.norm"
    lm_head_module = "lm_head"
    layer_module_tmp = "model.layers.{}"
    keep_original_weight = True
    # optim-K
    layers = [7]
    optim_num_step = 50
    lr = 2e-4
    # optim-V
    v_loss_layer = 31
    v_num_grad_steps = 25
    v_lr = 5e-1
    v_weight_decay = 1e-3
    clamp_norm_factor = 4

    ''' Misc '''
    device = 0
