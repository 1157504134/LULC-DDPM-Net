import model as Model
opt = {'name': 'ddpm-RS-CDHead-ALL-PCA', 
       'phase': 'train', 'gpu_ids': [0], 
       'path': {'log': 'experiments\\ddpm-RS-CDHead-ALL-PCA_230802_162323\\logs', 
                'tb_logger': 'experiments\\ddpm-RS-CDHead-ALL-PCA_230802_162323\\tb_logger', 
                'results': 'experiments\\ddpm-RS-CDHead-ALL-PCA_230802_162323\\results', 
                'checkpoint': 'experiments\\ddpm-RS-CDHead-ALL-PCA_230802_162323\\checkpoint',
                'resume_state': 'experiments/Channel/I500000_E109', 
                'experiments_root': 'experiments\\ddpm-RS-CDHead-ALL-PCA_230802_162323'}, 
       'path_cd': {'log': 'logs', 'tb_logger': 'tb_logger', 'results': 'results', 'checkpoint': 'checkpoint', 'resume_state': None}, 
       'datasets': {'train': {'name': 'MS_LCD', 'dataroot': 'dataset/MULT_PAN_ALL', 'resolution': 256, 'batch_size': 3, 'num_workers': 16, 'use_shuffle': True, 'data_len': -1}, 'val': {'name': 'MS_LCD', 'dataroot': 'dataset/MULT_PAN_ALL', 'resolution': 256, 'batch_size': 3, 'num_workers': 16, 'use_shuffle': True, 'data_len': -1}, 'test': {'name': 'MS_LCD', 'dataroot': 'dataset/MULT_PAN_ALL', 'resolution': 256, 'batch_size': 3, 'num_workers': 16, 'use_shuffle': False, 'data_len': -1}}, 'model_cd': {'feat_scales': [14, 11, 8, 5, 2], 'out_channels': 2, 'loss_type': 'ce', 'output_cm_size': 256, 'feat_type': 'dec', 't': [50, 100, 
        400]}, 'model': {'which_model_G': 'sr3', 'finetune_norm': False, 'unet': {'in_channel': 3, 'out_channel': 3, 'inner_channel': 128, 'channel_multiplier': [1, 2, 4, 8, 8], 'attn_res': [16], 'res_blocks': 2, 'dropout': 0.2, 'norm_groups': 32}, 'beta_schedule': {'train': {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-06, 'linear_end': 0.01}, 'val': {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-06, 'linear_end': 0.01}, 'test': {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-06, 'linear_end': 0.01}}, 'diffusion': {'image_size': 256, 'channels': 3, 'loss': 'l2', 'conditional': False}}, 'train': {'n_epoch': 120, 'train_print_freq': 500, 'val_freq': 1, 'val_print_freq': 50, 'optimizer': {'type': 'adam', 'lr': 0.0001}, 'sheduler': {'lr_policy': 'linear', 'n_steps': 3, 'gamma': 0.1}}, 'wandb': {'project': 'ddpm-RS-CDHead_MS_LCD'}, 'distributed': False, 'log_eval': True, 'enable_wandb': True, 'len_train_dataloader': 160, 'len_val_dataloader': 20}
# 加载模型
diffusion2 = Model.create_model(opt)
# Set noise schedule for the diffusion model
diffusion2.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])


