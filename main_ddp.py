# main.py (DDP版本)
import torch
import numpy as np
import os
import argparse
from train_ddp import train # 导入新的train函数

def main():
    # DDP会通过环境变量设置这些值
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 配置字典可以保持不变
    CONFIG = {
        "data_path": "/home/dytccq/AI+6G/data/train.h5",
        "save_dir": "/home/dytccq/AI+6G/saved_models/multi_input_ddp_v1/",
        # **注意**: batch_size现在是“每个GPU”的batch_size
        # 全局总batch_size将是 8 * 3 = 24
        "batch_size": 8,
        "learning_rate": 0.001,
        "epochs": 50,
        "validation_split": 0.2, # 注意：DDP模式下，random_split需要每个进程用相同seed
        "SAMPLING_RATE": 100_000_000,
        "START_FREQ": 2400.0,
        "END_FREQ": 2500.0,
        "NUM_BINS": 100000,
        "STFT_NPERSEG": 1024,
        "STFT_NOVERLAP": 512,
    }
    CONFIG['FREQS_1D'] = np.linspace(CONFIG['START_FREQ'], CONFIG['END_FREQ'], CONFIG['NUM_BINS'])

    # 设置随机种子保证所有进程初始化模型权重一致
    torch.manual_seed(42)
    
    # 启动训练
    train(rank, world_size, config=CONFIG)

if __name__ == '__main__':
    main()