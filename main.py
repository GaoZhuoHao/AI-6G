import torch
import os
from train import train  # 从我们之前定义的 src/train.py 中导入 train 函数
import numpy as np

def run_training():
    """
    主函数，用于定义配置并启动训练流程。
    """
    # --- 1. 定义项目配置 (替代YAML文件) ---
    # 将所有超参数和路径集中在此字典中，方便管理
    CONFIG = {
        # --- 路径配置 ---
        # 请确保这是您本地的正确数据路径
        "data_path": "../data/train.h5", 
        # 模型保存路径
        "save_dir": "../saved_models/multi_input_v1/",

        # --- 训练超参数 ---
        "learning_rate": 0.001,
        "batch_size": 8,       # 根据您的GPU显存大小调整
        "epochs": 50,          # 训练的总轮数
        "validation_split": 0.2, # 验证集占总数据的比例 (20%)

        # --- 数据处理参数 (必须与data_loader.py中的定义一致) ---
        "START_FREQ": 2400.0,
        "END_FREQ": 2500.0,
        "NUM_BINS": 100000,

        # 新增明确的物理采样率 (单位: Hz)
        "SAMPLING_RATE": 100_000_000, # <--- 新增: 100 MS/s
        
        # STFT参数
        "STFT_NPERSEG": 1024,
        "STFT_NOVERLAP": 512,
    }

    # --- 第二处修改：在这里预先计算并添加 FREQS_1D ---
    CONFIG['FREQS_1D'] = np.linspace(CONFIG['START_FREQ'], CONFIG['END_FREQ'], CONFIG['NUM_BINS'])

    # --- 2. 设置随机种子以保证可复现性 ---
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- 3. 打印配置信息并启动训练 ---
    print("--- 训练配置 ---")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("--------------------")

    # 调用训练函数，开始训练
    try:
        train(config=CONFIG)
        print("\n训练流程正常结束。")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        # 在这里可以添加更详细的错误处理或日志记录
        raise

if __name__ == '__main__':
    # 当直接运行 `python main.py` 时，执行此函数
    run_training()