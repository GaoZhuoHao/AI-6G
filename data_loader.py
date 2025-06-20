import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from scipy import signal

# --- 1. 参数与配置 ---
CONFIG = {
    'H5_PATH': '../data/train.h5', # 请确保这是您的正确路径
    'START_FREQ': 2400.0,
    'END_FREQ': 2483.5,
    'NUM_BINS': 100000,
    'STFT_NPERSEG': 1024,
    'STFT_NOVERLAP': 512,
}
# 预计算频率轴，避免重复计算
CONFIG['FREQS_1D'] = np.linspace(CONFIG['START_FREQ'], CONFIG['END_FREQ'], CONFIG['NUM_BINS'])

# --- 2. 核心处理函数 (多模态版本) ---
def process_sample_multi_input(waveform: np.ndarray, label_bytes: bytes, config: dict):
    """
    对单个样本生成时域、频域、时频域三种特征及目标掩码。
    这个版本可以安全地处理 label_bytes 为 None 的情况。
    """
    # === 特征 1: 时域特征 (I/Q) ===
    # 将复数波形分离为实部和虚部，作为两个通道
    iq_data = np.stack([waveform.real, waveform.imag], axis=0).astype(np.float32)

    # === 特征 2: 频域特征 (1D PSD) ===
    fft_result = np.fft.fft(waveform)
    psd = np.abs(fft_result) ** 2
    psd = np.fft.fftshift(psd)
    psd_min, psd_max = psd.min(), psd.max()
    if psd_max - psd_min > 1e-8:
        psd = (psd - psd_min) / (psd_max - psd_min)
    else:
        psd = np.zeros(config['NUM_BINS'], dtype=np.float32)
    psd = psd.astype(np.float32)

    # === 特征 3: 时频特征 (2D Spectrogram) ===
    fs_hz = config['SAMPLING_RATE'] # 单位是 Hz
    f, t, Zxx = signal.stft(waveform, fs=fs_hz, nperseg=config['STFT_NPERSEG'], noverlap=config['STFT_NOVERLAP'])
    spectrogram = np.log1p(np.abs(np.fft.fftshift(Zxx, axes=0)))
    spec_min, spec_max = spectrogram.min(), spectrogram.max()
    if spec_max - spec_min > 1e-8:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spectrogram = np.zeros_like(spectrogram, dtype=np.float32)
    spectrogram = spectrogram.astype(np.float32)
    
    # === 统一目标: 1D 频率掩码 ===
    #
    # ==================== 核心修改逻辑 ====================
    #
    # 先判断 label_bytes 是否为 None
    if label_bytes is not None:
        # 情况A: 有标签 (label_bytes 不是 None)，正常处理
        mask = np.zeros(config['NUM_BINS'], dtype=np.float32)
        try:
            interval_list = ast.literal_eval(label_bytes.decode('utf-8'))
            if interval_list:
                # 合并区间
                interval_list.sort()
                merged = [interval_list[0]]
                for current in interval_list[1:]:
                    last = merged[-1]
                    if current[0] <= last[1]:
                        last[1] = max(last[1], current[1])
                    else:
                        merged.append(current)
                
                # 创建掩码
                freqs_1d_axis = np.linspace(config['START_FREQ'], config['END_FREQ'], config['NUM_BINS'])
                for start_f, end_f in merged:
                    start_idx = np.searchsorted(freqs_1d_axis, start_f, side='left')
                    end_idx = np.searchsorted(freqs_1d_axis, end_f, side='right')
                    mask[start_idx:end_idx] = 1.0
        except (SyntaxError, ValueError, AttributeError):
            # 增加对 AttributeError 的捕获以防万一，但主要逻辑已由 if/else 处理
            pass
    else:
        # 情况B: 没有标签 (label_bytes 是 None)，创建全为零的虚拟掩码
        mask = np.zeros(config['NUM_BINS'], dtype=np.float32)
   
    return iq_data, psd, spectrogram, mask

# --- 3. 构建PyTorch Dataset 类 (多输入版本) ---

class MultiInputSignalDataset(Dataset):
    """为多输入模型提供时域、频域、时频域三种数据的Dataset。"""
    def __init__(self, h5_path: str, config: dict):
        self.h5_path = h5_path
        self.config = config
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = len(f['waveforms'])

    def __len__(self):
        return self.num_samples

    # =================================================================
    # !! 删除您现有的整个 __getitem__ 函数，然后粘贴下面的最终版本 !!
    # =================================================================
    def __getitem__(self, idx: int):
        
        # ================== 这是我们加入的调试探针 ==================
        #print(f">>>> 正在执行 data_loader.py 中【新的】__getitem__ 函数, 处理索引: {idx} <<<<")
        # ==========================================================

        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # 1. 首先，无条件地加载所有样本都包含的 'waveforms'
        waveform = self.h5_file['waveforms'][idx]

        # 2. 接下来，核心逻辑：先检查 'labels' 是否存在
        if 'labels' in self.h5_file:
            # 情况A: 文件中有标签 (例如训练或验证集)
            # 加载标签，并正常调用处理函数
            label_bytes = self.h5_file['labels'][idx]
            iq_data, psd, spectrogram, mask = process_sample_multi_input(waveform, label_bytes, self.config)
            # 根据处理后得到的 mask 创建 target
            targets = torch.from_numpy(mask) #.unsqueeze(0) # 根据需要调整形状

        else:
            # 情况B: 文件中没有标签 (例如测试集)
            # 传递 None 作为 label_bytes 调用处理函数
            # !! 重要假设 !!: 这要求您的 process_sample_multi_input 函数能够处理 label_bytes=None 的情况
            # 它在这种情况下应该返回一个虚拟的、无用的 mask。
            iq_data, psd, spectrogram, _ = process_sample_multi_input(waveform, None, self.config)
            # 创建一个空的 tensor 作为 targets 的占位符
            targets = torch.empty(0)

        # 3. 封装所有情况下都生成的 inputs 字典
        inputs = {
            "time": torch.from_numpy(iq_data),
            "freq": torch.from_numpy(psd).unsqueeze(0),
            "spec": torch.from_numpy(spectrogram).unsqueeze(0)
        }
        
        # 4. 返回 inputs 和对应的 targets (有标签或为空)
        return inputs, targets
    # =================================================================