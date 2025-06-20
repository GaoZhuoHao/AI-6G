# src/train_ddp.py (已修正数据集划分)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np

# --- DDP相关的导入 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 从我们自己创建的模块中导入
from data_loader import MultiInputSignalDataset
from model import MultiInputModel

# --- DDP 设置与清理函数 ---
def setup_ddp(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 任意未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # NCCL是NVIDIA GPU推荐的后端

def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# --- 损失函数和评估指标 (无需修改) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        # 假设logits已经是sigmoid之后的结果，如果不是，需要在这里加上
        probs = logits 
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

def dice_coefficient(probs, targets, smooth=1e-6):
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

# --- 训练和验证函数 (基本不变) ---
def train_one_epoch(model, loader, optimizer, criterion, device, rank, epoch):
    model.train()
    running_loss = 0.0
    # 设置sampler的epoch，保证每个epoch的shuffle都不同
    loader.sampler.set_epoch(epoch)
    
    # 只在主进程(rank 0)显示进度条
    progress_bar = tqdm(loader, desc=f"Training Epoch {epoch+1}", disable=(rank != 0))
    for inputs, targets in progress_bar:
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if rank == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # 在所有进程间同步损失值 (可选，但有助于获得准确的全局损失)
    # 注意：这里的损失是基于每个进程的batch数，而不是全局的
    total_loss_tensor = torch.tensor(running_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    
    # 为了得到全剧平均loss，需要除以全局的dataset size
    # 但为了简化，通常我们只关心主进程的loss趋势
    # 这里返回的是所有进程的loss总和，打印时再处理
    return total_loss_tensor.item() / len(loader.dataset) * dist.get_world_size()


def validate(model, loader, criterion, device):
    # 验证逻辑在DDP中无需特殊改动，因为模型权重是同步的
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    rank = dist.get_rank()
    
    # 为了得到精确的全局验证指标，需要收集所有进程的结果
    all_losses = []
    all_dices = []
    
    with torch.no_grad():
        # 每个进程独立验证自己的数据子集
        progress_bar = tqdm(loader, desc="Validating", disable=(rank != 0))
        for inputs, targets in progress_bar:
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            targets = targets.to(device)
            
            logits = model.module(inputs) # DDP模型需要访问.module
            loss = criterion(logits, targets)
            
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, targets)
            
            all_losses.append(loss.item())
            all_dices.append(dice.item())

    # ==================== 代码核心修改区域 START ====================
    # 每个进程先计算自己负责的数据子集的平均loss和dice
    local_avg_loss = np.mean(all_losses) if all_losses else 0.0
    local_avg_dice = np.mean(all_dices) if all_dices else 0.0

    # 将每个进程的平均值转换为tensor
    sum_loss_tensor = torch.tensor(local_avg_loss, device=device)
    sum_dice_tensor = torch.tensor(local_avg_dice, device=device)

    # 1. 使用 SUM 操作将所有进程的平均值累加起来
    dist.all_reduce(sum_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_dice_tensor, op=dist.ReduceOp.SUM)

    # 2. 获取进程总数
    world_size = dist.get_world_size()

    # 3. 将总和除以进程数，得到全局的平均值
    global_avg_loss = sum_loss_tensor.item() / world_size
    global_avg_dice = sum_dice_tensor.item() / world_size

    return global_avg_loss, global_avg_dice


# --- 主训练函数 (已修正) ---
def train(rank, world_size, config: dict):
    """
    DDP主训练流程函数
    """
    setup_ddp(rank, world_size)
    
    # 每个进程使用自己的GPU
    device = rank
    torch.cuda.set_device(device)
    
    # ==================== 代码核心修改区域 START ====================
    # 1. 实例化完整数据集
    full_dataset = MultiInputSignalDataset(h5_path=config['data_path'], config=config)

    # 2. 定义划分参数
    validation_split = config.get('validation_split', 0.2)
    random_seed = config.get('seed', 42)
    
    # 3. 计算划分大小
    total_size = len(full_dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    
    # 4. 使用固定的随机种子进行数据集划分，确保所有进程划分结果一致
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    if rank == 0:
        print(f"Dataset successfully split. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 5. 为训练集和验证集分别创建独立的分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 6. 为训练集和验证集分别创建独立的DataLoader
    # batch_size现在是每个GPU的batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False # shuffle必须为False，因为sampler已经处理了
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    # ==================== 代码核心修改区域 END ====================

    # 模型初始化并移动到对应GPU，然后用DDP包装
    model = MultiInputModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    best_dice = -1.0
    
    for epoch in range(config['epochs']):
        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # 训练时，使用train_loader
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, rank, epoch)
        
        # 验证时，使用val_loader
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 只在主进程(rank 0)进行日志打印和模型保存
        if rank == 0:
            print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            
            save_dir = config['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save(model.module.state_dict(), os.path.join(save_dir, 'checkpoint_latest.pth'))
            
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.module.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"🎉 New best model saved with Dice score: {best_dice:.4f}")

    cleanup_ddp()