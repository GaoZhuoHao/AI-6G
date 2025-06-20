import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np

# 从我们自己创建的模块中导入
from data_loader import MultiInputSignalDataset
from model import MultiInputModel

# --- 1. 定义损失函数 ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits已经是sigmoid之后的值
        probs = logits 
        
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        # 使用BCEWithLogitsLoss更稳定，它内部包含了sigmoid
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # 注意：BCEWithLogitsLoss需要未经过sigmoid的原始输出
        # 而我们的模型最后加了Sigmoid，为使用此损失，暂时先逆操作
        # 更好的做法是移除模型最后的Sigmoid，在这里统一处理
        # 假设model的输出为logits (w/o sigmoid)
        bce = self.bce_loss(logits, targets)
        
        # DiceLoss需要经过sigmoid的概率值
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice

# --- 2. 评估指标 ---
def dice_coefficient(probs, targets, smooth=1e-6):
    """计算Dice系数用于评估"""
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

# --- 3. 训练和验证函数 ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training")
    for inputs, targets in progress_bar:
        # 将数据移动到指定设备
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        targets = targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        # 注意：为使用BCEWithLogitsLoss，理想情况下模型不应有最后的sigmoid激活
        # 我们在这里假设模型输出的是logits
        logits = model(inputs)
        
        # 计算损失
        loss = criterion(logits, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating")
        for inputs, targets in progress_bar:
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            # 计算评估指标
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, targets)
            
            running_loss += loss.item()
            total_dice += dice.item()
            progress_bar.set_postfix(val_loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

    avg_loss = running_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice

# --- 4. 主训练函数 ---
def train(config: dict):
    """
    主训练流程函数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集
    full_dataset = MultiInputSignalDataset(h5_path=config['data_path'], config=config)
    
    # # 划分训练集和验证集
    # val_percent = config.get('validation_split', 0.2)
    # n_val = int(len(full_dataset) * val_percent)
    # n_train = len(full_dataset) - n_val
    # train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # --- 添加以下代码来创建一个用于快速调试的子集 ---
    from torch.utils.data import Subset
    # 只取前400个样本进行快速测试，您可以按需调整数量
    subset_indices = range(400) 
    debug_dataset = Subset(full_dataset, indices=subset_indices)
    print(f"--- 调试模式: 仅使用 {len(debug_dataset)} 个样本进行快速验证 ---")
    # --------------------------------------------------

    # 使用 debug_dataset 替代 full_dataset 进行后续操作
    val_percent = config.get('validation_split', 0.2)
    n_val = int(len(debug_dataset) * val_percent)
    n_train = len(debug_dataset) - n_val
    train_dataset, val_dataset = random_split(debug_dataset, [n_train, n_val])
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    print(f"Data loaded: {n_train} training samples, {n_val} validation samples.")

    # 初始化模型、损失函数和优化器
    model = MultiInputModel().to(device)
    # **重要**: 如果模型末尾有Sigmoid，请注释掉。损失函数BCEWithLogitsLoss会处理它。
    # 假设我们的MultiInputModel输出的是原始logits。
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # 学习率调度器（可选，但推荐）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # 训练循环
    best_dice = -1.0
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        scheduler.step() # 更新学习率

        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # 保存模型
        # 保存最新的模型
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_latest.pth'))
        
        # 如果是目前最好的模型，则单独保存
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"🎉 New best model saved with Dice score: {best_dice:.4f}")

    print("\n--- Training Finished ---")