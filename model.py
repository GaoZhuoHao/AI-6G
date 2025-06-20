import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# --- 辅助模块: 1D和2D卷积块，减少代码重复 ---
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# --- 主模型: MultiInputModel ---
class MultiInputModel(nn.Module):
    def __init__(self, time_channels=2, freq_channels=1, spec_channels=1):
        super().__init__()
        
        # === 1. 编码器分支 ===
        
        # 频域编码器 (与U-Net结构类似)
        self.freq_enc1 = ConvBlock1D(freq_channels, 64)
        self.freq_enc2 = ConvBlock1D(64, 128)
        self.freq_enc3 = ConvBlock1D(128, 256)
        
        # 时域编码器
        self.time_enc1 = ConvBlock1D(time_channels, 64)
        self.time_enc2 = ConvBlock1D(64, 128)
        self.time_enc3 = ConvBlock1D(128, 256)

        # 时频谱图编码器
        self.spec_enc1 = ConvBlock2D(spec_channels, 64)
        self.spec_enc2 = ConvBlock2D(64, 128)
        self.spec_enc3 = ConvBlock2D(128, 256)
        
        self.pool1d = nn.MaxPool1d(2)
        self.pool2d = nn.MaxPool2d(2)

        # === 2. 特征融合瓶颈 ===
        # 假设经过3次池化, 1D长度变为 100000 / 8 = 12500
        # 2D 形状: H/8, W/8. 例如: 1024/8=128, 197/8~24
        # 需要一个全连接层来统一维度
        # 这里的in_features需要根据实际池化后的维度计算，可能需要动态获取
        # 这是一个示例值，实际使用时可能需要调整
        self.fusion_fc_spec = nn.Linear(256 * 128 * 24, 256)
        self.fusion_fc_time = nn.Linear(256, 256)
        
        # 瓶颈处的卷积层
        # 输入通道 = freq(256) + time_proj(256) + spec_proj(256)
        self.bottleneck = ConvBlock1D(256 + 256 + 256, 512)
        
        # === 3. 解码器分支 (带跳跃连接) ===
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(256 + 256, 256) # 256(来自upconv3) + 256(来自freq_enc3)

        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(128 + 128, 128) # 128(来自upconv2) + 128(来自freq_enc2)

        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(64 + 64, 64) # 64(来自upconv1) + 64(来自freq_enc1)

        # === 4. 输出层 ===
        self.out_conv = nn.Conv1d(64, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_time = inputs['time']
        x_freq = inputs['freq']
        x_spec = inputs['spec']
        
        # --- 编码过程 ---
        # 频域 (保存跳跃连接的输出)
        f1 = self.freq_enc1(x_freq)
        f2 = self.freq_enc2(self.pool1d(f1))
        f3 = self.freq_enc3(self.pool1d(f2))
        freq_bottleneck = self.pool1d(f3)

        # 时域
        t1 = self.time_enc1(x_time)
        t2 = self.time_enc2(self.pool1d(t1))
        t3 = self.time_enc3(self.pool1d(t2))
        time_bottleneck = self.pool1d(t3)
        
        # 时频谱图
        s1 = self.spec_enc1(x_spec)
        s2 = self.spec_enc2(self.pool2d(s1))
        s3 = self.spec_enc3(self.pool2d(s2))
        spec_bottleneck = self.pool2d(s3)

        # --- 瓶颈融合 ---
        # 1. 展平并投射时频和时域特征
        spec_flat = spec_bottleneck.view(spec_bottleneck.size(0), -1)
        spec_proj = self.fusion_fc_spec(spec_flat)
        
        # 对1D特征使用全局平均池化来获得向量
        time_pooled = F.adaptive_avg_pool1d(time_bottleneck, 1).squeeze(-1)
        time_proj = self.fusion_fc_time(time_pooled)

        # 2. 扩展维度以匹配频域特征图的长度
        time_proj_expanded = time_proj.unsqueeze(-1).expand(-1, -1, freq_bottleneck.size(2))
        spec_proj_expanded = spec_proj.unsqueeze(-1).expand(-1, -1, freq_bottleneck.size(2))

        # 3. 拼接所有特征
        fused_bottleneck = torch.cat([freq_bottleneck, time_proj_expanded, spec_proj_expanded], dim=1)
        
        # 4. 瓶颈卷积
        b = self.bottleneck(fused_bottleneck)
        
        # --- 解码过程 (带跳跃连接) ---
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, f3], dim=1) # 跳跃连接
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, f2], dim=1) # 跳跃连接
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, f1], dim=1) # 跳跃连接
        d1 = self.dec1(d1)
        
        # --- 输出 ---
        output = self.out_conv(d1)
        output = output.squeeze(1)
        return output

# # --- 如何使用 ---
# if __name__ == '__main__':
#     # 模拟一个批次的数据
#     batch_size = 2
#     dummy_inputs = {
#         "time": torch.randn(batch_size, 2, 100000),
#         "freq": torch.randn(batch_size, 1, 100000),
#         "spec": torch.randn(batch_size, 1, 1024, 197) # 注意这里的尺寸与实际数据加载器一致
#     }

#     print("正在实例化模型...")
#     model = MultiInputModel()
#     print(model)
    
#     print("\n正在进行一次前向传播...")
#     output = model(dummy_inputs)
    
#     print(f"\n前向传播完成！")
#     print(f"输入 'freq' 的形状: {dummy_inputs['freq'].shape}")
#     print(f"输出掩码的形状: {output.shape}")
    
#     # 验证输出形状是否与输入频域特征的形状一致
#     assert output.shape == dummy_inputs['freq'].shape
#     print("\n断言通过：输出形状与目标形状一致。模型定义成功！")