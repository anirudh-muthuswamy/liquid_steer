import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchinfo import summary
import numpy as np

class WeightedMSE(nn.Module):
    def __init__(self, alpha=0.1):
        super(WeightedMSE, self).__init__()
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        # squared error
        squared_error = (predictions - targets)**2
        
        # weighting factor: w(y) = exp(|y|*alpha)
        weights = torch.exp(torch.abs(targets) * self.alpha)
        weighted_loss = squared_error * weights

        return weighted_loss.mean()

class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        kT, kH, kW = kernel_size
        pT, pH, pW = padding

        #spatial conv: only HxW
        self.spatial = nn.Conv3d(in_channels, out_channels, kernel_size=(1, kH, kW), padding=(0, pH, pW), bias=False)
        self.bn_spatial = nn.BatchNorm3d(out_channels)

        #temporal conv: only T
        self.temporal = nn.Conv3d(out_channels, out_channels, kernel_size=(kT, 1, 1), padding=(pT, 0, 0), bias=False)
        self.bn_temporal = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.spatial(x)
        x = self.bn_spatial(x)
        x = self.relu(x)
        x = self.temporal(x)
        x = self.bn_temporal(x)
        return self.relu(x)
    

class ResidualMain(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        p = tuple(k//2 for k in kernel_size)
        self.conv1 = Conv2Plus1D(in_channels, out_channels, kernel_size, p)
        self.conv2 = Conv2Plus1D(out_channels, out_channels, kernel_size, p)

    def forward(self, x):
        return self.conv2(self.conv1(x))
    
class Project(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.proj(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.main = ResidualMain(in_channels, out_channels, kernel_size)
        self.need_proj = (in_channels != out_channels)
        if self.need_proj:
            self.proj = Project(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.proj(x) if self.need_proj else x
        out = self.main(x)
        return self.relu(out + res)
    
class ResizeVideo(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        # collapse batch such that we treat frames as images
        frames = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        frames = F.interpolate(frames, size=self.out_size,
                               mode='bilinear', align_corners=False)
        # reshape back
        h, w = self.out_size
        return frames.reshape(B, T, C, h, w).permute(0,2,1,3,4)
    
class TemporalResNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 seq_len=16,
                 height=224,
                 width=224):
        super().__init__()

        self.stem = nn.Sequential(
            Conv2Plus1D(in_channels, 16, kernel_size=(3,7,7), padding=(1,3,3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            ResizeVideo((height//2, width//2))
        )
        #residual stages, downsampling spatially between them
        self.stage1 = nn.Sequential(
            ResidualBlock(16,  16, (3,3,3)),
            ResizeVideo((height//4, width//4))
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(16,  32, (3,3,3)),
            ResizeVideo((height//8, width//8))
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(32,  64, (3,3,3)),
            ResizeVideo((height//16, width//16))
        )
        self.stage4 = ResidualBlock(64, 128, (3,3,3))

        # global spatio‑temporal pooling with linear regressor
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))  # [B, 128, 1,1,1]
        self.fc1   = nn.Linear(128, 64)
        self.leaky = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x):
        # [B, C, T, H, W]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).view(x.size(0), -1) # [B, 128]
        return self.fc2(self.leaky(self.fc1(x))) # [B,1]

if __name__ == "__main__":
    # random input
    B, C, T, H, W = 2, 1, 16, 123, 455
    dummy = torch.randn(B, C, T, H, W)

    model = TemporalResNet(in_channels=C,
                           seq_len=T,
                           height=H,
                           width=W)
    
    summary(model=model, input_size=dummy.shape)
