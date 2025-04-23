import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """带 BatchNorm 和 Dropout 的残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(0.2)

        # 匹配维度的投影
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += identity
        return F.relu(out)


class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()

        # 初始通道：3 → 32
        self.block1 = ResBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # (4,6) → (2,3)

        # 32 → 64
        self.block2 = ResBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)  # (2,3) → (1,1)

        # 64 → 128
        self.block3 = ResBlock(64, 128)  # (1,1)

        self.flatten_dim = 128 * 1 * 1  # 输出特征尺寸

        # 全连接层
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 76)

    def forward(self, x):
        b, c, seq, h, w = x.shape
        x = x.reshape(b * seq, c, h, w)

        x = self.pool1(self.block1(x))  # → (b*seq, 32, 2, 3)
        x = self.pool2(self.block2(x))  # → (b*seq, 64, 1, 1)
        x = self.block3(x)              # → (b*seq, 128, 1, 1)

        x = x.view(b, seq, -1)          # → (b, seq, 128)

        # 平均多个空间点
        x = x.mean(dim=1)               # → (b, 128)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)                 # → (b, 76)

        return x
