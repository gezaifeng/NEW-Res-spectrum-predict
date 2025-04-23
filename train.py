import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ResCNN
from dataset import SpectralDataset
from torch.optim.lr_scheduler import StepLR

# ========================= 参数配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "D:\\Desktop\\model\\Res spectrum predict\\dataset-4.17\\train"
val_dir = "D:\\Desktop\\model\\Res spectrum predict\\dataset-4.17\\val"
save_path = "D:\\Desktop\\model\\Res spectrum predict\\best_model.pth"
log_path = "D:\\Desktop\\model\\Res spectrum predict\\training_log.csv"
fig_path = "D:\\Desktop\\model\\Res spectrum predict\\loss_curve.png"

num_epochs = 150
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-4
step_size = 50
gamma = 0.5  # 每 step_size epoch 将 lr 乘以 gamma

# ========================= 数据加载 =========================
train_dataset = SpectralDataset(train_dir, normalize_spectra=False)
val_dataset = SpectralDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========================= 模型、优化器、调度器 =========================
model = ResCNN().to(device)
criterion = nn.MSELoss()                                               # MSELoss /SmoothL1Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)         #  CosineAnnealingLR/ReduceLROnPlateau
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# ========================= 日志记录变量 =========================
log = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "learning_rate": []
}

best_val_loss = float("inf")

# ========================= 训练循环 =========================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, spectra in train_loader:
        images, spectra = images.to(device).float(), spectra.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, spectra)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, spectra in val_loader:
            images, spectra = images.to(device).float(), spectra.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs, spectra).item()

    val_loss /= len(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # 日志记录
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    log["learning_rate"].append(current_lr)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

    scheduler.step()

# ========================= 保存日志和图像 =========================
df = pd.DataFrame(log)
df.to_csv(log_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.grid(True)
plt.legend()
# ✅ 添加超参数文本框
hyperparams = (
    f"Epochs: {num_epochs}\n"
    f"Batch Size: {batch_size}\n"
    f"Learning Rate: {learning_rate}\n"
    f"Weight Decay: {weight_decay}\n"
    f"LR Step Size: {step_size}\n"
    f"Gamma: {gamma}"
)
plt.gcf().text(0.98, 0.5, hyperparams, fontsize=10, va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(fig_path)
plt.close()

print(f"📊 训练日志已保存到: {log_path}")
print(f"📈 Loss 曲线图已保存到: {fig_path}")
print("🎯 训练完成！")
