import torch
import numpy as np
import os
from model import ResCNN  # 确保 `model.py` 里定义了 `SpectralCNNTransformer`
import matplotlib
matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from  dataset import SpectralDataset



# 1. 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载训练好的模型
model_path = "D:\\Desktop\\model\\Res spectrum predict\\best_model.pth"
model = ResCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()



def visualize_prediction(true_spectrum, predicted_spectrum, wavelengths=None):
    """
    可视化真实光谱 vs 预测光谱
    :param true_spectrum: 真实光谱数据 (10,)
    :param predicted_spectrum: 预测光谱数据 (10,)
    :param wavelengths: 波长点 (可选, 默认为 1-10)
    """
    if wavelengths is None:
        wavelengths = np.arange(1, 77)  # 默认波长点 1~77
    mse = mean_squared_error(true_spectrum, predicted_spectrum)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_spectrum, predicted_spectrum)
    mape = np.mean(np.abs((true_spectrum - predicted_spectrum) / (true_spectrum + 1e-8))) * 100  # 避免除零
    r2 = r2_score(true_spectrum, predicted_spectrum)

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, true_spectrum, 'bo-', label='real spectrum')
    plt.plot(wavelengths, predicted_spectrum, 'ro-', label='predict spectrum')

    plt.xlabel("wavelength")
    plt.ylabel("absorbance")
    plt.title("Spectral prediction results")
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}%\nR²: {r2:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid()
    plt.show(block=True)  # 强制 Matplotlib 阻塞运行，等待窗口关闭



def predict(rgb_npy_path, dataset):
    """
    预测单个 RGB 数据对应的光谱数据
    :param rgb_npy_path: 需要预测的 RGB 数据路径 (4, 6, 100, 3)
    :param dataset: 训练时的 Dataset（用于反归一化）
    :return: 反归一化的光谱数据
    """
    # 加载 RGB 数据
    rgb_data = np.load(rgb_npy_path).astype(np.float32)  # (4, 6, 100, 3)

    # 归一化 RGB 数据
    rgb_data /= 255.0

    # 交换维度: (4, 6, 100, 3) → (3, 100, 4, 6)
    rgb_data = np.transpose(rgb_data, (3, 2, 0, 1))

    # 转换为 PyTorch tensor 并添加 batch 维度
    rgb_tensor = torch.tensor(rgb_data).unsqueeze(0).to(device)  # (1, 3, 100, 4, 6)

    # 预测
    with torch.no_grad():
        predicted_spectrum = model(rgb_tensor).cpu().numpy().flatten()  # (10,)

    # 反归一化
    predicted_spectrum = predicted_spectrum * dataset.std + dataset.mean
    return predicted_spectrum





# 训练数据路径（用于获取数据均值 & 标准差）
train_dir = "D:\\Desktop\\model\\Res spectrum predict\\dataset-4.17\\train"

# 2. 加载训练数据集（确保光谱数据归一化）
train_dataset = SpectralDataset(train_dir, normalize_spectra=True)

# 3. 预测单个样本
test_rgb_path = "D:\\Desktop\\model\\Res spectrum predict\\dataset-4.17\\val\\rgb_0007.npy"
predicted_spectrum = predict(test_rgb_path, train_dataset)  # 传入 train_dataset

print(f"预测的光谱数据: {predicted_spectrum}")

# 可视化单个预测
true_spectrum = np.load("D:\\Desktop\\model\\Res spectrum predict\\dataset-4.17\\val\\spectral_0007.npy")
visualize_prediction(true_spectrum, predicted_spectrum)