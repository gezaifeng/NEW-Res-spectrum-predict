import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ========================= 数据集定义 =========================
class SpectralDataset(Dataset):
    def __init__(self, data_dir, normalize_spectra=False):
        """
        读取存储在 data_dir 目录下的 RGB 和光谱数据
        注意：光谱吸光度具有物理意义，建议设置 normalize_spectra=False
        """
        self.data_dir = data_dir
        self.rgb_files = sorted([f for f in os.listdir(data_dir) if f.startswith("rgb") and f.endswith(".npy")])
        self.spectral_files = sorted([f.replace("rgb_", "spectral_") for f in self.rgb_files])

        self.normalize_spectra = normalize_spectra
        self.mean, self.std = None, None

        if self.normalize_spectra:
            self.mean, self.std = self.compute_spectra_stats()

    def compute_spectra_stats(self):
        all_spectra = []
        for spectral_file in self.spectral_files:
            spectral_path = os.path.join(self.data_dir, spectral_file)
            spectral_data = np.load(spectral_path).astype(np.float32)
            all_spectra.append(spectral_data)

        all_spectra = np.stack(all_spectra)
        return np.mean(all_spectra, axis=0), np.std(all_spectra, axis=0)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.data_dir, self.rgb_files[idx])
        spectral_path = os.path.join(self.data_dir, self.spectral_files[idx])

        rgb_data = np.load(rgb_path).astype(np.float32) / 255.0
        spectral_data = np.load(spectral_path).astype(np.float32)

        # (4,6,100,3) → (3,100,4,6)
        rgb_data = np.transpose(rgb_data, (3, 2, 0, 1))

        if self.normalize_spectra:
            spectral_data = (spectral_data - self.mean) / self.std

        return torch.tensor(rgb_data), torch.tensor(spectral_data, dtype=torch.float32)



