import torch
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
import pickle


class SegmentDataset(Dataset):
    def __init__(self, data_path, split="train"):
        self.data = pickle.load(open(data_path, "rb"))
        if split == "train":
            self.data = self.data[: int(0.8 * len(self.data))]
        else:
            self.data = self.data[int(0.8 * len(self.data)) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx]["segment"], dtype=torch.float32)
        y = torch.tensor(self.data[idx]["label"], dtype=torch.float32)
        return X, y
