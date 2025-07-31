import torch
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import random


class FLDDataset(Dataset):

    def __init__(
        self, data_path: Path,  context=51, forecast=8
    ):
        self.data = pickle.load(open(data_path, "rb"))
        self.context = context
        self.forecast = forecast

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx % (len(self.data))
        start_idx = random.randint(
            a=0,
            b=self.data[0]["observed"].shape[1] - self.context - self.forecast,
        )
        X_t = torch.tensor(
            self.data[idx]["observed"][:, start_idx : start_idx + self.context].T,
            dtype=torch.float32,
        )
        X_tk = torch.zeros(self.forecast, *X_t.shape, dtype=torch.float32)

        for i in range(self.forecast):
           
            X_tk[i,...] = torch.tensor(
                self.data[idx]["observed"][:, start_idx+i : start_idx + self.context+i].T, dtype=torch.float32
            )
       
        return X_t, X_tk
