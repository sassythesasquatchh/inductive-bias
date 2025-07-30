"""
Building blocks for the autoencoder models studied in this experiment.
"""

from torch import nn


class PointEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PointEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PointDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PointDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(hidden_layers - 1)
            ],
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
