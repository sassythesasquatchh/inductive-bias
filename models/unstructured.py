from torch import nn
import torch
from jaxtyping import Float
from .blocks import MLP


class Unstructured(nn.Module):
    def __init__(self, observable_dim=12, hidden_dim=30, embedding_dim=3):
        super(Unstructured, self).__init__()
        self.encoder = MLP(observable_dim, hidden_dim, embedding_dim, hidden_layers=1)
        
        self.dynamics = MLP(
            embedding_dim, hidden_dim, embedding_dim, hidden_layers=2
        )
        
        self.decoder = MLP(embedding_dim, hidden_dim, observable_dim, hidden_layers=1)
        

    def forward(self, x: torch.Tensor, k: int = 1):
        z = self.encoder(x)
        for _ in range(k):
            z = self.dynamics(z)
        return self.decoder(z)


class UnstructuredDirect(nn.Module):
    def __init__(self, observable_dim=12, hidden_dim=30, embedding_dim=3):
        super(UnstructuredDirect, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observable_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observable_dim),
        )

    def forward(self, x: torch.Tensor, k: int = 1):
        z = self.encoder(x)
        z = torch.cat(
            [
                z,
                torch.tensor([[k]], dtype=z.dtype, device=z.device).expand(
                    z.size(0), 1
                ),
            ],
            dim=1,
        )

        z = self.dynamics(z)
        return self.decoder(z)


class UnstructuredEnd2End(nn.Module):
    def __init__(self, observable_dim=12, hidden_dim=30, embedding_dim=3):
        super(UnstructuredEnd2End, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observable_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observable_dim),
        )

    def forward(self, x: torch.Tensor, k: int = 1):
        for _ in range(k):
            z = self.encoder(x)
            z = self.dynamics(z)
            x = self.decoder(z)
        return x


class RWMUnstructuredBaseline(nn.Module):
    def __init__(self, observable_dim=12, hidden_dim=64, context=33, forecast=9):
        super(RWMUnstructuredBaseline, self).__init__()
        self.dynamics = MLP(
            observable_dim * context, hidden_dim, observable_dim, hidden_layers=2
        )
        self.forecast = forecast
        self.context = context
        self.observable_dim = observable_dim

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim"])-> Float[torch.Tensor, "batch forecast observable_dim"]:
        B = x.size(0)

        output = torch.zeros(
            (B, self.forecast, self.observable_dim), dtype=x.dtype, device=x.device
        )
        x_next = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for i in range(self.forecast):
            z = self.dynamics(x)
            output[:, i, :] = z
            # Move the last n-1 to be the first n-1
            x_next[:, : x.size(1) - self.observable_dim] = x[:, self.observable_dim :]
            # Add the new prediction to the end
            x_next[:, -self.observable_dim :] = z
            x = x_next.clone()

        # Return the next self.forecast predictions
        return output


class RWMLatentUnstructured(nn.Module):
    def __init__(
        self, observable_dim=12, hidden_dim=64, latent_dim=3, context=32, forecast=8
    ):
        super(RWMLatentUnstructured, self).__init__()
        self.encoder =MLP(
            observable_dim * context, hidden_dim, latent_dim*context, hidden_layers=2
        )
        self.dynamics = MLP(
            latent_dim*context, hidden_dim, latent_dim, hidden_layers=2
        )
        self.decoder = MLP(
            latent_dim * (forecast + context), hidden_dim, observable_dim * forecast, hidden_layers=2
        )
        self.forecast = forecast
        self.context = context
        self.observable_dim = observable_dim
        self.latent_dim = latent_dim

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim"])-> Float[torch.Tensor, "batch forecast observable_dim"]:
        B = x.size(0)

        z_forecast = torch.zeros(
            (B, self.forecast, self.latent_dim), dtype=x.dtype, device=x.device
        )
        z_context = self.encoder(x)
        z_context_next = torch.zeros_like(
            z_context, dtype=z_context.dtype, device=z_context.device
        )

        for i in range(self.forecast):
            z = self.dynamics(z_context)
            z_forecast[:, i, :] = z
            # Move the last n-1 to be the first n-1
            z_context_next[:, : z_context.size(1) - self.latent_dim] = z_context[
                :, self.latent_dim :
            ]
            # Add the new prediction to the end
            z_context_next[:, -self.latent_dim :] = z
            z_context = z_context_next.clone()

        z_total = torch.cat(
            [
                z_context,
                z_forecast.reshape(B, -1),
            ],
            dim=1,
        )

        return self.decoder(z_total).reshape(B, self.forecast, self.observable_dim)


class Baseline(nn.Module):
    def __init__(self, observable_dim=12, hidden_dim=128, embedding_dim=3):
        super(Baseline, self).__init__()

        # self.dynamics = nn.Sequential(
        #     nn.Linear(observable_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, observable_dim),
        # )
        self.dynamics = MLP(
            observable_dim, hidden_dim, observable_dim, hidden_layers=2
        )

    def forward(self, x: torch.Tensor, k: int = 1):
        for _ in range(k):
            x = self.dynamics(x)
        return x
