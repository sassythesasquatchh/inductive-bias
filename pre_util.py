import argparse
import pytorch_lightning as pl
from models import *
from typing import Union
import torch
from pathlib import Path
from torch import nn
from typing import Optional, Dict, Any
from datasets import PointDataset, SegmentDataset, TrajectoryDataset
from torch.utils.data import Dataset
from constants import *
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import ipdb


class LitModel(pl.LightningModule):
    """PyTorch Lightning module encapsulating model, training, and validation logic."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            **(self.hparams.optimizer_kwargs or {}),
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer, **(self.hparams.scheduler_kwargs or {})
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: Union[str, Path], **kwargs
    ) -> pl.LightningModule:
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )


def get_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    """Initialize and return training and validation datasets."""
    data_path = Path(args.data_path)
    test_data_path = Path(args.visualisation_data_path)

    if args.dataset == "point":
        train_dataset = PointDataset(data_path=data_path, split="train")
        val_dataset = PointDataset(data_path=data_path, split="val")
        test_dataset = TrajectoryDataset(data_path=test_data_path, split="test")
    elif args.dataset == "segment":
        train_dataset = SegmentDataset(split="train")
        val_dataset = SegmentDataset(split="val")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    return train_dataset, val_dataset, test_dataset


def get_model(
    args: argparse.Namespace, model_name: str = None
) -> tuple[nn.Module, nn.Module]:
    """Initialize and return model and criterion based on arguments."""
    model_kwargs = {
        "observable_dim": args.observable_dim,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
    }

    if model_name is None:
        model_name = args.model

    if model_name == "informed":
        if args.end2end:
            model = InformedEnd2End(dt=args.dt, g=args.g, l=args.l, **model_kwargs)
        else:
            model = Informed(dt=args.dt, g=args.g, l=args.l, **model_kwargs)
        criterion = nn.MSELoss()
    elif model_name == "hybrid":
        if args.end2end:
            model = HybridEnd2End(dt=args.dt, g=args.g, l=args.l, **model_kwargs)
        else:
            model = Hybrid(dt=args.dt, g=args.g, l=args.l, **model_kwargs)
        criterion = nn.MSELoss()
    elif model_name == "fld":
        model = FLD(
            segment_length=args.segment_length,
            **model_kwargs,
        )
        criterion = nn.MSELoss()
    elif model_name == "unstructured":
        # model = Unstructured(**model_kwargs)
        if args.end2end:
            model = UnstructuredEnd2End(**model_kwargs)
        else:
            model = Unstructured(**model_kwargs)
        # model = UnstructuredDirect(**model_kwargs)
        criterion = nn.MSELoss()
    elif model_name == "baseline":
        model = Baseline(**model_kwargs)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Model {args.model} not supported.")

    return model, criterion


def calculate_energy(trajectory):
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    # Assumes also unit mass and length
    X = trajectory[:, :4]
    Y = trajectory[:, 4:8]
    vel = trajectory[:, 8:12]

    # numerator = (X * Y).sum(dim=1)
    # denominator = (X**2).sum(dim=1)
    # eps = 1e-8
    # a = numerator / (denominator + eps)
    # v_x = torch.ones_like(a)
    # v_y = a
    theta = torch.atan2(X.reshape(-1), Y.reshape(-1)) + np.pi
    # theta = torch.remainder(theta, 2 * np.pi)
    # theta = theta.reshape(-1, 4)
    height = 1 - torch.cos(theta)
    height = height.reshape(-1, 4)
    height = torch.mean(height, dim=1)
    ang_vel = vel / torch.tensor(SAMPLING_POSITIONS).to(vel.device)
    ang_vel = torch.mean(ang_vel, dim=1)
    energy = 0.5 * (ang_vel**2) + GRAVITY * height

    # residuals = Y - a[:, None] * X
    # variance = (residuals**2).mean(dim=1)
    # plt.plot(
    #     np.arange(0, len(variance)),
    #     variance.cpu().numpy(),
    # )
    # plt.savefig("variance.png")
    # plt.clf()

    # plt.plot(np.arange(0, len(X)), X.cpu().numpy())
    # plt.savefig("X.png")
    # plt.clf()
    # Remove large values from Y by setting them to 0
    # v_y = torch.where(torch.abs(v_y) > 10, torch.tensor(0.0, device=v_y.device), v_y)
    # plt.plot(np.arange(0, len(v_y)), v_y.cpu().numpy())
    # plt.savefig("Y.png")
    # plt.clf()

    # theta = theta.cpu().numpy()
    # plt.plot(np.arange(0, len(theta)), theta)
    # plt.savefig("theta.png")
    # plt.clf()
    # plt.plot(np.arange(0, len(ang_vel)), ang_vel)
    # plt.savefig("ang_vel.png")
    # plt.clf()
    # plt.plot(np.arange(0, len(energy)), energy)
    # plt.savefig("energy.png")
    # plt.clf()
    return energy


def get_length(trajectory):
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    X = trajectory[:, :4]
    Y = trajectory[:, 4:8]
    length = torch.sqrt(X**2 + Y**2).sum(dim=1)
    return length


def get_variance(trajectory):
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    X = trajectory[:, :4]
    Y = trajectory[:, 4:8]

    theta = torch.atan2(X.reshape(-1), Y.reshape(-1)) + np.pi
    theta = theta.reshape(-1, 4)
    theta = torch.mean(theta, dim=1)

    X_best = torch.cos(theta)
    Y_best = torch.sin(theta)
    v = torch.stack((X_best, Y_best), dim=1)
    v = v / torch.norm(v, dim=1, keepdim=True)

    points = torch.empty(X.size(0), 4, 2).to(X.device)
    points[:, :, 0] = X
    points[:, :, 1] = Y
    centered = points - points.mean(dim=1, keepdim=True)
    projections = torch.einsum("bni,bi->bn", centered, v)
    residuals = centered - projections.unsqueeze(-1) * v.unsqueeze(1)
    squared_distances = torch.sum(residuals**2, dim=-1)
    variances = torch.mean(squared_distances, dim=1)
    # plt.plot(
    #     np.arange(0, len(variances)),
    #     variances.cpu().numpy(),
    # )
    # plt.savefig("variance.png")
    # plt.clf()
    return variances


def process_trajectory(model, x, args, device):
    x = x.squeeze(0).to(device).transpose(0, 1)
    reconstructed_trajectory = torch.zeros_like(x)
    latent_trajectory = torch.zeros(x.size(0), args.embedding_dim).to(device)
    latent_reconstructed_trajectory = torch.zeros_like(latent_trajectory)
    if not args.training == "rwm":
        ic = x[0].unsqueeze(0)
        reconstructed_trajectory[0] = ic
        upper_bound = x.size(0) - 1
    else:
        x_t = x[: args.context, :].reshape(1, -1)
        x_t1 = torch.zeros_like(x_t)
        reconstructed_trajectory[: args.context, :] = x[: args.context, :]
        upper_bound = x.size(0) - args.context
        # try:
        #     latent_reconstructed_trajectory[: args.context, :] = model.model.encoder(
        #         x[: args.context, :].reshape(-1)
        #     ).reshape(args.context, args.embedding_dim)
        #     latent_trajectory[: args.context, :] = model.model.encoder(
        #         x[: args.context, :].reshape(-1)
        #     ).reshape(args.context, args.embedding_dim)
        # except Exception as e:
        #     pass
    for j in range(0, upper_bound):
        if not args.training == "rwm":
            reconstructed_trajectory[j + 1] = model.model(
                reconstructed_trajectory[j].unsqueeze(0)
            )
        else:
            # Get the first prediction
            # ipdb.set_trace()
            z = model.model(x_t)[:, 0, :].squeeze()
            reconstructed_trajectory[j + args.context] = z
            x_t1[:, : x_t.size(1) - args.observable_dim] = x_t[:, args.observable_dim :]
            # Add the new prediction to the end
            x_t1[:, -args.observable_dim :] = z
            x_t = x_t1.clone()
            try:
                latent_reconstructed_trajectory[j + args.context] = model.model.encoder(
                    x_t
                ).reshape(args.context, args.embedding_dim)[-1, :]
                latent_trajectory[j + args.context] = model.model.encoder(
                    x[j : j + args.context, :].reshape(1, -1)
                ).reshape(args.context, args.embedding_dim)[-1, :]
            except Exception as e:
                pass
    try:
        if not args.training == "rwm":
            latent_trajectory = model.model.encoder(x)
            latent_reconstructed_trajectory = model.model.encoder(
                reconstructed_trajectory
            )
        else:
            latent_trajectory[: args.context, :] = latent_trajectory[args.context, :]
            latent_reconstructed_trajectory[: args.context, :] = (
                latent_reconstructed_trajectory[args.context, :]
            )
    except Exception as e:
        pass

    return (
        x,
        reconstructed_trajectory,
        latent_trajectory,
        latent_reconstructed_trajectory,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pendulum Dynamics Learning")

    # Experiment config
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--tags", type=str, help="Comma-separated list of tags")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")

    # Data config
    parser.add_argument("--dataset", type=str, default="point")
    parser.add_argument(
        "--train_path", type=str, default="data/normal_training_1000.pkl"
    )
    parser.add_argument("--val_path", type=str, default="data/validation.pkl")
    parser.add_argument(
        "--visualisation_data_path",
        type=str,
        default="data/testing.pkl",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--early_stopping", action="store_true")

    # Model config
    parser.add_argument("--model", type=str, default="unstructured")
    parser.add_argument("--observable_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=3)
    parser.add_argument("--segment_length", type=int, default=51)

    # Physics parameters
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--g", type=float, default=GRAVITY)
    parser.add_argument("--l", type=float, default=L)

    # Training config
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--end2end", action="store_true")
    # parser.add_argument("--loss", type=str, default="rwm")
    parser.add_argument("--energy_factor", type=float, default=0.0)
    parser.add_argument("--length_factor", type=float, default=0.0)
    parser.add_argument("--rwm_factor", type=float, default=1.0)

    # Optimization config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # RWM config
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--forecast", type=int, default=8)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = (
            f"{args.model}_{args.embedding_dim}_{args.max_k}_{args.hidden_dim}"
        )
    return args
