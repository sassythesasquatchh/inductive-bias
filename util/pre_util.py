import argparse
import pytorch_lightning as pl
from typing import Union
import torch
from pathlib import Path
from torch import nn
from typing import Optional, Dict, Any
from torch.utils.data import Dataset
from util.config import *
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from jaxtyping import Float


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
    # elif args.dataset == "segment":
    #     train_dataset = SegmentDataset(split="train")
    #     val_dataset = SegmentDataset(split="val")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    return train_dataset, val_dataset, test_dataset


def calculate_energy(
    trajectory: Float[torch.Tensor, "batch N obs"], config: Config
) -> Float[torch.Tensor, "batch N"]:
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    # Assumes also unit mass and length
    X = trajectory[:, :, :4]
    Y = trajectory[:, :, 4:8]
    vel = trajectory[:, :, 8:12]

    theta = torch.atan2(X, Y) + np.pi
    theta = torch.mean(theta, dim=-1)  # Average over sampled points
    height = torch.ones_like(theta) - torch.cos(theta)
    ang_vel = vel / torch.tensor(config.SAMPLING_POSITIONS).to(vel.device)
    ang_vel = torch.mean(ang_vel, dim=-1)
    energy = 0.5 * (ang_vel**2) + config.GRAVITY * height
    return energy


def get_length(
    trajectory: Float[torch.Tensor, "batch N obs"], config: Config
) -> Float[torch.Tensor, "batch N"]:
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    X = trajectory[:, :, :4]
    Y = trajectory[:, :, 4:8]
    length = (
        torch.sqrt(X**2 + Y**2)
        / torch.tensor(config.SAMPLING_POSITIONS).to(X.device).view(1, 1, 4)
    ).sum(dim=-1)
    return length


def parse_args() -> argparse.Namespace:
    config = Config()
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
        default="data/visualisation.pkl",
    )
    parser.add_argument(
        "--continuity_data_path", type=str, default="data/continuity_test.pkl"
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)

    # Model config
    # parser.add_argument("--model", type=str, default="unstructured")
    parser.add_argument("--observable_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=3)
    parser.add_argument("--segment_length", type=int, default=51)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--decoder", type=str)
    parser.add_argument("--dynamics", type=str)

    # Physics parameters
    parser.add_argument("--dt", type=float, default=config.DT)
    parser.add_argument("--g", type=float, default=config.GRAVITY)
    parser.add_argument("--l", type=float, default=config.L)

    # Training config
    parser.add_argument("--noise", type=float, default=0.0)

    # Optimization config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # RWM config
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--forecast", type=int, default=8)

    args = parser.parse_args()

    return args
