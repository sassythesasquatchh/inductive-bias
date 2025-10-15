import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch
from pathlib import Path
import wandb
from pytorch_lightning.loggers import Logger, WandbLogger
from typing import Any, Callable, Dict, Optional, Union


import argparse
import os

from pathlib import Path


def create_run_name(args: argparse.Namespace) -> str:
    """Create a run name based on model configuration."""
    try:
        run_name = args.run_name
    except AttributeError:
        pass
    try:
        run_name = f"{args.encoder}_{args.dynamics}_{args.decoder}"
    except AttributeError:
        pass
    if not run_name:
        run_name = "default"
    args.run_name = run_name
    return run_name


def get_logger(args: argparse.Namespace) -> Optional[Logger]:
    """Initialize and return logger based on configuration."""
    if args.debug:
        return None

    run_name = create_run_name(args)

    wandb.login(key=os.getenv("WANDB_KEY"))
    logger = WandbLogger(
        project="inductive-biases",
        name=run_name,
        log_model="best",
        tags=args.tags.split(",") if args.tags else None,
    )

    # Log args
    logger.experiment.config.update(vars(args))
    print("Configured wandb logger")
    return logger


def get_callbacks(args: argparse.Namespace) -> list[pl.Callback]:
    """Return list of callbacks for training."""
    checkpoint = ModelCheckpoint(
        dirpath=Path("checkpoints") / args.run_name,
        filename="{epoch}-{val_loss:.2f}",
        # monitor="val_loss",
        # save_top_k=3,
        # mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    return [checkpoint]


def load_args_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
    """Load args from a checkpoint's wandb config file."""
    run_id = checkpoint_path.split("/")[-2]  # Extract the run ID from path
    api = wandb.Api()
    run = api.run(f"inductive-biases/{run_id}")  # Fetch the wandb run

    loaded_args = run.config  # Get stored args
    return argparse.Namespace(**loaded_args)


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
        custom_eval: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        self.custom_eval = custom_eval

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

    def on_validation_epoch_end(self):
        pass

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
