import argparse
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable
from dotenv import load_dotenv

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, Logger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
import wandb

from datasets import *
from models import *
from visualisation import (
    plot_trajectories,
    advect_trajectory,
    advect_trajectory_latent_rollout,
)
from pre_util import get_model, parse_args
from test_continuity import test_continuity
from test_rollout import test_rollout
from torch import nn
import ipdb

# Load environment variables first
load_dotenv()


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

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.model(x, k)

    # def on_train_start(self):
    #     wandb.define_metric("test_orbit", step_metric="epoch")
    #     wandb.define_metric("test_open", step_metric="epoch")

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y, k = batch
        y_hat = self(x.squeeze(), k)
        loss = self.criterion(y_hat, y.squeeze())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y, k = batch
        y_hat = self(x.squeeze(), k)
        loss = self.criterion(y_hat, y.squeeze())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # if self.current_epoch % 10 == 0:
        self.custom_eval(self, self.global_step)

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


def get_logger(args: argparse.Namespace) -> Optional[Logger]:
    """Initialize and return logger based on configuration."""
    if args.debug:
        return None

    try:
        run_name = args.run_name
    except AttributeError:
        run_name = f"{args.model}_{args.embedding_dim}_{args.max_k}_{args.noise}"
    args.run_name = run_name

    wandb.login(key=os.getenv("WANDB_KEY"))
    logger = WandbLogger(
        project="inductive-biases",
        name=args.run_name,
        log_model="all",
        tags=args.tags.split(",") if args.tags else None,
    )

    # Log args
    logger.experiment.config.update(vars(args))
    return logger


def get_callbacks(args: argparse.Namespace) -> list[pl.Callback]:
    """Return list of callbacks for training."""
    checkpoint = ModelCheckpoint(
        dirpath=Path("checkpoints") / args.run_name,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=7, mode="min", verbose=True)

    if args.early_stopping:
        return [checkpoint, early_stop]
    else:
        return [checkpoint]


def load_args_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
    """Load args from a checkpoint's wandb config file."""
    run_id = checkpoint_path.split("/")[-2]  # Extract the run ID from path
    api = wandb.Api()
    run = api.run(f"inductive-biases/{run_id}")  # Fetch the wandb run

    loaded_args = run.config  # Get stored args
    return argparse.Namespace(**loaded_args)


def main(args: argparse.Namespace) -> None:
    # If loading from checkpoint, override args with saved ones
    if args.checkpoint:
        args = load_args_from_checkpoint(args.checkpoint)

    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Initialize components
    train_dataset = PointDatasetPrebatched(
        args.train_path,
        noise=args.noise,
        batch_size=args.batch_size,
        max_forward_pred=args.max_k,
    )
    val_dataset = PointDatasetPrebatched(
        args.val_path,
        split="val",
        batch_size=args.batch_size,
        max_forward_pred=args.max_k,
    )
    model, criterion = get_model(args)

    test_data = TrajectoryDataset(data_path=args.visualisation_data_path, split="test")
    test_loader = DataLoader(test_data, batch_size=1)

    lit_model = LitModel(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": args.weight_decay},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-6,
            "verbose": True,
        },
        learning_rate=args.learning_rate,
        custom_eval=lambda model, step: test_rollout(
            model, test_loader, args, step=step
        ),
    )

    logger = get_logger(args)
    callbacks = get_callbacks(args) if not args.debug else []

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=not args.debug,
        log_every_n_steps=10 if not args.debug else 1,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        fast_dev_run=args.debug,
        overfit_batches=10 if args.debug else 0,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=os.cpu_count())

    # Resume training if checkpoint is provided
    if args.checkpoint:
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.checkpoint)
    else:
        trainer.fit(lit_model, train_loader, val_loader)

    if not args.debug and hasattr(trainer, "checkpoint_callback"):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            lit_model = LitModel.load_from_checkpoint(
                best_model_path,
                model=model,
                criterion=criterion,
            )

    # advect_trajectory(lit_model, test_loader, args)
    advect_trajectory_latent_rollout(lit_model, test_loader, args)
    test_rollout(
        lit_model, test_loader, args, step=trainer.global_step, as_figure=False
    )

    test_data = TrajectoryDataset(
        data_path="data/continuity_test_data.pkl", split="test"
    )
    test_loader = DataLoader(test_data, batch_size=1)

    try:
        test_continuity(lit_model, test_loader, args)
    except Exception as e:
        print("Error during continuity test:", e)

    # Finalize logging
    if logger and isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
