import argparse
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
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
from visualisation import advect_trajectory
from test_continuity import test_continuity
from pre_util import LitModel, get_datasets, get_model, parse_args

# Load environment variables first
load_dotenv()


def get_logger(args: argparse.Namespace) -> Optional[Logger]:
    """Initialize and return logger based on configuration."""
    if args.debug:
        return None

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

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    return [checkpoint, early_stop]


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

    # Initialize components
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    model, criterion = get_model(args)

    lit_model = LitModel(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": args.weight_decay},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=os.cpu_count()
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=os.cpu_count())

    # Resume training if checkpoint is provided
    if args.checkpoint:
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.checkpoint)
    else:
        trainer.fit(lit_model, train_loader, val_loader)

    # plot_trajectories(lit_model, test_loader, args)
    advect_trajectory(lit_model, test_loader, args)

    test_data = TrajectoryDataset(
        data_path="data/continuity_test_data.pkl", split="test"
    )
    test_loader = DataLoader(test_data, batch_size=1)

    test_continuity(lit_model, test_loader, args)

    # Finalize logging
    if logger and isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
