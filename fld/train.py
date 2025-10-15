import argparse
import os
from typing import Optional

import ipdb
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import Logger, WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from util.dataset import TorchTrajectoryDataset
from util.pre_util import parse_args
from util.rollout import evaluate_rollout
from util.test_continuity import test_continuity
from util.train import LitModel, get_callbacks, get_logger, load_args_from_checkpoint
from util.visualisation import animate_trajectories

from .data import FLDDataset
from .model import FLD

# Load environment variables first
load_dotenv()


class FLDLoss(nn.Module):
    def __init__(self, forecast=8, alpha=0.9):
        super(FLDLoss, self).__init__()
        self.forecast = forecast
        self.alpha = alpha
        self.register_buffer(
            "weights", torch.tensor([alpha**k for k in range(forecast)])
        )

    def forward(self, pred, target):
        # Pred and target have dimension (batch_size, forecast_length, segment_length, observable_dim)
        return torch.mean((pred - target) ** 2 * self.weights.view(1, -1, 1, 1))


def train(
    train_path: str,
    val_path: str,
    hidden_dim: int = 64,
    embedding_dim: int = 2,
    context: int = 51,
    forecast: int = 8,
    weight_decay: float = 1e-4,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 200,
    debug: bool = False,
    checkpoint_path: Optional[str] = None,
    logger: Optional[Logger] = None,
    callbacks: Optional[list] = None,
) -> LitModel:
    criterion = FLDLoss(forecast=forecast)

    train_dataset = FLDDataset(
        data_path=train_path,
        context=context,
        forecast=forecast,
    )
    val_dataset = FLDDataset(
        data_path=val_path,
        context=context,
        forecast=forecast,
    )

    observable_dim = train_dataset.get_observable_dim()

    model = FLD(
        observable_dim=observable_dim,
        hidden_dim=hidden_dim,
        latent_dim=embedding_dim,
        segment_length=context,
        forecast=forecast,
    )

    lit_model = LitModel(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": weight_decay},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
            # "verbose": True,
        },
        learning_rate=learning_rate,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=not debug,
        log_every_n_steps=10 if not debug else 1,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        fast_dev_run=debug,
        overfit_batches=10 if debug else 0,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=os.cpu_count()
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=checkpoint_path)

    return lit_model


def main(args: argparse.Namespace) -> None:
    # If loading from checkpoint, override args with saved ones
    if args.checkpoint:
        args = load_args_from_checkpoint(args.checkpoint)

    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(True)

    logger = get_logger(args)
    callbacks = get_callbacks(args) if not args.debug else []

    model = train(
        observable_dim=args.observable_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        context=args.context,
        forecast=args.forecast,
        train_path=args.train_path,
        val_path=args.val_path,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        debug=args.debug,
        checkpoint_path=args.checkpoint,
        logger=logger,
        callbacks=callbacks,
    )

    visualisation_dataset = TorchTrajectoryDataset(
        data_path=args.visualisation_data_path, type="observed"
    )

    continuity_dataset = TorchTrajectoryDataset(
        data_path=args.continuity_data_path, type="observed"
    )
    model.eval()
    with torch.no_grad():
        rollout = model.model.rollout(visualisation_dataset.data.to(model.device))

    evaluate_rollout(rollout, visualisation_dataset)
    animate_trajectories(
        rollout,
        visualisation_dataset.config,
        visualisation_dataset.traj_names,
        args.run_name,
    )

    with torch.no_grad():
        rollout = model.model.rollout(continuity_dataset.data.to(model.device))

    # Continuity test
    test_continuity(rollout, continuity_dataset.initial_velocities)

    # Finalize logging
    if logger and isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    args.run_name = (
        f"fld_{args.embedding_dim}_{args.forecast}_{args.context}_{args.hidden_dim}"
    )
    args.training = "fld"
    try:
        main(args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
