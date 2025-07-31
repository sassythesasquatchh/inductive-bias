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
from pre_util import get_model, parse_args, calculate_energy, get_length
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # def on_train_start(self):
    #     wandb.define_metric("test_orbit", step_metric="epoch")
    #     wandb.define_metric("test_open", step_metric="epoch")

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
        # if self.current_epoch > 0 and self.current_epoch % 30 == 0:
        #     self.custom_eval(self, self.global_step)
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
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # early_stop = EarlyStopping(monitor="val_loss", patience=7, mode="min", verbose=True)

    # return [checkpoint, early_stop]
    return [checkpoint]


def load_args_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
    """Load args from a checkpoint's wandb config file."""
    run_id = checkpoint_path.split("/")[-2]  # Extract the run ID from path
    api = wandb.Api()
    run = api.run(f"inductive-biases/{run_id}")  # Fetch the wandb run

    loaded_args = run.config  # Get stored args
    return argparse.Namespace(**loaded_args)


class RMMLoss(nn.Module):
    def __init__(self, alpha=0.9, forecast=8):
        super(RMMLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer(
            "weights", torch.tensor([alpha**k for k in range(forecast)])
        )

    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2 * self.weights.view(1, -1, 1))


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.9,
        forecast=8,
        rwm_factor=1,
        energy_factor=0.01,
        length_factor=0.05,
    ):
        super(CombinedLoss, self).__init__()
        self.rmmloss = RMMLoss(alpha=alpha, forecast=forecast)
        self.mse = nn.MSELoss()
        self.energy_factor = energy_factor
        self.length_factor = length_factor
        self.rwm_factor = rwm_factor
        self.forecast = forecast

    def forward(self, pred, target):
        rwm_loss = self.rmmloss(pred, target)

        pred_energy = calculate_energy(
            pred.reshape(pred.size(0) * self.forecast, -1)
        ).reshape(pred.size(0), self.forecast)
        energy_loss = torch.var(pred_energy, dim=1).mean()
        pred_length = get_length(
            pred.reshape(pred.size(0) * self.forecast, -1)
        ).reshape(pred.size(0), self.forecast)
        gt_length = get_length(
            target.reshape(target.size(0) * self.forecast, -1)
        ).reshape(target.size(0), self.forecast)
        length_loss = self.mse(pred_length, gt_length)
        # length_loss = torch.var(pred_length, dim=1).mean()
        # ipdb.set_trace()
        return (
            self.rwm_factor * rwm_loss
            + self.energy_factor * energy_loss
            + self.length_factor * length_loss
        )


def get_model(args):
    """Return model and loss function based on configuration."""
    if args.model == "unstructured":
        model = RWMLatentUnstructured(
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.embedding_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "baseline":
        model = RWMUnstructuredBaseline(
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "informed":
        model = InformedRWM(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "hybrid":
        model = HybridRWM(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "fully-informed":
        model = FullyInformed(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "informed-cnn":
        model = InformedCNN(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )
    elif args.model == "informed-hybrid":
        model = HybridRWM(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )

    elif args.model == "partially-informed":
        model = PartiallyInformed(
            dt=args.dt,
            g=args.g,
            l=args.l,
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            context=args.context,
            forecast=args.forecast,
        )

    elif args.model == "fld":
        model = FLD(
            observable_dim=args.observable_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            sequence_length=args.context,
        )
    # if args.loss == "rwm":
    #     criterion = RMMLoss(forecast=args.forecast)
    # elif args.loss == "rmm_energy":
    criterion = CombinedLoss(
        forecast=args.forecast,
        rwm_factor=args.rwm_factor,
        energy_factor=args.energy_factor,
        length_factor=args.length_factor,
    )
    return model, criterion


def main(args: argparse.Namespace) -> None:
    # If loading from checkpoint, override args with saved ones
    if args.checkpoint:
        args = load_args_from_checkpoint(args.checkpoint)

    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(True)

    
    model, criterion = get_model(args)
    # Initialize components
    train_dataset = RWMDataset(
        args.train_path,
        context=args.context,
        forecast=args.forecast,
        flatten=not (hasattr(model, "encoder") and isinstance(model.encoder, CNNEncoder)),
    )
    val_dataset = RWMDataset(
        args.val_path,
        split="val",
        context=args.context,
        forecast=args.forecast,
        flatten=not (hasattr(model, "encoder") and isinstance(model.encoder, CNNEncoder)),
    )

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
            "patience": 10,
            "min_lr": 1e-6,
            # "verbose": True,
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=os.cpu_count()
    )

    if args.model != "fully-informed":
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
    test_rollout(
        lit_model, test_loader, args, step=trainer.global_step, as_figure=False
    )
    advect_trajectory(lit_model, test_loader, args)

    # test_data = TrajectoryDataset(
    #     data_path="data/continuity_test_data.pkl", split="test"
    # )
    # test_loader = DataLoader(test_data, batch_size=1)

    # try:
    #     test_continuity(lit_model, test_loader, args)
    # except Exception as e:
    #     print("Error during continuity test:", e)

    # Finalize logging
    if logger and isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    args.run_name = f"{args.model}_{args.embedding_dim}_{args.forecast}_{args.context}_{args.hidden_dim}"
    args.training = "rwm" if args.model not in ["fully-informed"] else "normal"
    try:
        main(args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
