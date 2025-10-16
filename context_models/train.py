import argparse
import os
from typing import Optional

import ipdb
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import Logger, WandbLogger
from torch.utils.data import DataLoader

from context_models.decoders import BaseDecoder
from context_models.dynamics import BaseDynamics
from context_models.encoders import BaseEncoder
from context_models.model import ContextModel
from context_models.util import RWMLoss
from util.dataset import TorchTrajectoryDataset
from util.pre_util import parse_args
from util.rollout import evaluate_rollout
from util.test_continuity import test_continuity
from util.train import (
    LitModel,
    create_run_name,
    get_callbacks,
    get_logger,
    load_args_from_checkpoint,
)
from util.visualisation import animate_trajectories

from .config import decoder_dict, dynamics_dict, encoder_dict
from .data import ContextDataset

# Load environment variables first
load_dotenv()


def train(
    train_path: str,
    val_path: str,
    encoder_class: BaseEncoder,
    dynamics_class: BaseDynamics,
    decoder_class: BaseDecoder,
    hidden_dim: int = 64,
    embedding_dim: int = 2,
    context: int = 33,
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
    criterion = RWMLoss(forecast=forecast)

    train_dataset = ContextDataset(
        data_path=train_path,
        context=context,
        forecast=forecast,
    )
    val_dataset = ContextDataset(
        data_path=val_path,
        context=context,
        forecast=forecast,
    )

    config = train_dataset.config

    observable_dim = train_dataset.get_observable_dim()

    print(
        f"Encoder: {encoder_class} \nDynamics: {dynamics_class} \nDecoder: {decoder_class}"
    )

    model = ContextModel(
        encoder=encoder_class(
            observable_dim=observable_dim,
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            context=context,
            config=config,
        ),
        dynamics=dynamics_class(
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            context=context,
            config=config,
        ),
        decoder=decoder_class(
            observable_dim=observable_dim,
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            context=context,
            config=config,
        ),
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

    encoder_class = encoder_dict[args.encoder]
    dynamics_class = dynamics_dict[args.dynamics]
    decoder_class = decoder_dict[args.decoder]

    if any(c is None for c in [encoder_class, dynamics_class, decoder_class]):
        raise ValueError("Invalid model component specified.")

    model = train(
        encoder_class=encoder_class,
        dynamics_class=dynamics_class,
        decoder_class=decoder_class,
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
    create_run_name(args)
    args.training = "rwm"
    try:
        main(args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
