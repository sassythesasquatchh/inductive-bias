import argparse

import numpy as np
import torch
from jaxtyping import Float

from util.config import Config


def calculate_energy(
    trajectory: Float[torch.Tensor, "batch N obs"], config: Config
) -> Float[torch.Tensor, "batch N"]:
    # Assumes 12D representation where the first 8 are positions and the last 4 and velocities
    # Assumes also unit mass
    X = trajectory[:, :, :4]
    Y = trajectory[:, :, 4:8]
    vel = trajectory[:, :, 8:12]

    theta_from_equilibrium = torch.atan2(Y, X) + np.pi / 2
    theta = torch.mean(theta_from_equilibrium, dim=-1)  # Average over sampled points
    height = config.L * (torch.ones_like(theta) - torch.cos(theta))
    ang_vel = vel / torch.tensor(config.SAMPLING_POSITIONS).to(vel.device)
    ang_vel = torch.mean(ang_vel, dim=-1)
    linear_velocity = ang_vel * config.L
    energy = 0.5 * (linear_velocity**2) + config.GRAVITY * height
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
    ).mean(dim=-1)
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
    parser.add_argument("--val_path", type=str, default="data/validation_100.pkl")
    parser.add_argument(
        "--visualisation_data_path",
        type=str,
        default="data/visualisation.pkl",
    )
    parser.add_argument(
        "--continuity_data_path", type=str, default="data/continuity_test.pkl"
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)

    # Model config
    # parser.add_argument("--model", type=str, default="unstructured")
    parser.add_argument("--observable_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=32)
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
    parser.add_argument("--noise", type=float, default=0.001)

    # Optimization config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # RWM config
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--forecast", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.9)

    # Loss config
    parser.add_argument("--supervise_rollout", action="store_true")
    parser.add_argument("--supervise_end_to_end", action="store_true")
    parser.add_argument("--penalise_latent_magnitude", action="store_true")
    parser.add_argument("--penalise_latent_mismatch", action="store_true")
    parser.add_argument("--penalise_latent_dynamics", action="store_true")

    args = parser.parse_args()

    return args
