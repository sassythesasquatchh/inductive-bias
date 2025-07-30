import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models import *
from typing import Union
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import ipdb
from pre_util import LitModel, get_model, parse_args, process_trajectory
from datasets import TrajectoryDataset
import matplotlib.animation as animation
import wandb
import os
from dotenv import load_dotenv

load_dotenv()


def calculate_metric_numpy(z, z_prime):
    """
    Calculates a symmetric distance metric between two latent trajectories.

    For each trajectory, sum the distance between each point and its closest
    neighbor in the other trajectory. The final metric is the sum of the two
    summed distances.

    Args:
        z: np.ndarray, shape (n_points, latent_dim)
        z_prime: np.ndarray, shape (m_points, latent_dim)

    Returns:
        float: scalar distance metric
    """
    # Compute pairwise distances between all points in z and z_prime
    distances = np.linalg.norm(z[:, np.newaxis] - z_prime[np.newaxis, :], axis=2)

    # For each point in z, find the minimum distance to any point in z_prime
    min_z_to_zprime = distances.min(axis=1)

    # For each point in z_prime, find the minimum distance to any point in z
    min_zprime_to_z = distances.min(axis=0)

    # Sum all the minimum distances
    total_distance = min_z_to_zprime.sum() + min_zprime_to_z.sum()

    return total_distance / (2 * len(z))


def test_continuity(model, loader, args):
    model.eval()
    model.freeze()
    device = model.device
    model_name = args.model
    last_z = None
    initial_velocities = np.linspace(
        0.5,
        12,
        len(loader.dataset),
    )
    dv = initial_velocities[1] - initial_velocities[0]
    derivatives = []
    norms = []
    l2_distances = []
    for trajectory in loader:
        # x = trajectory.squeeze(0).to(device).transpose(0, 1)
        # ipdb.set_trace()
        _, _, z, _ = process_trajectory(model, trajectory, args, device)
        z = z.detach().cpu().numpy()
        if last_z is not None:
            distance = calculate_metric_numpy(last_z, z)
            derivatives.append(distance / dv)
            l2_distances.append(np.linalg.norm(z - last_z))
        norms.append(np.linalg.norm(z))
        last_z = z

    deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), derivatives),
        columns=["Initial Velocity", "Difference Quotient"],
    )
    l2_deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), l2_distances),
        columns=["Initial Velocity", "L2 Distance"],
    )
    norm_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), norms),
        columns=["Initial Velocity", "Norm"],
    )

    try:
        wandb.log(
            {
                "derivative": wandb.plot.line(
                    deriv_table,
                    "Initial Velocity",
                    "Difference Quotient",
                    title="Continuity - Custom Metric",
                )
            }
        )

        wandb.log(
            {
                "l2_distance": wandb.plot.line(
                    l2_deriv_table,
                    "Initial Velocity",
                    "L2 Distance",
                    title="Continuity - L2 Metric",
                )
            }
        )

        wandb.log(
            {
                "norm": wandb.plot.line(
                    norm_table, "Initial Velocity", "Norm", title="Norm"
                )
            }
        )
    except Exception as e:
        print(f"Error logging to wandb: {e}")

    # plot_continuity(initial_velocities[1:].tolist(), derivatives, model_name)


def test_continuity_canonical(loader):
    last_z = None
    initial_velocities = np.linspace(
        0.5,
        12,
        len(loader.dataset),
    )
    dv = initial_velocities[1] - initial_velocities[0]
    derivatives = []
    norms = []
    l2_distances = []
    for trajectory in loader:
        trajectory = trajectory.squeeze(0).transpose(0, 1)
        z = torch.zeros(trajectory.shape[0], 3)
        z[:, 0] = torch.cos(trajectory[:, 0])
        z[:, 1] = torch.sin(trajectory[:, 0])
        z[:, 2] = trajectory[:, 1]
        if last_z is not None:
            distance = calculate_metric_numpy(last_z, z)
            derivatives.append(distance / dv)
            l2_distances.append(np.linalg.norm(z - last_z))
        norms.append(np.linalg.norm(z))
        last_z = z

    deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), derivatives),
        columns=["Initial Velocity", "Difference Quotient"],
    )

    wandb.log(
        {
            "derivative": wandb.plot.line(
                deriv_table, "Initial Velocity", "Difference Quotient"
            )
        }
    )

    l2_deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), l2_distances),
        columns=["Initial Velocity", "L2 Distance"],
    )
    wandb.log(
        {
            "l2_distance": wandb.plot.line(
                l2_deriv_table, "Initial Velocity", "L2 Distance"
            )
        }
    )

    norm_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), norms),
        columns=["Initial Velocity", "Norm"],
    )
    wandb.log({"norm": wandb.plot.line(norm_table, "Initial Velocity", "Norm")})


def test_continuity_observed(loader):
    last_traj = None
    initial_velocities = np.linspace(
        0.5,
        12,
        len(loader.dataset),
    )
    dv = initial_velocities[1] - initial_velocities[0]
    derivatives = []
    norms = []
    l2_distances = []
    for trajectory in loader:
        trajectory = trajectory.squeeze(0).transpose(0, 1)
        if last_traj is not None:
            distance = calculate_metric_numpy(last_traj, trajectory)
            derivatives.append(distance / dv)
            l2_distances.append(np.linalg.norm(trajectory - last_traj))
        norms.append(np.linalg.norm(trajectory))
        last_traj = trajectory

    deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), derivatives),
        columns=["Initial Velocity", "Difference Quotient"],
    )

    wandb.log(
        {
            "derivative": wandb.plot.line(
                deriv_table, "Initial Velocity", "Difference Quotient"
            )
        }
    )

    l2_deriv_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), l2_distances),
        columns=["Initial Velocity", "L2 Distance"],
    )
    wandb.log(
        {
            "l2_distance": wandb.plot.line(
                l2_deriv_table, "Initial Velocity", "L2 Distance"
            )
        }
    )

    norm_table = wandb.Table(
        data=to_wandb(initial_velocities[1:].tolist(), norms),
        columns=["Initial Velocity", "Norm"],
    )
    wandb.log({"norm": wandb.plot.line(norm_table, "Initial Velocity", "Norm")})


def to_wandb(_x: list, _y: list):
    return [[x, y] for x, y in zip(_x, _y)]


def plot_continuity(velocities, derivatives, model_name):
    plt.plot(velocities, derivatives)
    plt.xlabel("Initial velocity")
    plt.ylabel("Difference quotient")
    plt.title(f"Derivative")
    plt.savefig(f"data/plots/{model_name}/derivative.png")


def get_run_id(project_name, run_name):
    """Gets the run ID of an existing W&B run."""

    api = wandb.Api()

    try:
        runs = api.runs(project_name, filters={"displayName": run_name})

        if not runs:
            print(f"Run '{run_name}' not found in project '{project_name}'.")
            return None

        if len(runs) > 1:
            print(
                f"Warning: Multiple runs with name '{run_name}' found. Returning the id of the most recent one."
            )

        run_id = runs[0].id
        return run_id

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    args = parse_args()
    if args.model == "unstructured":
        model_path = "checkpoints/unstructured/18-0.09.ckpt"
    elif args.model == "informed":
        model_path = "checkpoints/informed/99-0.00-v1.ckpt"
    elif args.model == "hybrid":
        model_path = "checkpoints/hybrid/97-0.00.ckpt"
    else:
        raise ValueError("Invalid model name")

    run_id = get_run_id(
        os.getenv("WANDB_PROJECT"),
        args.model,
    )

    run = wandb.init(project=os.getenv("WANDB_PROJECT"), id=run_id, resume="must")

    model, criterion = get_model(args)
    model = LitModel.load_from_checkpoint(model_path, model=model, criterion=criterion)
    test_data = TrajectoryDataset(
        data_path="data/continuity_test_data.pkl", split="test"
    )
    test_loader = DataLoader(test_data, batch_size=1)

    test_continuity(model, test_loader, args)


def canonical():
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name="canonical",
        tags=["canonical"],
    )
    test_data = TrajectoryDataset(
        data_path="data/continuity_test_data.pkl", split="test", type="phase"
    )
    test_loader = DataLoader(test_data, batch_size=1)
    test_continuity_canonical(test_loader)


def observed():
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name="observed",
        tags=["observed"],
    )
    test_data = TrajectoryDataset(
        data_path="data/continuity_test_data.pkl", split="test", type="observed"
    )
    test_loader = DataLoader(test_data, batch_size=1)
    test_continuity_observed(test_loader)


if __name__ == "__main__":
    try:
        # main()
        # canonical()
        observed()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
    finally:
        wandb.finish()
