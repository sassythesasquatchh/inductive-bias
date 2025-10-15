import numpy as np
import torch
import wandb

from common.classes import RolloutOutput
from util.config import Config
from util.dataset import TorchTrajectoryDataset
from util.pre_util import calculate_energy, get_length
from util.wandb_util import to_wandb


def evaluate_rollout(
    rollout: RolloutOutput,
    dataset: TorchTrajectoryDataset,
):
    """
    Compute metrics from rollout and log to wandb.
    """

    config = dataset.config
    names = dataset.traj_names

    log_energy(rollout, config, names)
    log_length(rollout, config, names)
    log_error(rollout, names)


def log_energy(rollout: RolloutOutput, config: Config, names: list[str]):
    latent_rollout_energy = (
        calculate_energy(rollout["obs_latent_rollout"], config).detach().cpu().numpy()
    )
    end_to_end_energy = (
        calculate_energy(rollout["obs_end_to_end"], config).detach().cpu().numpy()
    )
    for rollout_method, data in zip(
        ["latent_rollout", "end_to_end"],
        [latent_rollout_energy, end_to_end_energy],
    ):
        times = np.arange(0, data.shape[1])

        for trajectory_name, i in zip(names, range(len(data))):
            energy = data[i]
            energy_table = wandb.Table(
                data=to_wandb(times.tolist(), energy.tolist()),
                columns=["Frame number", "Energy"],
            )
            try:
                plot_name = f"energy_{rollout_method}_{trajectory_name}"
                wandb.log(
                    {
                        plot_name: wandb.plot.line(
                            energy_table,
                            "Frame number",
                            "Energy",
                            title=plot_name,
                        ),
                    },
                )
            except Exception as e:
                print(
                    f"Failed to log {rollout_method} {trajectory_name} energy to wandb: {e}"
                )


def log_length(rollout: RolloutOutput, config: Config, names: list[str]):
    latent_rollout_length = (
        get_length(rollout["obs_latent_rollout"], config).detach().cpu().numpy()
    )
    end_to_end_length = (
        get_length(rollout["obs_end_to_end"], config).detach().cpu().numpy()
    )
    for rollout_method, data in zip(
        ["latent_rollout", "end_to_end"],
        [latent_rollout_length, end_to_end_length],
    ):
        times = np.arange(0, data.shape[1])

        for trajectory_name, i in zip(names, range(len(data))):
            length = data[i]
            length_table = wandb.Table(
                data=to_wandb(times.tolist(), length.tolist()),
                columns=["Frame number", "Length"],
            )
            try:
                plot_name = f"length_{rollout_method}_{trajectory_name}"
                wandb.log(
                    {
                        plot_name: wandb.plot.line(
                            length_table,
                            "Frame number",
                            "Length",
                            title=plot_name,
                        ),
                    },
                )
            except Exception as e:
                print(
                    f"Failed to log {rollout_method} {trajectory_name} length to wandb: {e}"
                )


def log_error(rollout: RolloutOutput, names: list[str]):
    latent_rollout_error = torch.norm(
        rollout["obs_latent_rollout"] - rollout["obs_gt"], dim=-1
    )
    end_to_end_error = torch.norm(rollout["obs_end_to_end"] - rollout["obs_gt"], dim=-1)
    true_obs_norm = torch.norm(rollout["obs_gt"], dim=-1)
    for rollout_method, data in zip(
        ["latent_rollout", "end_to_end"],
        [latent_rollout_error, end_to_end_error],
    ):
        times = np.arange(0, data.shape[1])

        for trajectory_name, i in zip(names, range(len(data))):
            difference = data[i]
            difference /= true_obs_norm[i]
            difference *= 100
            difference = difference.detach().cpu().numpy()
            error_table = wandb.Table(
                data=to_wandb(times.tolist(), difference.tolist()),
                columns=["Frame number", "Relative Error"],
            )
            try:
                plot_name = f"error_{rollout_method}_{trajectory_name}"
                wandb.log(
                    {
                        plot_name: wandb.plot.line(
                            error_table,
                            "Frame number",
                            "Relative Error",
                            title=plot_name,
                        ),
                    },
                )
            except Exception as e:
                print(
                    f"Failed to log {rollout_method} {trajectory_name} error to wandb: {e}"
                )
