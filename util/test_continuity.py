import numpy as np
import torch
from dotenv import load_dotenv
from jaxtyping import Float

import wandb
from common.classes import RolloutOutput
from util.wandb_util import to_wandb

load_dotenv()


def test_continuity(
    rollout: RolloutOutput, initial_velocities: Float[torch.Tensor, "batch"]
):
    latent_trajectories = rollout["latent_gt"]

    # Pairwise distances for all points between consecutive trajectories
    # B-1 x N x N
    distances = torch.norm(
        latent_trajectories[1:, np.newaxis, :, :]
        - latent_trajectories[:-1, :, np.newaxis, :],
        dim=-1,
    )

    # For each point in z_t, find the minimum distance to any point in z_t+1
    # B-1 x N
    min_z_to_zprime = distances.min(dim=2).values

    # For each point in z_t+1, find the minimum distance to any point in z_t
    # B-1 x N
    min_zprime_to_z = distances.min(dim=1).values

    # Sum all the minimum distances
    total_distance = min_z_to_zprime.sum(dim=1) + min_zprime_to_z.sum(dim=1)

    # Normalise by 2*N
    d_traj = total_distance / (2 * latent_trajectories.size(1))

    # NOTE to properly calculate the derivative, we would need to know the change in initial condition
    # However, if, as assumed here, the initial conditions are evenly spaced, then this is just a constant scaling factor
    # Therefore, we can still use the difference quotient as a measure of continuity

    l2_differences = torch.norm(
        latent_trajectories[1:] - latent_trajectories[:-1], dim=(1, 2)
    )
    l2_norms = torch.norm(latent_trajectories, dim=(1, 2))

    deriv_table = wandb.Table(
        data=to_wandb(
            initial_velocities[1:].detach().cpu().numpy().tolist(),
            d_traj.detach().cpu().numpy().tolist(),
        ),
        columns=["initial-velocity", "difference-quotient"],
    )

    l2_deriv_table = wandb.Table(
        data=to_wandb(
            initial_velocities[1:].detach().cpu().numpy().tolist(),
            l2_differences.detach().cpu().numpy().tolist(),
        ),
        columns=["initial-velocity", "l2-distance"],
    )

    norm_table = wandb.Table(
        data=to_wandb(
            initial_velocities.detach().cpu().numpy().tolist(),
            l2_norms.detach().cpu().numpy().tolist(),
        ),
        columns=["initial-velocity", "norm"],
    )

    try:
        wandb.log(
            {
                "derivative": wandb.plot.line(
                    deriv_table,
                    "initial-velocity",
                    "difference-quotient",
                    title="Continuity - Custom Metric",
                )
            }
        )

        wandb.log(
            {
                "l2_distance": wandb.plot.line(
                    l2_deriv_table,
                    "initial-velocity",
                    "l2-distance",
                    title="Continuity - L2 Metric",
                )
            }
        )

        wandb.log(
            {
                "norm": wandb.plot.line(
                    norm_table, "initial-velocity", "norm", title="Norm"
                )
            }
        )
    except Exception as e:
        print(f"Error logging to wandb: {e}")
