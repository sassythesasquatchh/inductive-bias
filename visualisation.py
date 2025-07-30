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
from pre_util import LitModel, get_model, parse_args
from datasets import TrajectoryDataset
import matplotlib.animation as animation
import os
import wandb
import matplotlib
from constants import *
from pre_util import process_trajectory

matplotlib.use("Agg")


def plot_trajectories(
    model: pl.LightningModule,
    dataset: DataLoader,
    args: argparse.Namespace = None,
    model_name: str = None,
    plot_as_phase_portrait: bool = False,
):
    """Plot trajectories of the model on the dataset."""
    model.eval()
    model.freeze()
    device = model.device
    if args is not None:
        model_name = args.model
    for j, trajectory in enumerate(dataset):
        x = trajectory.squeeze(0).to(device).transpose(0, 1)
        z = torch.zeros(x.size(0), 3).to(device)
        for i in range(x.size(0)):
            z[i] = model.model.encoder(x[i].unsqueeze(0))
        plot_latent_trajectory(z.detach().cpu().numpy(), j, model_name)
        if plot_as_phase_portrait:
            plot_latent_as_phase_portrait(z.detach().cpu().numpy(), j, model_name)


def plot_latent_trajectory(z: np.ndarray, i: int, model_name: str):
    x = z[:, 0]
    y = z[:, 1]
    z = z[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, label="3D trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plot_dir = Path(f"data/plots/{model_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"data/plots/{model_name}/latent_trajectory_{i}.png")
    plt.close()


def plot_latent_as_phase_portrait(z: np.ndarray, i: int, model_name: str):
    x = z[:, 0]
    y = z[:, 1]
    theta_dot = z[:, 2]
    theta = np.arctan2(y, x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theta, theta_dot, label="Phase portrait")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Theta_dot")
    plot_dir = Path(f"data/plots/{model_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"data/plots/{model_name}/latent_phase_portrait_{i}.png")


def animate_trajectory(
    folder,
    trajectory,
    reconstructed_traj=None,
    latent_traj=None,
    recon_latent_traj=None,
    name: str = "",
):
    L = 1
    fig = plt.figure(figsize=(12, 6))

    # Setup 2D pendulum plot
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-L * 1.1, L * 1.1)
    ax1.set_ylim(-L * 1.1, L * 1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    title = "Pendulum Motion"
    ax1.set_title(title)

    # Setup 3D latent space plot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Latent Space Trajectory")

    # Calculate max range for 3D plot
    x_max = 1
    y_max = 1
    z_max = 1
    x_min = -1
    y_min = -1
    z_min = -1
    if latent_traj is not None:
        x_max, y_max, z_max = np.max(latent_traj, axis=1)
        x_min, y_min, z_min = np.min(latent_traj, axis=1)
    if recon_latent_traj is not None:
        x_max = max(x_max, np.max(recon_latent_traj[0, :]))
        y_max = max(y_max, np.max(recon_latent_traj[1, :]))
        z_max = max(z_max, np.max(recon_latent_traj[2, :]))

        x_min = min(x_min, np.min(recon_latent_traj[0, :]))
        y_min = min(y_min, np.min(recon_latent_traj[1, :]))
        z_min = min(z_min, np.min(recon_latent_traj[2, :]))

    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    z_padding = 0.1 * (z_max - z_min)

    x_max += x_padding
    x_min -= x_padding
    y_max += y_padding
    y_min -= y_padding
    z_max += z_padding
    z_min -= z_padding

    # Set axis limits
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    # Initialize pendulum plot elements
    p = trajectory.shape[0] // 3
    scat = ax1.scatter(
        [], [], c=[], cmap="viridis", vmin=0, vmax=1, s=50, label="Original (circles)"
    )
    origin_line = ax1.plot([], [], "k--", lw=1)[0]  # Dashed line from origin
    origin_line_recon = ax1.plot([], [], "r--", lw=1)[0]  # Dashed line from origin
    frame_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes)  # Frame counter

    scat_recon = None
    max_vel = np.max(np.abs(trajectory[2 * p : 3 * p]))
    if max_vel == 0:
        max_vel = 1

    # Initialize latent space plot elements
    latent_lines = []
    latent_dots = []
    proj_lines = []

    if reconstructed_traj is not None:
        scat_recon = ax1.scatter(
            [],
            [],
            c=[],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=50,
            marker="s",
            label="Reconstructed (squares)",
        )

    # Plot full latent trajectories
    if latent_traj is not None:
        line = ax2.plot(
            latent_traj[0, 0:1],
            latent_traj[1, 0:1],
            latent_traj[2, 0:1],
            color="blue",
            alpha=0.5,
        )[0]
        latent_lines.append(line)
        dot = ax2.plot([], [], [], "o", color="blue", markersize=8, label="Original")[0]
        latent_dots.append(dot)

    if recon_latent_traj is not None:
        line = ax2.plot(
            recon_latent_traj[0, 0:1],
            recon_latent_traj[1, 0:1],
            recon_latent_traj[2, 0:1],
            color="orange",
            alpha=0.5,
        )[0]
        latent_lines.append(line)
        dot = ax2.plot(
            [], [], [], "s", color="orange", markersize=8, label="Reconstructed"
        )[0]
        latent_dots.append(dot)

    ax1.legend(loc="upper right")

    def update(frame):
        nonlocal proj_lines

        # Clear previous projection lines
        for line in proj_lines:
            line.remove()
        proj_lines = []

        # Update pendulum plot
        x = trajectory[0:p, frame]
        y = trajectory[p : 2 * p, frame]
        linear_vel = np.abs(trajectory[2 * p : 3 * p, frame])
        colors = linear_vel / max_vel
        scat.set_offsets(np.column_stack((x, y)))
        scat.set_array(colors)

        origin_line.set_data([0, x[0]], [0, y[0]])

        # Update frame counter
        frame_text.set_text(f"Frame: {frame}")

        # Update reconstructed trajectory if exists
        if reconstructed_traj is not None:
            x_recon = reconstructed_traj[0:p, frame]
            y_recon = reconstructed_traj[p : 2 * p, frame]
            linear_vel_recon = np.abs(reconstructed_traj[2 * p : 3 * p, frame])
            colors_recon = linear_vel_recon / max_vel
            scat_recon.set_offsets(np.column_stack((x_recon, y_recon)))
            scat_recon.set_array(colors_recon)
            origin_line_recon.set_data([0, x_recon[0]], [0, y_recon[0]])

        # Update latent space visualization
        artists = [scat, origin_line, frame_text]
        if scat_recon is not None:
            artists.append(scat_recon)

        for i, traj in enumerate([latent_traj, recon_latent_traj]):
            if traj is None:
                continue

            # Update trajectory line (whole history)
            latent_lines[i].set_data_3d(
                traj[0, : frame + 1], traj[1, : frame + 1], traj[2, : frame + 1]
            )

            # Update current position dot
            current_pos = traj[:, frame]
            latent_dots[i].set_data_3d(
                [current_pos[0]], [current_pos[1]], [current_pos[2]]
            )

        return artists + latent_dots + latent_lines + proj_lines

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=trajectory.shape[1], interval=50, blit=True
    )

    # Save animation
    filename = f"data/plots/{folder}/{name}"
    os.makedirs(f"data/plots/{folder}", exist_ok=True)

    # Save animation with fallback options
    try:
        # First try MP4 with ffmpeg
        filename += ".mp4"
        ani.save(filename, writer="ffmpeg", fps=int(1 / DT))
        print(f"Animation saved as {filename}")
    except Exception as e:
        print(f"MP4/ffmpeg not available: {e}")
        try:
            # Fallback to GIF with Pillow
            filename += ".gif"
            ani.save(filename, writer="pillow", fps=int(1 / DT))
            print(f"Animation saved as {filename} (GIF format)")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.close(fig)
    return filename


def advect_trajectory(
    model: pl.LightningModule, dataset: DataLoader, args: argparse.Namespace = None
):
    """
    Advect trajectories from the initial point.
    Dataset should be a trajectory dataloader
    """
    model.eval()
    model.freeze()
    device = model.device
    if args is not None:
        model_name = args.model
    for i, trajectory in enumerate(dataset):

        (
            x,
            reconstructed_trajectory,
            latent_trajectory,
            latent_reconstructed_trajectory,
        ) = process_trajectory(model, trajectory, args, device)
        # reconstructed_trajectory = torch.zeros_like(x)
        # latent_trajectory = torch.zeros(x.size(0), args.embedding_dim).to(device)
        # latent_reconstructed_trajectory = torch.zeros_like(latent_trajectory)
        # if not args.training == "rwm":
        #     ic = x[0].unsqueeze(0)
        #     reconstructed_trajectory[0] = ic
        #     upper_bound = x.size(0) - 1
        # else:
        #     x_t = x[: args.context, :].reshape(1, -1)
        #     x_t1 = torch.zeros_like(x_t)
        #     reconstructed_trajectory[: args.context, :] = x[: args.context, :]
        #     upper_bound = x.size(0) - args.context
        #     latent_reconstructed_trajectory[: args.context, :] = model.model.encoder(
        #         x[: args.context, :].reshape(-1)
        #     ).reshape(args.context, args.embedding_dim)
        #     latent_trajectory[: args.context, :] = model.model.encoder(
        #         x[: args.context, :].reshape(-1)
        #     ).reshape(args.context, args.embedding_dim)
        # for j in range(0, upper_bound):

        #     if not args.training == "rwm":
        #         reconstructed_trajectory[j + 1] = model.model(
        #             reconstructed_trajectory[j].unsqueeze(0)
        #         )
        #     else:
        #         # Get the first prediction
        #         z = model.model(x_t)[:, 0, :].squeeze()
        #         reconstructed_trajectory[j + args.context] = z
        #         x_t1[:, : x_t.size(1) - args.observable_dim] = x_t[
        #             :, args.observable_dim :
        #         ]
        #         # Add the new prediction to the end
        #         x_t1[:, -args.observable_dim :] = z
        #         x_t = x_t1.clone()
        #         latent_reconstructed_trajectory[j + args.context] = model.model.encoder(
        #             x_t
        #         ).reshape(args.context, args.embedding_dim)[-1, :]
        #         latent_trajectory[j + args.context] = model.model.encoder(
        #             x[j : j + args.context, :].reshape(1, -1)
        #         ).reshape(args.context, args.embedding_dim)[-1, :]
        # try:
        #     if not args.training == "rwm":
        #         latent_trajectory = model.model.encoder(x)
        #         latent_reconstructed_trajectory = model.model.encoder(
        #             reconstructed_trajectory
        #         )

        filename = animate_trajectory(
            trajectory=x.detach().cpu().numpy().T,
            reconstructed_traj=reconstructed_trajectory.detach().cpu().numpy().T,
            latent_traj=latent_trajectory.detach().cpu().numpy().T,
            recon_latent_traj=latent_reconstructed_trajectory.detach().cpu().numpy().T,
            folder=args.run_name,
            name=f"animation_{i}",
        )
        # except:
        #     filename = animate_trajectory(
        #         trajectory=x.detach().cpu().numpy().T,
        #         reconstructed_traj=reconstructed_trajectory.detach().cpu().numpy().T,
        #         folder=args.run_name,
        #         name=f"animation_{i}",
        #     )

        try:
            wandb.log(
                {
                    "closed_orbit" if i == 0 else "open_orbit": wandb.Video(
                        filename, caption=f"Animation {i}"
                    )
                }
            )
        except Exception as e:
            print(f"Failed to log video to wandb: {e}")


def advect_trajectory_latent_rollout(
    model: pl.LightningModule, dataset: DataLoader, args: argparse.Namespace = None
):
    """
    Advect trajectories from the initial point.
    Dataset should be a trajectory dataloader
    """
    model.eval()
    model.freeze()
    device = model.device
    if args is not None:
        model_name = args.model
    for i, trajectory in enumerate(dataset):
        x = trajectory.squeeze(0).to(device).transpose(0, 1)
        reconstructed_trajectory = torch.zeros_like(x)
        ic = x[0].unsqueeze(0)
        reconstructed_trajectory[0] = ic
        for j in range(0, x.size(0) - 1):
            timestep = j % args.max_k + 1
            ic = reconstructed_trajectory[(j // args.max_k) * args.max_k].unsqueeze(0)
            reconstructed_trajectory[j + 1] = model.model(ic, timestep)

        try:
            latent_trajectory = model.model.encoder(x)
            latent_reconstructed_trajectory = model.model.encoder(
                reconstructed_trajectory
            )

            filename = animate_trajectory(
                trajectory=x.detach().cpu().numpy().T,
                reconstructed_traj=reconstructed_trajectory.detach().cpu().numpy().T,
                latent_traj=latent_trajectory.detach().cpu().numpy().T,
                recon_latent_traj=latent_reconstructed_trajectory.detach()
                .cpu()
                .numpy()
                .T,
                folder=args.run_name,
                name=f"animation_latent_maxk_{i}",
            )

        except Exception as e:
            print(f"Error during animation creation: {e}")
            filename = animate_trajectory(
                trajectory=x.detach().cpu().numpy().T,
                reconstructed_traj=reconstructed_trajectory.detach().cpu().numpy().T,
                folder=args.run_name,
                name=f"animation_latent_maxk_{i}",
            )

        try:
            wandb.log(
                {
                    "closed_orbit" if i == 0 else "open_orbit": wandb.Video(
                        filename, caption=f"Animation {i}"
                    )
                }
            )
        except Exception as e:
            print(f"Failed to log video to wandb: {e}")


if __name__ == "__main__":
    args = parse_args()
    if args.model == "unstructured":
        model_path = "checkpoints/unstructured_3_8_128/4-0.00.ckpt"
    elif args.model == "informed":
        model_path = "checkpoints/informed_3_10_0.0/56-0.00.ckpt"
    elif args.model == "hybrid":
        model_path = "checkpoints/hybrid_3_8_128/6-0.00.ckpt"
    elif args.model == "baseline":
        model_path = "checkpoints/baseline_3_10_0.0/140-0.00.ckpt"
    else:
        raise ValueError("Invalid model name")
    model, criterion = get_model(args)
    model = LitModel.load_from_checkpoint(model_path, model=model, criterion=criterion)
    test_data = TrajectoryDataset(data_path=args.visualisation_data_path, split="test")
    test_loader = DataLoader(test_data, batch_size=1, num_workers=8)

    LOG = False

    if LOG:
        from wandb_util import get_run_id

        run_id = get_run_id(
            os.getenv("WANDB_PROJECT"),
            args.run_name,
        )
        # run_id = "pnnq6tsk"

        run = wandb.init(project=os.getenv("WANDB_PROJECT"), id=run_id, resume="must")

    try:
        if args.end2end:
            advect_trajectory(model, test_loader, args)
        else:
            advect_trajectory_latent_rollout(model, test_loader, args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
    # advect_trajectory(model, test_loader, args)
