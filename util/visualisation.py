import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import wandb

from common.classes import RolloutOutput
from util.config import Config

matplotlib.use("Agg")


def animate_trajectory(
    folder,
    trajectory,
    reconstructed_traj=None,
    latent_traj=None,
    recon_latent_traj=None,
    name: str = "",
    config: Config = Config(),
):
    L = config.L
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
        fname = filename + ".mp4"
        ani.save(fname, writer="ffmpeg", fps=int(1 / config.DT))
        print(f"Animation saved as {fname}")
    except Exception as e:
        print(f"MP4/ffmpeg not available: {e}")
        try:
            # Fallback to GIF with Pillow
            fname = filename + ".gif"
            # Note: Framerate for gifs is not reliable
            ani.save(fname, writer="pillow", fps=int(1 / config.DT))
            print(f"Animation saved as {fname} (GIF format)")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.close(fig)
    return fname


def animate_trajectories(
    rollout: RolloutOutput, config: Config, names: list[str], folder_name: str
):
    """
    Names: List of trajectory names.
    """
    for i, name in zip(range(len(rollout["obs_gt"])), names):
        trajectory = rollout["obs_gt"][i].detach().cpu().numpy().T
        reconstructed_trajectory = rollout["obs_end_to_end"][i].detach().cpu().numpy().T
        latent_trajectory = rollout["latent_gt"][i].detach().cpu().numpy().T
        latent_reconstructed_trajectory = (
            rollout["latent_end_to_end"][i].detach().cpu().numpy().T
        )

        if latent_trajectory.shape[0] > 3:
            print(f"Skipping {name} latent animation as latent dim > 3")
            latent_trajectory = None
            latent_reconstructed_trajectory = None
        if latent_trajectory.shape[0] == 2:
            latent_trajectory = np.vstack(
                (latent_trajectory, np.zeros(latent_trajectory.shape[1]))
            )
            latent_reconstructed_trajectory = np.vstack(
                (
                    latent_reconstructed_trajectory,
                    np.zeros(latent_reconstructed_trajectory.shape[1]),
                )
            )

        video_name = f"animation_end_to_end_{name}"
        filename = animate_trajectory(
            trajectory=trajectory,
            reconstructed_traj=reconstructed_trajectory,
            latent_traj=latent_trajectory,
            recon_latent_traj=latent_reconstructed_trajectory,
            folder=folder_name,
            name=video_name,
            config=config,
        )

        try:
            wandb.log({video_name: wandb.Video(filename, caption=video_name)})
        except Exception as e:
            print(f"Failed to log video to wandb: {e}")

    for i, name in zip(range(len(rollout["obs_gt"])), names):
        trajectory = rollout["obs_gt"][i].detach().cpu().numpy().T
        reconstructed_trajectory = (
            rollout["obs_latent_rollout"][i].detach().cpu().numpy().T
        )
        latent_trajectory = rollout["latent_gt"][i].detach().cpu().numpy().T
        latent_reconstructed_trajectory = (
            rollout["latent_rollout"][i].detach().cpu().numpy().T
        )

        if latent_trajectory.shape[0] > 3:
            print(f"Skipping {name} latent animation as latent dim > 3")
            latent_trajectory = None
            latent_reconstructed_trajectory = None
        if latent_trajectory.shape[0] == 2:
            latent_trajectory = np.vstack(
                (latent_trajectory, np.zeros(latent_trajectory.shape[1]))
            )
            latent_reconstructed_trajectory = np.vstack(
                (
                    latent_reconstructed_trajectory,
                    np.zeros(latent_reconstructed_trajectory.shape[1]),
                )
            )

        video_name = f"animation_latent_rollout_{name}"
        filename = animate_trajectory(
            trajectory=trajectory,
            reconstructed_traj=reconstructed_trajectory,
            latent_traj=latent_trajectory,
            recon_latent_traj=latent_reconstructed_trajectory,
            folder=folder_name,
            name=video_name,
            config=config,
        )

        try:
            wandb.log(
                {
                    video_name: wandb.Video(
                        filename, caption=video_name, format=filename.split(".")[-1]
                    )
                }
            )
        except Exception as e:
            print(f"Failed to log video to wandb: {e}")
