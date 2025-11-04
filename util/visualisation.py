import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

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


def pca_to_3d(rollout: RolloutOutput) -> RolloutOutput:
    B, C, D = rollout["latent_gt"].shape
    x_np = (
        torch.cat(
            [
                rollout["latent_gt"],
                rollout["latent_end_to_end"],
                rollout["latent_rollout"],
            ],
            dim=0,
        )
        .cpu()
        .numpy()
    )
    x_np = x_np.reshape(3 * B * C, D)

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_np)
    x_pca = x_pca.reshape(3 * B, C, 3)

    rollout["latent_gt"] = torch.tensor(
        x_pca[0:B, :, :],
        device=rollout["latent_gt"].device,
        dtype=rollout["latent_gt"].dtype,
    )
    rollout["latent_end_to_end"] = torch.tensor(
        x_pca[B : 2 * B, :, :],
        device=rollout["latent_end_to_end"].device,
        dtype=rollout["latent_end_to_end"].dtype,
    )
    rollout["latent_rollout"] = torch.tensor(
        x_pca[2 * B : 3 * B, :, :],
        device=rollout["latent_rollout"].device,
        dtype=rollout["latent_rollout"].dtype,
    )
    return rollout


def animate_trajectories(
    rollout: RolloutOutput, config: Config, names: list[str], folder_name: str
):
    """
    Names: List of trajectory names.
    """

    latent_dim = rollout["latent_gt"].shape[2]
    if latent_dim > 3:
        rollout = pca_to_3d(rollout)

    for i, name in zip(range(len(rollout["obs_gt"])), names):
        trajectory = rollout["obs_gt"][i].detach().cpu().numpy().T
        reconstructed_trajectory = rollout["obs_end_to_end"][i].detach().cpu().numpy().T

        latent_trajectory = rollout["latent_gt"][i].detach().cpu().numpy().T
        latent_reconstructed_trajectory = (
            rollout["latent_end_to_end"][i].detach().cpu().numpy().T
        )

        if latent_trajectory is not None and latent_trajectory.shape[0] > 3:
            print(f"Skipping {name} latent animation as latent dim > 3")
            latent_trajectory = None
            latent_reconstructed_trajectory = None
        if latent_trajectory is not None and latent_trajectory.shape[0] == 2:
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
            wandb.log(
                {
                    video_name: wandb.Video(
                        filename, caption=video_name, format=filename.split(".")[-1]
                    )
                }
            )
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

        if latent_trajectory is not None and latent_trajectory.shape[0] > 3:
            print(f"Skipping {name} latent animation as latent dim > 3")
            latent_trajectory = None
            latent_reconstructed_trajectory = None
        if latent_trajectory is not None and latent_trajectory.shape[0] == 2:
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


def visualise_latent_space(rollout: RolloutOutput, initial_velocities: torch.Tensor):
    from plotly import graph_objects as go
    # Visualise continuity of embedding

    embedding_latent_trajectories = rollout["latent_gt"]
    rollout_latent_trajectories = rollout["latent_rollout"]

    latent_dim = embedding_latent_trajectories.shape[2]

    def pad_to_3d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[2] == 2:
            return torch.cat(
                [
                    tensor,
                    torch.zeros(
                        tensor.shape[0],
                        tensor.shape[1],
                        1,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    ),
                ],
                dim=2,
            )
        return tensor

    if latent_dim == 2:
        # Pad to 3D for visualisation
        embedding_latent_trajectories = pad_to_3d(embedding_latent_trajectories)

        rollout_latent_trajectories = pad_to_3d(rollout_latent_trajectories)

    elif latent_dim > 3:
        # Use PCA to reduce to 3D for visualisation
        rollout = pca_to_3d(rollout)
        embedding_latent_trajectories = rollout["latent_gt"]
        rollout_latent_trajectories = rollout["latent_rollout"]

    STEP = 3
    embedding_latent_trajectories_plot = (
        embedding_latent_trajectories.detach().cpu().numpy()
    )
    embedding_latent_trajectories_plot = embedding_latent_trajectories_plot[::STEP]
    initial_velocities = initial_velocities[::STEP]

    fig = go.Figure()
    for i in range(len(embedding_latent_trajectories_plot)):
        traj = embedding_latent_trajectories_plot[i]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(
                    width=2,
                    color=np.full(traj.shape[0], initial_velocities[i].item()),
                    colorscale="bluered",
                    cmin=min(initial_velocities).item(),
                    cmax=max(initial_velocities).item(),
                ),
            )
        )

    fig.update_layout(showlegend=False)

    # fig.show()

    try:
        wandb.log({"embedding_continuity": fig})
    except Exception as e:
        print(f"Error logging embedding continuity to wandb: {e}")

    # Latent space phase portrait

    rollout_latent_trajectories_plot = (
        rollout_latent_trajectories.detach().cpu().numpy()
    )
    rollout_latent_trajectories_plot = rollout_latent_trajectories_plot[::STEP]

    fig = go.Figure()
    Xs, Ys, Zs = [], [], []
    Us, Vs, Ws = [], [], []

    step = 12  # sampling stride along each trajectory
    target_scale = 0.05  # scale factor relative to scene size

    for traj in rollout_latent_trajectories_plot:
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        u, v, w = np.gradient(x), np.gradient(y), np.gradient(z)

        # normalize directions
        mag = np.sqrt(u**2 + v**2 + w**2)
        norm = np.where(mag == 0, 1, mag)
        u, v, w = u / norm, v / norm, w / norm

        # trajectory line
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=2, color="blue"),
                showlegend=False,
            )
        )

        # sample along the curve
        idx = slice(0, len(x), step)
        Xs.extend(x[idx])
        Ys.extend(y[idx])
        Zs.extend(z[idx])
        Us.extend(u[idx])
        Vs.extend(v[idx])
        Ws.extend(w[idx])

    # compute scene scale for cone size
    all_points = np.vstack([Xs, Ys, Zs])
    scene_range = np.ptp(all_points, axis=1).max()  # overall plot scale
    target_len = target_scale * scene_range

    cone_trace = go.Cone(
        x=np.array(Xs),
        y=np.array(Ys),
        z=np.array(Zs),
        u=np.array(Us),
        v=np.array(Vs),
        w=np.array(Ws),
        sizemode="absolute",
        sizeref=target_len,
        anchor="tail",
        colorscale="bluered",
        showscale=False,
        visible=True,  # default: visible
    )

    fig.add_trace(cone_trace)

    # add toggle button for cones
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Show Arrows",
                        method="update",
                        args=[
                            {
                                "visible": [True]
                                * (len(rollout_latent_trajectories_plot) + 1)
                            }
                        ],
                    ),
                    dict(
                        label="Hide Arrows",
                        method="update",
                        args=[
                            {
                                "visible": [True]
                                * len(rollout_latent_trajectories_plot)
                                + [False]
                            }
                        ],
                    ),
                ],
                x=0.02,
                y=1.05,
                xanchor="left",
                yanchor="top",
            )
        ],
        showlegend=False,
        # title="Latent Space Phase Portrait",
    )

    # fig.show()

    try:
        wandb.log({"latent_phase_portrait": fig})
    except Exception as e:
        print(f"Error logging latent phase portrait to wandb: {e}")

    # Combined plot

    fig = go.Figure()
    for i in range(len(embedding_latent_trajectories_plot)):
        embedding_traj = embedding_latent_trajectories_plot[i]
        fig.add_trace(
            go.Scatter3d(
                x=embedding_traj[:, 0],
                y=embedding_traj[:, 1],
                z=embedding_traj[:, 2],
                mode="lines",
                line=dict(width=2, color="gray"),
                name="Embedding" if i == 0 else None,
                showlegend=(i == 0),
            )
        )
        rollout_traj = rollout_latent_trajectories_plot[i]
        fig.add_trace(
            go.Scatter3d(
                x=rollout_traj[:, 0],
                y=rollout_traj[:, 1],
                z=rollout_traj[:, 2],
                mode="lines",
                line=dict(width=2, color="red"),
                name="Phase Portrait" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

    # fig.show()
    try:
        wandb.log({"structure_overlay": fig})
    except Exception as e:
        print(f"Error logging structure overlay to wandb: {e}")

    # Log latent alignment
    distances = torch.norm(
        rollout_latent_trajectories.reshape(1, -1, 3)
        - embedding_latent_trajectories.reshape(-1, 1, 3),
        dim=-1,
    )

    chamfer_distance = torch.mean(torch.min(distances, dim=1).values) + torch.mean(
        torch.min(distances, dim=0).values
    )

    norm = torch.mean(
        torch.norm(rollout_latent_trajectories.reshape(-1, 3), dim=-1)
    ) + torch.mean(torch.norm(embedding_latent_trajectories.reshape(-1, 3), dim=-1))
    chamfer_distance = chamfer_distance / norm

    try:
        wandb.log({"latent_alignment": chamfer_distance.item()})
    except Exception as e:
        print(f"Error logging latent alignment to wandb: {e}")
