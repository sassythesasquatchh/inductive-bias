import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from constants import *
import argparse
import ipdb

PLOT = True

# Constants
# m = 1  # mass of pendulum (kg)
# L = 1  # length of pendulum (m)
# g = 1  # gravitational acceleration (m/s^2)


def pendulum_ode(t, y):
    theta, theta_dot = y
    dydt = [theta_dot, -(GRAVITY / L) * np.sin(theta)]
    return dydt


def velocity_verlet(pendulum_ode, y0, t_eval):
    dt = t_eval[1] - t_eval[0]
    y = np.zeros((len(y0), len(t_eval)))
    y[:, 0] = y0

    theta, theta_dot = y0
    for i in range(1, len(t_eval)):
        # Update position (theta) using current velocity
        theta += theta_dot * dt

        # Compute acceleration at the new position
        _, theta_ddot = pendulum_ode(t_eval[i], [theta, theta_dot])

        # Update velocity using the new acceleration
        theta_dot += theta_ddot * dt

        # Store the updated state
        y[:, i] = [theta, theta_dot]

    return y, t_eval


def simulate_trajectory(initial_conditions, t_eval):
    y, t = velocity_verlet(pendulum_ode, initial_conditions, t_eval)
    return y, t


def plot_phase_space(phase_space_traj):
    plt.figure(figsize=(6, 6))
    plt.plot(phase_space_traj[0], phase_space_traj[1])
    plt.xlabel("Angular Position (theta)")
    plt.ylabel("Angular Velocity (theta_dot)")
    plt.title("Phase Space Trajectory")
    plt.grid(True)
    filename = f"data/plots/ics/phase_theta_{phase_space_traj[0,0]:.2f}_vel_{phase_space_traj[1,0]:.2f}.png"
    plt.savefig(filename)
    plt.close()


def animate_trajectory(trajectory, initial_theta, initial_theta_dot):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-L * 1.1, L * 1.1)
    ax.set_ylim(-L * 1.1, L * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pendulum Motion with Velocity Coloring")

    p = trajectory.shape[0] // 3
    scat = ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=1, s=50)
    max_vel = np.max(np.abs(trajectory[2 * p : 3 * p, :]))
    if max_vel == 0:
        max_vel = 1  # Avoid division by zero

    def update(frame):
        x = trajectory[0:p, frame]
        y = trajectory[p : 2 * p, frame]
        linear_vel = np.abs(trajectory[2 * p : 3 * p, frame])
        colors = linear_vel / max_vel
        scat.set_offsets(np.column_stack((x, y)))
        scat.set_array(colors)
        return (scat,)

    ani = animation.FuncAnimation(
        fig, update, frames=trajectory.shape[1], interval=50, blit=True
    )
    filename = f"animation_theta_{initial_theta:.2f}_vel_{initial_theta_dot:.2f}.mp4"
    try:
        ani.save(filename, writer="ffmpeg", fps=30)
    except Exception as e:
        print(f"Error saving animation: {e}")
    plt.close(fig)


def get_ranges(option: str, num_trajectories: int, only_small: bool = False):

    if not only_small:
        if option == "normal_training":
            range_1 = jnp.linspace(0.5, 3, int(num_trajectories * 2.5 / 9.5))
            range_2 = jnp.linspace(4, 8, int(num_trajectories * 4 / 9.5))
            range_3 = jnp.linspace(9, 12, int(num_trajectories * 3 / 9.5))
            theta_dot0_vals = jnp.concatenate((range_1, range_2, range_3))
        elif option == "validation":
            range_1 = jnp.linspace(3, 3.4, num_trajectories // 4)
            range_2 = jnp.linspace(3.6, 4, num_trajectories // 4)
            range_3 = jnp.linspace(8, 8.4, num_trajectories // 4)
            range_4 = jnp.linspace(8.6, 9, num_trajectories // 4)
            theta_dot0_vals = jnp.concatenate((range_1, range_2, range_3, range_4))
        elif option == "testing":
            theta_dot0_vals = jnp.array([3.5, 8.5])
        elif option == "sparse_training":
            range_1 = jnp.linspace(0.5, 1, int(num_trajectories * 2.5 / 9.5))
            range_2 = jnp.linspace(6, 7, int(num_trajectories * 4 / 9.5))
            range_3 = jnp.linspace(11.5, 12, int(num_trajectories * 3 / 9.5))
            theta_dot0_vals = jnp.concatenate((range_1, range_2, range_3))

    else:
        if option == "normal_training":
            theta_dot0_vals = jnp.linspace(0.5, 3, num_trajectories)
        elif option == "validation":
            theta_dot0_vals = jnp.linspace(3, 4, num_trajectories)
        elif option == "testing":
            theta_dot0_vals = jnp.array([3.5])
        elif option == "sparse_training":
            theta_dot0_vals = jnp.linspace(0.5, 1, num_trajectories)
    
    return theta_dot0_vals


def generate_trajectories(args, t_eval=np.linspace(0, TIMESPAN, NUM_SAMPLES)):
    trajectories = []
    initial_conditions_list = []

    theta_dot0_vals = get_ranges(args.option, args.num_trajectories, args.only_small)

    for theta_dot0 in theta_dot0_vals:
        initial_conditions_list.append((0, theta_dot0))

    # Sample points along the pendulum
    sampling_positions = L * np.array(SAMPLING_POSITIONS)

    for i, (theta0, theta_dot0) in enumerate(initial_conditions_list):
        y, t = simulate_trajectory([theta0, theta_dot0], t_eval)

        theta = y[0]
        theta_dot = y[1]

        # Calculate observables
        x = sampling_positions.reshape(-1, 1) * np.sin(theta)
        y = -sampling_positions.reshape(-1, 1) * np.cos(theta)
        linear_velocity = sampling_positions.reshape(-1, 1) * theta_dot

        canonical = np.vstack((theta, theta_dot))
        observables = np.vstack((x, y, linear_velocity))

        if args.noise > 0:
            noise = np.random.normal(0, args.noise, size=observables.shape)
            noise[:4, :] = sampling_positions.reshape(-1, 1) * noise[:4, :]
            noise[4:8, :] = -sampling_positions.reshape(-1, 1) * noise[4:8, :]
            noise[8:12, :] = sampling_positions.reshape(-1, 1) * noise[8:12, :]
            observables += noise

        # Create DataFrame for current trajectory
        trajectories.append(
            {
                "phase": canonical,
                "observed": observables,
            }
        )

        if PLOT:
            if i % 10 == 0:
                plot_phase_space(canonical)
                # animate_trajectory(observables, theta0, theta_dot0)

    # file_path = "data/visualisation_trajectories_single.pkl"
    filename = f"{args.option}"
    if args.option not in ["validation", "testing"]:
        filename += f"_{args.num_trajectories}"
        if args.noise > 0:
            filename += f"_{args.noise}"
    file_path = f"data/{filename}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pendulum trajectories.")
    parser.add_argument("--option", type=str, choices=["normal_training", "validation", "testing", "sparse_training"], required=True, help="Type of trajectory generation.")
    parser.add_argument("--num_trajectories", type=int, default=1000, help="Number of trajectories to generate.")
    parser.add_argument("--noise", type=float, default=0.0, help="Amount of noise to add to the observations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--only-small", action="store_true", help="Generate only small trajectories.")

    args = parser.parse_args()
    generate_trajectories(args)
