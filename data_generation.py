import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from constants import *

PLOT = False

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


def simulate_trajectory(initial_conditions, t_span, t_eval):
    # solution = solve_ivp(pendulum_ode, t_span, initial_conditions, t_eval=t_eval)
    y, t = velocity_verlet(pendulum_ode, initial_conditions, t_eval)
    return y, t


def plot_phase_space(phase_space_traj):
    plt.figure(figsize=(6, 6))
    plt.plot(phase_space_traj[0], phase_space_traj[1])
    plt.xlabel("Angular Position (theta)")
    plt.ylabel("Angular Velocity (theta_dot)")
    plt.title("Phase Space Trajectory")
    plt.grid(True)
    filename = (
        f"phase_theta_{phase_space_traj[0,0]:.2f}_vel_{phase_space_traj[1,0]:.2f}.png"
    )
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


def generate_trajectories(
    num_trajectories, t_eval=np.linspace(0, TIMESPAN, NUM_SAMPLES)
):
    trajectories = []
    initial_conditions_list = []

    # Generate initial conditions
    theta0_vals = np.linspace(-np.pi, np.pi, int(np.sqrt(num_trajectories)))
    theta_dot0_vals = np.linspace(-10, 10, int(np.sqrt(num_trajectories)))
    for theta0 in theta0_vals:
        for theta_dot0 in theta_dot0_vals:
            initial_conditions_list.append((theta0, theta_dot0))

    # Sample points along the pendulum
    sampling_positions = L * np.array(SAMPLING_POSITIONS)
    num_samples = len(sampling_positions)

    for theta0, theta_dot0 in initial_conditions_list:
        y, t = simulate_trajectory(
            [theta0, theta_dot0], (t_eval[0], t_eval[-1]), t_eval
        )

        theta = y[0]
        theta_dot = y[1]
        time = t

        # Calculate observables
        x = sampling_positions.reshape(-1, 1) * np.sin(theta)
        y = -sampling_positions.reshape(-1, 1) * np.cos(theta)
        linear_velocity = sampling_positions.reshape(-1, 1) * theta_dot

        canonical = np.vstack((theta, theta_dot))
        observables = np.vstack((x, y, linear_velocity))

        # Create DataFrame for current trajectory
        trajectories.append(
            {
                "phase": canonical,
                "observed": observables,
            }
        )

        if PLOT:
            phase_space = np.vstack((theta, theta_dot))
            plot_phase_space(phase_space)
            observables = np.vstack((x, y, linear_velocity))
            animate_trajectory(observables, theta0, theta_dot0)

    file_path = "data/pendulum_trajectories.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    generate_trajectories(num_trajectories=1024)
