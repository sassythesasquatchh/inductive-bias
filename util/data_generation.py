import argparse
import pickle

import numpy as np

from util.config import Config

config = Config()


def pendulum_ode(t, y):
    theta, theta_dot = y
    dydt = [theta_dot, -(config.GRAVITY / config.L) * np.sin(theta)]
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


def get_ranges(option: str, num_trajectories: int, only_small: bool = False):
    if not only_small:
        if option == "normal_training":
            range_1 = np.linspace(0.5, 3, int(num_trajectories * 2.5 / 9.5))
            range_2 = np.linspace(4, 8, int(num_trajectories * 4 / 9.5))
            range_3 = np.linspace(9, 12, int(num_trajectories * 3 / 9.5))
            theta_dot0_vals = np.concatenate((range_1, range_2, range_3))
        elif option == "validation":
            range_1 = np.linspace(3, 3.4, num_trajectories // 4)
            range_2 = np.linspace(3.6, 4, num_trajectories // 4)
            range_3 = np.linspace(8, 8.4, num_trajectories // 4)
            range_4 = np.linspace(8.6, 9, num_trajectories // 4)
            theta_dot0_vals = np.concatenate((range_1, range_2, range_3, range_4))
        elif option == "visualisation":
            theta_dot0_vals = np.array([2, 3.5, 6, 8.5])
        elif option == "sparse_training":
            range_1 = np.linspace(0.5, 1, int(num_trajectories * 2.5 / 9.5))
            range_2 = np.linspace(6, 7, int(num_trajectories * 4 / 9.5))
            range_3 = np.linspace(11.5, 12, int(num_trajectories * 3 / 9.5))
            theta_dot0_vals = np.concatenate((range_1, range_2, range_3))
        elif option == "continuity_test":
            theta_dot0_vals = np.linspace(0.5, 12, num_trajectories)

    else:
        if option == "normal_training":
            theta_dot0_vals = np.linspace(0.5, 3, num_trajectories)
        elif option == "validation":
            theta_dot0_vals = np.linspace(3, 4, num_trajectories)
        elif option == "visualisation":
            theta_dot0_vals = np.array([2, 3.5])
        elif option == "sparse_training":
            theta_dot0_vals = np.linspace(0.5, 1, num_trajectories)
        elif option == "continuity_test":
            theta_dot0_vals = np.linspace(0.5, 4, num_trajectories)

    return theta_dot0_vals


def get_trajectory_names(args):
    if args.option == "visualisation":
        if args.only_small:
            return ["closed_in_dist", "closed_out_dist"]
        else:
            return [
                "closed_in_dist",
                "closed_out_dist",
                "open_in_dist",
                "open_out_dist",
            ]

    return None


def generate_trajectories(
    args, t_eval=np.linspace(0, config.TIMESPAN, config.NUM_SAMPLES)
):
    trajectories = []
    initial_conditions_list = []

    theta_dot0_vals = get_ranges(args.option, args.num_trajectories, args.only_small)

    for theta_dot0 in theta_dot0_vals:
        initial_conditions_list.append((0, theta_dot0))

    # Sample points along the pendulum
    sampling_positions = config.L * np.array(config.SAMPLING_POSITIONS)

    for i, (theta0, theta_dot0) in enumerate(initial_conditions_list):
        y, t = simulate_trajectory([theta0, theta_dot0], t_eval)

        theta = y[0]
        theta_dot = y[1]

        # Calculate observables
        x = sampling_positions.reshape(-1, 1) * np.sin(theta)
        y = -sampling_positions.reshape(-1, 1) * np.cos(theta)
        linear_velocity = sampling_positions.reshape(-1, 1) * theta_dot

        canonical = np.vstack((theta, theta_dot)).T  # Shape (N, 2)
        observables = np.vstack((x, y, linear_velocity)).T  # Shape (N, 12)

        if args.noise > 0:
            noise = np.random.normal(0, args.noise, size=observables.shape)
            noise[:4, :] = sampling_positions.reshape(-1, 1) * noise[:4, :]
            noise[4:8, :] = -sampling_positions.reshape(-1, 1) * noise[4:8, :]
            noise[8:12, :] = sampling_positions.reshape(-1, 1) * noise[8:12, :]
            observables += noise

        trajectories.append(
            {
                "phase": canonical,
                "observed": observables,
            }
        )

    filename = f"{args.option}"
    if ("training" in args.option) or ("validation" in args.option):
        filename += f"_{args.num_trajectories}"
        if args.noise > 0:
            filename += f"_{args.noise}"

    result = {
        "trajectories": trajectories,
        "simulation_config": config.model_dump(),
        "dataset_config": {
            "trajectory_type": args.option,
            "noise": args.noise,
            "seed": args.seed,
            "only_closed_orbits": args.only_small,
        },
        "names": get_trajectory_names(args),
    }
    # if args.only_small:
    #     filename += "_closed_traj"
    file_path = f"data/{filename}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pendulum trajectories.")
    parser.add_argument(
        "--option",
        type=str,
        choices=[
            "normal_training",
            "validation",
            "visualisation",
            "sparse_training",
            "continuity_test",
        ],
        required=True,
        help="Type of trajectory generation.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=1000,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Amount of noise to add to the observations.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--only-small", action="store_true", help="Generate only small trajectories."
    )

    args = parser.parse_args()

    if args.option == "visualisation":
        args.num_trajectories = 2 if args.only_small else 4
    if args.option == "continuity_test":
        args.num_trajectories = 100
    generate_trajectories(args)
