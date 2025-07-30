import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import wandb
import numpy as np
import ipdb
from datasets import TrajectoryDataset
from pre_util import LitModel, get_model, parse_args, calculate_energy, get_length
import os
import matplotlib.pyplot as plt

from wandb_util import to_wandb
from pre_util import process_trajectory, get_variance


def test_rollout(
    model: pl.LightningModule,
    dataset: DataLoader,
    args: argparse.Namespace = None,
    step: int = 0,
    as_figure: bool = True,
):
    """
    Advect trajectories from the initial point.
    Dataset should be a trajectory dataloader
    """
    model.eval()
    # model.freeze()
    device = model.device
    with torch.no_grad():
        for i, trajectory in enumerate(dataset):
            # x = trajectory.squeeze(0).to(device).transpose(0, 1)
            # reconstructed_trajectory = torch.zeros_like(x)
            # if not args.training == "rwm":
            #     ic = x[0].unsqueeze(0)
            #     reconstructed_trajectory[0] = ic
            #     upper_bound = x.size(0) - 1
            # else:
            #     x_t = x[: args.context, :].reshape(1, -1)
            #     x_t1 = torch.zeros_like(x_t)
            #     reconstructed_trajectory[: args.context, :] = x[: args.context, :]
            #     upper_bound = x.size(0) - args.context
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

            (
                x,
                reconstructed_trajectory,
                _,
                _,
            ) = process_trajectory(model, trajectory, args, device)
            # energy = calculate_energy(reconstructed_trajectory).cpu().numpy()
            energy = calculate_energy(reconstructed_trajectory).cpu().numpy()
            length = get_length(reconstructed_trajectory).cpu().numpy()
            variance = get_variance(x).cpu().numpy()
            # plt.plot(np.arange(0, x.size(0)), energy)
            # plt.savefig("energy.png")

            difference = torch.norm(reconstructed_trajectory - x, dim=1)

            difference /= torch.mean(torch.norm(x, dim=1))
            # difference /= torch.norm(x, dim=1)
            difference *= 100
            difference = difference.cpu().numpy()
            times = np.arange(0, x.size(0))

            energy_table = wandb.Table(
                data=to_wandb(times.tolist(), energy.tolist()),
                columns=["Frame number", "Energy"],
            )

            length_table = wandb.Table(
                data=to_wandb(times.tolist(), length.tolist()),
                columns=["Frame number", "Length"],
            )

            # variance_table = wandb.Table(
            #     data=to_wandb(times.tolist(), variance.tolist()),
            #     columns=["Frame number", "Variance"],
            # )

            try:
                wandb.log(
                    {
                        "orbit_energy" if i == 0 else "open_energy": wandb.plot.line(
                            energy_table,
                            "Frame number",
                            "Energy",
                            title=f'Energy ({"orbit" if i == 0 else "open"})',
                        ),
                    },
                )

                wandb.log(
                    {
                        "orbit_length" if i == 0 else "open_length": wandb.plot.line(
                            length_table,
                            "Frame number",
                            "Length",
                            title=f'Length ({"orbit" if i == 0 else "open"})',
                        ),
                    },
                )

                plt.plot(times, energy, label=f"Trajectory {i}")
                plt.xlabel("Frame number")
                plt.ylabel("Energy")
                plt.title("Energy")
                if i == 0:  # Orbit
                    plt.axhline(y=5.78, color="b", linestyle="--", label="Val")
                    plt.axhline(y=6.48, color="b", linestyle="--", label="Val")
                    plt.axhline(y=4.5, color="r", linestyle="--", label="Train")
                    plt.axhline(y=8, color="r", linestyle="--", label="Train")
                    plt.axhline(y=6.125, color="g", linestyle="--", label="True")

                if i == 1:  # Open
                    plt.axhline(y=35.28, color="b", linestyle="--", label="Val")
                    plt.axhline(y=36.98, color="b", linestyle="--", label="Val")
                    plt.axhline(y=32, color="r", linestyle="--", label="Train")
                    plt.axhline(y=40.5, color="r", linestyle="--", label="Train")
                    plt.axhline(y=36.125, color="g", linestyle="--", label="True")

                plt.legend()

                wandb.log(
                    {
                        (
                            "test_energy_fig_orbit"
                            if i == 0
                            else "test_energy_fig_open"
                        ): wandb.Image(
                            plt,
                            caption=f'Energy ({"orbit" if i == 0 else "open"})',
                        ),
                    },
                )
                plt.clf()
                plt.close()

                # wandb.log(
                #     {
                #         (
                #             "orbit_variance" if i == 0 else "open_variance"
                #         ): wandb.plot.line(
                #             variance_table,
                #             "Frame number",
                #             "Variance",
                #             title=f"Variance ({"open" if i == 0 else "orbit"})",
                #         ),
                #     },
                # )

                if as_figure:
                    plt.plot(
                        times,
                        difference,
                        label=f"Trajectory {i}",
                    )
                    plt.xlabel("Frame number")
                    plt.ylabel("Relative Error")
                    plt.title("Percentage Error")

                    wandb.log(
                        {
                            "test_orbit" if i == 0 else "test_open": wandb.Image(
                                plt,
                                caption=f'Percentage Error ({"orbit" if i == 0 else "open"})',
                            ),
                        },
                    )
                    plt.clf()
                    plt.close()

                else:
                    error_table = wandb.Table(
                        data=to_wandb(times.tolist(), difference.tolist()),
                        columns=["Frame number", "Relative Error"],
                    )
                    wandb.log(
                        {
                            "test_orbit" if i == 0 else "test_open": wandb.plot.line(
                                error_table,
                                "Frame number",
                                "Relative Error",
                                title=f'Percentage Error ({"orbit" if i == 0 else "open"})',
                            ),
                        },
                    )
            except Exception as e:
                print(f"Failed to log data to wandb: {e}")


if __name__ == "__main__":
    args = parse_args()
    if args.model == "unstructured":
        model_path = "checkpoints/unstructured_3_10_0.0/99-0.00.ckpt"
    elif args.model == "informed":
        model_path = "checkpoints/informed_3_10_0.0/56-0.00.ckpt"
    elif args.model == "hybrid":
        model_path = "checkpoints/hybrid_3_10_0.0/99-0.00.ckpt"
    elif args.model == "baseline":
        model_path = "checkpoints/baseline_3_10_0.0/140-0.00.ckpt"
    else:
        raise ValueError("Invalid model name")
    model, criterion = get_model(args)
    model = LitModel.load_from_checkpoint(model_path, model=model, criterion=criterion)
    test_data = TrajectoryDataset(data_path=args.visualisation_data_path, split="test")
    test_loader = DataLoader(test_data, batch_size=1, num_workers=8)

    LOG = True

    if LOG:
        from wandb_util import get_run_id

        run_id = get_run_id(
            os.getenv("WANDB_PROJECT"),
            args.run_name,
        )
        run_id = "pnnq6tsk"

        run = wandb.init(project=os.getenv("WANDB_PROJECT"), id=run_id, resume="must")

    try:
        test_rollout(model, test_loader, args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
