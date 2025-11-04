import json
import os
import re
from collections import defaultdict
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import wandb

BASE_DIR = Path("results")
os.makedirs(BASE_DIR, exist_ok=True)

TAGS = [
    "loss_experiment_2",
]

# energy values include both closed and open trajectories
BACKUP_REF_VALS = {
    ("In-Distribution", "Length", r"$\mu$"): 1.0,
    ("In-Distribution", "Energy", r"$\mu$"): 10.045,
    ("Out-of-Distribution", "Length", r"$\mu$"): 1.0,
    ("Out-of-Distribution", "Energy", r"$\mu$"): 21.225,
}

run_name_map = {
    "identity_unstructured_identity": "Baseline",
    "supervise_rollout": "SR",
    "supervise_end_to_end": "SE",
    "supervise_both": "SB",
    "supervise_both_penalise_mismatch": "SBPM",
    "supervise_both_all_penalties": "SBAP",
}

table_names = [
    "error_latent_rollout_closed_in_dist_table",
    "error_latent_rollout_closed_out_dist_table",
    "error_latent_rollout_open_in_dist_table",
    "error_latent_rollout_open_out_dist_table",
    "energy_latent_rollout_closed_in_dist_table",
    "energy_latent_rollout_closed_out_dist_table",
    "energy_latent_rollout_open_in_dist_table",
    "energy_latent_rollout_open_out_dist_table",
    "length_latent_rollout_closed_in_dist_table",
    "length_latent_rollout_closed_out_dist_table",
    "length_latent_rollout_open_in_dist_table",
    "length_latent_rollout_open_out_dist_table",
]

plotly_artifacts = ["structure_overlay"]


def convert_name(name: str) -> str:
    """Convert a model name to its more descriptive form using the name_map.

    Args:
        name (str): The original model name.

    Returns:
        str: The converted model name.
    """
    if name in run_name_map:
        return run_name_map[name]
    modules = name.split("_")
    if len(modules) > 3:
        modules = [modules[0], "alt-hybrid", modules[-1]]
    module_abbreviations = [module[0].upper() for module in modules]
    return "".join(module_abbreviations)


def tags_from_config_file(config_file: Path) -> list[str]:
    """Reads a config file and returns a list of tags.

    Args:
        config_file (Path): Path to the config file.

    Returns:
        list[str]: A list of tags extracted from the config file.
    """
    tags = []
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tags.append(line.split(" ")[-2])
    return tags


def bold_best(df: pd.DataFrame, mode="min", ref_label="III", precision=2):
    """
    Return a copy of a MultiIndex-column DataFrame with the best value in each column bolded.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns (e.g. (dist, metric, stat)).
    mode : str or dict
        Either one of {"min", "max", "closest"} for all columns,
        or a dict specifying mode per column or per metric:
          {("In-Distribution", "Length", "$\\mu$"): "closest", ...}
          or {("In-Distribution", "Length"): "closest"} (applies to both μ and σ).
    ref_label : str
        Row label used as reference for "closest" mode (excluded from output).
    precision : int
        Number of decimal places to format.
    """
    df_fmt = df.copy().astype(object)
    df_num = df.apply(pd.to_numeric, errors="coerce")

    # Store reference row for "closest" mode
    ref_vals = df_num.loc[ref_label] if ref_label in df.index else None

    if ref_label in df_fmt.index:
        df_fmt = df_fmt.drop(ref_label)
        df_num = df_num.drop(ref_label)

    for col in df.columns:
        col_vals = df_num[col].dropna()
        if col_vals.empty:
            continue

        # Determine mode for this column
        if isinstance(mode, dict):
            this_mode = mode.get(col)
            # Fallback: match partial (dist, metric)
            if this_mode is None and len(col) > 1:
                this_mode = mode.get(col[:-1], "min")
            if this_mode is None:
                this_mode = "min"
        else:
            this_mode = mode

        # Identify the "best" index
        if this_mode == "min":
            best_idx = col_vals.idxmin()
        elif this_mode == "max":
            best_idx = col_vals.idxmax()
        elif this_mode == "closest":
            if ref_vals is None or ref_label not in df.index:
                ref_val = BACKUP_REF_VALS.get(col)
                if ref_val is None:
                    continue
            else:
                ref_val = ref_vals[col]
            best_idx = (df_num[col] - ref_val).abs().idxmin()
        else:
            raise ValueError(f"Invalid mode '{this_mode}' for column {col}")

        # Apply LaTeX bold formatting
        for idx, v in df_num[col].items():
            if np.isnan(v):
                df_fmt.loc[idx, col] = "-"
            elif idx == best_idx:
                df_fmt.loc[idx, col] = rf"\textbf{{{v:.{precision}f}}}"
            else:
                df_fmt.loc[idx, col] = f"{v:.{precision}f}"

    return df_fmt


def main(args):
    dotenv.load_dotenv()
    api_key = os.getenv("WANDB_KEY")
    wandb.login(key=api_key)
    api = wandb.Api()
    print("Fetching runs from wandb...")

    if args.config_file:
        tags = tags_from_config_file(args.config_file)
        if args.tag_suffix:
            tags = [tag + f"_{args.tag_suffix}" for tag in tags]
    else:
        tags = TAGS
    runs = api.runs(
        path="padowd-eth-z-rich/inductive-biases",
        filters={"tags": {"$in": tags}},
        per_page=1000,
    )
    print(f"Found {len(runs)} runs")

    run_sets = [[r for r in runs if (tag in r.tags)] for tag in tags]

    for runs, tag in zip(run_sets, tags):
        print(f"Processing tag: {tag} with {len(runs)} runs")
        experiment_directory = BASE_DIR / tag
        os.makedirs(experiment_directory, exist_ok=True)
        data = defaultdict(list)
        index_set = [convert_name(run.config["run_name"]) for run in runs]
        for table_name in table_names:
            for run in runs:
                path = f"padowd-eth-z-rich/inductive-biases/run-{run.id}-{table_name}:latest"
                local_dir = f"./artifacts/{run.id}/{table_name}"
                local_json = os.path.join(local_dir, f"{table_name}.table.json")
                if not os.path.exists(local_json):
                    try:
                        artifact = api.artifact(path, type="run_table")
                    except Exception:
                        break
                    print(f"Downloading artifact for run {run.id}, table {table_name}")
                    dir = artifact.download(root=local_dir)
                else:
                    dir = local_dir

                with open(local_json, "r") as f:
                    _data = json.load(f)

                df = pd.DataFrame(_data["data"], columns=_data["columns"])
                column_name = list(set(df.columns.to_list()) - set(("Frame number",)))[
                    0
                ]
                mean = df[column_name].mean()
                std = df[column_name].std()

                base_tuple = (
                    "In-Distribution"
                    if "in_dist" in table_name
                    else "Out-of-Distribution",
                    table_name.split("_")[0].capitalize(),  # eg. Length
                    table_name.split("_")[-4],  # open/closed
                )

                data[(*base_tuple, "mean")].append(mean)
                data[(*base_tuple, "var")].append(std**2)

        df = pd.DataFrame(data=data, index=index_set).T
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["dist", "metric", "energy_level", "stat"]
        )

        latent_alignments = []
        for run in runs:
            latent_alignment = run.summary.get("latent_alignment")
            latent_alignments.append(latent_alignment)

        collapsed = df.groupby(level=("dist", "metric", "stat")).mean()

        collapsed.loc[pd.IndexSlice[:, :, "var"], :] = np.sqrt(
            collapsed.loc[pd.IndexSlice[:, :, "var"], :]
        )
        collapsed = collapsed.rename(
            index=lambda x: r"$\sigma$"
            if x == "var"
            else (r"$\mu$" if x == "mean" else x),
            level="stat",
        ).T

        collapsed[r"$d_{A}$"] = latent_alignments

        collapsed = bold_best(
            collapsed,
            mode={
                ("In-Distribution", "Length", r"$\mu$"): "closest",
                ("In-Distribution", "Energy", r"$\mu$"): "closest",
                ("In-Distribution", "Error", r"$\mu$"): "min",
                ("Out-of-Distribution", "Length", r"$\mu$"): "closest",
                ("Out-of-Distribution", "Energy", r"$\mu$"): "closest",
                ("Out-of-Distribution", "Error", r"$\mu$"): "min",
                ("In-Distribution", "Length", r"$\sigma$"): "min",
                ("In-Distribution", "Energy", r"$\sigma$"): "min",
                ("In-Distribution", "Error", r"$\sigma$"): "min",
                ("Out-of-Distribution", "Length", r"$\sigma$"): "min",
                ("Out-of-Distribution", "Energy", r"$\sigma$"): "min",
                ("Out-of-Distribution", "Error", r"$\sigma$"): "min",
                "latent_alignment": "min",
            },
            ref_label="III",
            precision=2,
        )

        latex_table = collapsed.to_latex(
            escape=False,  # allow math symbols like μ, σ
            multicolumn=True,
            multicolumn_format="c",
            column_format="l" + "r" * len(collapsed.columns),
            caption=("blank"),
            label=f"tab:{tag}_results",
            float_format="%.2f",
        )

        m = re.search(r"\\midrule(.*?)\\bottomrule", latex_table, flags=re.S)
        if not m:
            raise RuntimeError(
                "Could not locate table body between \\midrule and \\bottomrule."
            )
        body_rows = m.group(1).strip()

        custom_header = r"""
\begin{table}
\centering
\begin{tabular}{lrrrrrrrrrrrr|r}
\toprule
& \multicolumn{6}{c}{In-Distribution} & \multicolumn{6}{c}{Out-of-Distribution} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13}
& \multicolumn{2}{c}{Energy} & \multicolumn{2}{c}{Error} & \multicolumn{2}{c}{Length} 
& \multicolumn{2}{c}{Energy} & \multicolumn{2}{c}{Error} & \multicolumn{2}{c}{Length} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} 
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} 
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} 
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} 
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} 
& \multicolumn{1}{c}{$\mu$} & \multicolumn{1}{c}{$\sigma$} & d_A \\
\midrule
        """.lstrip("\n")

        custom_footer = r"""
\bottomrule
\end{tabular}
\caption{\textbf{Title}. }
\end{table}
        """.lstrip("\n")

        latex_table = custom_header + body_rows + "\n" + custom_footer

        with open(experiment_directory / "results_table.tex", "w") as f:
            f.write(latex_table)

        for plotly_artifact in plotly_artifacts:
            for run in runs:
                for file in run.files():
                    if plotly_artifact in file.name:
                        dest_path = (
                            experiment_directory
                            / f"{run.config['run_name']}_{plotly_artifact}.pdf"
                        )

                        if dest_path.exists():
                            print(
                                f"Plot {dest_path} already exists, skipping download."
                            )
                            continue
                        print(
                            f"Downloading plotly artifact {plotly_artifact} for run {run.id}"
                        )
                        plot_path = file.download(replace=True)

                        with open(plot_path.name, "r") as f:
                            plot_data = json.load(f)

                        fig = go.Figure(plot_data)
                        fig.update_layout(showlegend=False)

                        fig.write_image(dest_path)


if __name__ == "__main__":
    import argparse

    import ipdb

    parser = argparse.ArgumentParser(
        description="Process and visualize results from wandb runs."
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=Path,
        help="Path to the configuration file containing experiment tags.",
    )
    parser.add_argument(
        "--tag_suffix",
        "-t",
        type=str,
        default="",
        help="Suffix to append to each tag from the config file.",
    )
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
