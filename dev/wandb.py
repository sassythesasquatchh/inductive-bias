from wandb.apis.public import Api
import ipdb
import wandb
import dotenv
import os
import json
import pandas as pd
from functools import reduce

name_map = {
    "mlp-informed-decoder": "MLP (Informed decoder)",
    "unstructured":"MLP (Latent Space)",
    "informed": "Informed Dynamics",
    "fully-informed": "Control",
    "baseline": "MLP (Baseline)",
    "hybrid":"Hybrid Dynamics",
    # "partially-informed": "Partially-informed"
}

chart_name_map={
    "orbit_length_table": "in_distribution_length",
    "open_length_table":"out_of_distribution_length",
    "orbit_energy_table": "in_distribution_energy",
    "open_energy_table": "out_of_distribution_energy",
    "test_orbit_table": "in_distribution_error",
    "test_open_table": "test_out_of_distribution_error"
}

# def df_to_latex(df:pd.DataFrame):
#     # csv_file = "/home/patrick/Downloads/wandb_export_2025-08-27T17_06_44.629+12_00.csv"
#     # df = pd.read_csv(csv_file)
#     df= df[df["Frame number"] < 540]

#     for col_name in chart_name_map.values():

#     stats = df.groupby("name")["Length"].agg(["mean", "std"]).reset_index()
#     stats["name"] = stats["name"].str.split("_", n=1).str[0]

#     stats = stats[stats["name"].isin(name_map.keys())]

#     # Apply the mapping
#     stats["name"] = stats["name"].map(name_map)

#     stats = stats.rename(columns={
#     "name": "Model",
#     "mean": "Mean Length",
#     "std": "Std. Dev."
# })

#     table = stats.to_latex(index=False, float_format="%.3f")

#     print(table)
#     return table

def df_to_latex(df: pd.DataFrame):
    # Filter rows
    df = df[df["Frame number"] < 540]

    # Keep only columns to summarize (excluding 'Frame number' and 'name')
    value_columns = [col for col in df.columns if col not in ["Frame number", "name"]]

    # Initialize an empty list to collect the aggregated DataFrames
    agg_list = []

    for col in value_columns:
        stats = df.groupby("name")[col].agg(["mean", "std"]).reset_index()
        stats["name"] = stats["name"].str.split("_", n=1).str[0]
        stats = stats[stats["name"].isin(name_map.keys())]
        stats["name"] = stats["name"].map(name_map)
        stats = stats.rename(columns={
            "name": "Model",
            "mean": f"{col} Mean",
            "std": f"{col} Std. Dev."
        })
        agg_list.append(stats)

    # Merge all aggregated columns on "Model"
    combined_stats = reduce(lambda left, right: pd.merge(left, right, on="Model"), agg_list)
    latex_table = combined_stats.to_latex(index=False, float_format="%.2f")


    return latex_table



dotenv.load_dotenv()

api_key = os.getenv("WANDB_KEY")

wandb.login(key=api_key)
api = wandb.Api()

runs = api.runs(
    path = "padowd-eth-z-rich/inductive-biases",
)


normal_runs = [r for r in runs if r.config.get("train_path","") == "data/normal_training_1000_closed_traj.pkl" and "19_08_25" in r.tags]
sparse_runs = [r for r in runs if r.config.get("train_path","") == "data/sparse_training_100_closed_traj.pkl" and "19_08_25" in r.tags]
few_data_runs = [r for r in runs if r.config.get("train_path","") == "data/normal_training_20_closed_traj.pkl" and "19_08_25" in r.tags]
small_forecast = [r for r in runs if r.config.get("context", 0) == 5 and "20_08_25" in r.tags]

run_sets = [normal_runs, sparse_runs, few_data_runs, small_forecast]

assert all([len(rs)!=0 for rs in run_sets])

table_names = ["Normal", "Sparse", "Limited Data", "Reduced Forecast"]

for runs, table_name in zip(run_sets, table_names):
    dfs=[]
    for artifact_name in ["open_length_table", "orbit_length_table", "open_energy_table", "orbit_energy_table", "test_orbit_table", "test_open_table"]:
        df =None
        for run in runs:
            path = f"padowd-eth-z-rich/inductive-biases/run-{run.id}-{artifact_name}:latest"
            local_dir = f"./artifacts/{run.id}/{artifact_name}"
            local_json = os.path.join(local_dir, f"{artifact_name}.table.json")

            if not os.path.exists(local_json):
                artifact = api.artifact(path, type="run_table")
                dir = artifact.download(root=local_dir)
            else:
                dir = local_dir

            with open(local_json, "r") as f:
                data = json.load(f)

            new_df = pd.DataFrame(data=data["data"], columns=data["columns"])
            new_df["name"] = run.config["run_name"]
            col_name = list(set(new_df.columns.to_list())-set(("name", "Frame number")))[0]

            new_df.rename(columns={col_name: chart_name_map[artifact_name]}, inplace=True)

            if df is not None:
                df = pd.concat([df, new_df])
            else:
                df = new_df

        dfs.append(df)


    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['Frame number', 'name']), dfs)
    table =df_to_latex(combined_df)
    with open(f"{table_name.lower().replace(' ','_')}_summary_table.tex", "w") as f:
        f.write(table)
