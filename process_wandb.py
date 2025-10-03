import pandas as pd
import ipdb
import wandb
import requests
import dotenv
import os

dotenv.load_dotenv()

name_map = {
    "mlp-informed-decoder": "MLP (Informed decoder)",
    "unstructured":"MLP (Latent Space)",
    "informed": "Informed Dynamics",
    # "fully-informed": "Control",
    "baseline": "MLP (Baseline)",
    "hybrid":"Hybrid Dynamics",
    # "partially-informed": "Partially-informed"
}

def df_to_latex(df:pd.DataFrame):
    # csv_file = "/home/patrick/Downloads/wandb_export_2025-08-27T17_06_44.629+12_00.csv"
    # df = pd.read_csv(csv_file)
    df= df[df["Frame number"] < 500]
    stats = df.groupby("name")["Length"].agg(["mean", "std"]).reset_index()
    stats["name"] = stats["name"].str.split("_", n=1).str[0]

    stats = stats[stats["name"].isin(name_map.keys())]

    # Apply the mapping
    stats["name"] = stats["name"].map(name_map)

    stats = stats.rename(columns={
    "name": "Model",
    "mean": "Mean Length",
    "std": "Std. Dev."
})

    table = stats.to_latex(index=False, float_format="%.3f")

    print(table)
    return table


if __name__ == "__main__":
    import ipdb

    try:
        pass
    except Exception as e:
        print(e)
        ipdb.post_mortem()