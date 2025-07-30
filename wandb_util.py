import wandb
from dotenv import load_dotenv
import ipdb

load_dotenv()


def to_wandb(_x: list, _y: list):
    return [[x, y] for x, y in zip(_x, _y)]


def get_run_id(project_name, run_name):
    """Gets the run ID of an existing W&B run."""

    api = wandb.Api()
    try:
        runs = api.runs(project_name, filters={"displayName": run_name})

        if not runs:
            print(f"Run '{run_name}' not found in project '{project_name}'.")
            return None

        if len(runs) > 1:
            print(
                f"Warning: Multiple runs with name '{run_name}' found. Returning the id of the most recent one. "
                f"All: {[run.id for run in runs]}"
            )

        run_id = runs[0].id
        return run_id

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_model_weights_path(run_name, project_name="inductive-biases"):

    api = wandb.Api()
    run_id = get_run_id(project_name, run_name)
    run = api.run(f"{project_name}/{run_id}")
    model_name = None
    for artifact in run.logged_artifacts():
        if "model" in artifact.name:
            model_name = artifact.name.split(":")[0]
            break

    if model_name is None:
        print(f"No model artifact found for run {run_name}.")
        return
    artifact = api.artifact(f"{project_name}/{model_name}:best")
    model_dir = artifact.download()
    return model_dir / "model.ckpt"
