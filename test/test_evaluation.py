import pytest
import torch

from common.classes import RolloutOutput
from util.config import Config
from util.dataset import TorchTrajectoryDataset
from util.pre_util import calculate_energy, get_length
from util.rollout import evaluate_rollout
from util.test_continuity import test_continuity
from util.visualisation import animate_trajectories

from .util import DEFAULT_CONTINUITY_PATH, DEFAULT_MODEL, DEFAULT_VISUALISATION_PATH


@pytest.fixture(scope="module")
def vis_rollout():
    return _vis_rollout()


def _vis_rollout():
    model = DEFAULT_MODEL
    dataset = TorchTrajectoryDataset(
        data_path=DEFAULT_VISUALISATION_PATH, type="observed"
    )
    # B N D
    rollout_data = dataset.data[:2, :40, :]
    rollout = model.rollout(rollout_data)
    return rollout, dataset


def test_rollout_eval(vis_rollout):
    evaluate_rollout(vis_rollout[0], vis_rollout[1])


def test_continuity_metric():
    model = DEFAULT_MODEL
    dataset = TorchTrajectoryDataset(data_path=DEFAULT_CONTINUITY_PATH, type="observed")
    # B N D
    rollout = model.rollout(dataset.data[:, :40, :])

    test_continuity(rollout, dataset.initial_velocities)


def test_animate(vis_rollout: RolloutOutput):
    vis_rollout = vis_rollout[0]
    traj_names = [str(i) for i in range(vis_rollout["latent_gt"].shape[0])]
    animate_trajectories(vis_rollout, Config(), traj_names, "test")


def test_metrics(vis_rollout):
    rollout = vis_rollout[0]["obs_gt"]
    config = vis_rollout[1].config
    length = get_length(rollout, config)
    energy = calculate_energy(rollout, config)

    assert torch.isclose(length, torch.Tensor([config.L]), atol=1e-3).all()
    assert torch.isclose(
        energy, torch.mean(energy, dim=-1).unsqueeze(1), atol=7e-1
    ).all()


if __name__ == "__main__":
    import ipdb

    try:
        rollout = _vis_rollout()
        test_rollout_eval(rollout)
        test_continuity_metric()
        test_animate(rollout)
        test_metrics(rollout)
    except Exception as e:
        print(e)
        ipdb.post_mortem()
