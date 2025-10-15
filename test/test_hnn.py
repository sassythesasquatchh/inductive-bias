from hnn.train import train
from util.dataset import JaxTrajectoryDataset
from util.jax import to_torch

from .util import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_VISUALISATION_PATH,
)


def test_hnn():
    model = train(
        training_data_path=DEFAULT_TRAIN_PATH,
        validation_data_path=DEFAULT_VAL_PATH,
        num_epochs=1,
        batch_size=2,
        sequence_length=10,
    )

    visualisation_dataset = JaxTrajectoryDataset(
        data_path=DEFAULT_VISUALISATION_PATH, type="observed"
    )

    rollout = model.rollout(visualisation_dataset.data)
    rollout = {k: to_torch(v) for k, v in rollout.items()}


if __name__ == "__main__":
    import ipdb

    try:
        test_hnn()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
