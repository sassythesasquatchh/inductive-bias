from fld.train import train
from test.util import DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, DEFAULT_VISUALISATION_PATH
from util.dataset import TorchTrajectoryDataset


def test_fld():
    model = train(
        train_path=DEFAULT_TRAIN_PATH,
        val_path=DEFAULT_VAL_PATH,
        hidden_dim=8,
        embedding_dim=2,
        batch_size=2,
        epochs=1,
        debug=True,
    )

    visualisation_dataset = TorchTrajectoryDataset(
        data_path=DEFAULT_VISUALISATION_PATH, type="observed"
    )

    model.model.rollout(visualisation_dataset.data)


if __name__ == "__main__":
    import ipdb

    try:
        test_fld()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
