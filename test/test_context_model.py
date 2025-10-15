from context_models.config import decoder_dict, dynamics_dict, encoder_dict
from context_models.train import train
from test.util import DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, DEFAULT_VISUALISATION_PATH
from util.dataset import TorchTrajectoryDataset


def test_default_model():
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


def all_combinations():
    for encoder_class in encoder_dict.values():
        for dynamics_class in dynamics_dict.values():
            for decoder_class in decoder_dict.values():
                model = train(
                    train_path=DEFAULT_TRAIN_PATH,
                    val_path=DEFAULT_VAL_PATH,
                    encoder_class=encoder_class,
                    dynamics_class=dynamics_class,
                    decoder_class=decoder_class,
                    hidden_dim=8,
                    embedding_dim=3,
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
    import torch

    torch.autograd.set_detect_anomaly(True)

    try:
        all_combinations()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
