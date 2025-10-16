import torch

from context_models.decoders import InformedDecoder
from context_models.encoders import InformedEncoder
from util.dataset import TorchTrajectoryDataset

from .util import DEFAULT_VISUALISATION_PATH


def test_informed():
    test_dataset = TorchTrajectoryDataset(
        data_path=DEFAULT_VISUALISATION_PATH, type="observed"
    )
    encoder = InformedEncoder(config=test_dataset.config, context=1)
    decoder = InformedDecoder(config=test_dataset.config)

    x = test_dataset.data

    z = encoder(x)
    x_recon = decoder(z)

    assert torch.isclose(x, x_recon, atol=1e-3).all()


if __name__ == "__main__":
    import ipdb

    try:
        test_informed()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
