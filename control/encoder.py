"""
Configures the encoder that maps the observable space to the latent
space used as input to the policy
"""

from models import Informed, Hybrid, UnstructuredDirect
from pre_util import parse_args, get_model, LitModel


def get_encoder(model: str = None, model_path: str = None):
    args = parse_args()

    if model_path is None:
        if model == "unstructured":
            model_path = "checkpoints/unstructured_3_10_0.0/99-0.00.ckpt"
        elif model == "informed":
            model_path = "checkpoints/informed_3_10_0.0/56-0.00.ckpt"
        elif model == "hybrid":
            model_path = "checkpoints/hybrid_3_10_0.0/99-0.00.ckpt"
        else:
            raise ValueError("Invalid model name")

    model, criterion = get_model(args, model_name=model)
    model = LitModel.load_from_checkpoint(model_path, model=model, criterion=criterion)
    return model.model.encoder


if __name__ == "__main__":
    encoder = get_encoder(model="unstructured")
    print(encoder)
