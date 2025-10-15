import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional
import argparse
from pre_util import LitModel
from common.classes import BaseModel, RolloutOutput


def test_rollout(
    model: LitModel,
    dataset: DataLoader,
    context_length: int = 8,
):
    model.eval()
    device = model.device
    inference_model: BaseModel = model.model
    with torch.no_grad():
        for i, trajectory in enumerate(dataset):
            rollout: RolloutOutput = inference_model.rollout(
                trajectory.to(device), context_length=context_length
            )
