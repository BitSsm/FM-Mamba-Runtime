"""Local model construction and synthetic data for installation checks."""
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


def build_classifier(config_path):
    from models.mamba2d import Mamba2DBackbone
    from models.mamba2d_classifier import Mamba2DClassifier
    from models.utils import MlpHead

    config = OmegaConf.load(config_path)
    args = OmegaConf.to_container(config.model.init_args, resolve=True)
    backbone = Mamba2DBackbone(**args.pop('backbone')['init_args'])
    head = MlpHead(**args.pop('head')['init_args'])
    return Mamba2DClassifier(backbone=backbone, head=head, **args)


def load_weights(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    return len(state)


class SmokeDataModule(L.LightningDataModule):
    """Random images and labels: checks execution, never measures accuracy."""

    def __init__(self, batch_size: int = 2, num_workers: int = 0, cutmix: bool = False):
        super().__init__()
        if cutmix:
            raise ValueError('SmokeDataModule uses hard synthetic labels; set cutmix=false.')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutmix = cutmix

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)
        self.train_dataset = TensorDataset(
            torch.randn(4, 3, 224, 224, generator=generator),
            torch.randint(0, 1000, (4,), generator=generator),
        )
        self.val_dataset = TensorDataset(
            torch.randn(2, 3, 224, 224, generator=generator),
            torch.randint(0, 1000, (2,), generator=generator),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
