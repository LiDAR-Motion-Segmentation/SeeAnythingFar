"""Shared construction of datasets / loaders / system from a composed Hydra config."""
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from seeanythingfar.datasets.collate import collate_fn


def make_loader(cfg, split: str, shuffle: bool) -> DataLoader:
    dataset = instantiate(cfg.dataset[split])
    return DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle,
                      num_workers=cfg.data.num_workers, collate_fn=collate_fn,
                      pin_memory=True, drop_last=shuffle, persistent_workers=cfg.data.num_workers > 0)


def make_system(cfg):
    return instantiate(cfg.system, _recursive_=True)
