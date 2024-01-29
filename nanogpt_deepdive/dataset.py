from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pickle
import torch


@dataclass
class Dataset:
    name: str
    run: str
    datadir: Path

    block_size: int
    batch_size: int
    device: str
    device_type: str

    train_bin: Path = Path()
    val_bin: Path = Path()
    meta_pkl: Path = Path()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    meta: Optional[dict] = None
    meta_vocab_size: Optional[int] = None

    def __post_init__(self):
        self.train_bin = self.datadir / self.name / self.run / "train.bin"
        self.val_bin = self.datadir / self.name / self.run / "val.bin"

        # attempt to derive vocab_size from the dataset
        self.meta_pkl = self.datadir / self.name / self.run / "meta.pkl"
        if self.meta_pkl.exists():
            with open(self.meta_pkl, "rb") as f:
                self.meta = pickle.load(f)
            self.meta_vocab_size = self.meta["vocab_size"]
            print(f"found vocab_size = {self.meta_vocab_size} (inside {self.meta_pkl})")

        # poor man's data loader
        self.train_data = np.memmap(self.train_bin, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(self.val_bin, dtype=np.uint16, mode="r")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        block_size, device = self.block_size, self.device

        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y


if __name__ == "__main__":
    import sys
    from nanogpt_deepdive.config import Config
    from nanogpt_deepdive.experiment import Experiment, dir_from_env

    assert len(sys.argv) == 3
    expt = Experiment(sys.argv[1], sys.argv[2], create=True)
    cfg = Config(**expt.get_cfg_from_srcdir())
    data = Dataset(
        name=cfg.data_name,
        run=cfg.data_run,
        datadir=dir_from_env("NANOGPT_DATADIR"),
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        device=cfg.device,
        device_type=cfg.device_type,
    )
    print(data)
