from dataclasses import dataclass


@dataclass
class RowDataset:
    """Summary row for dataset, to put in db."""

    name: str
    run: str
    datadir: str

    block_size: int
    batch_size: int
    device: str
    device_type: str
    meta: str

    # Stats
    n_train: int
    n_val: int
