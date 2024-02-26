---
title: "tiktoken"
date: 2024-01-27T13:21:21+01:00
draft: true
---

```python

@dataclass
class Sample:
    idx: int       # sample idx
    data: np.array # single sample

@dataclass
class Batch:
    idx: int               # batch idx
    sample_idxs: list[int] # keep reference
    data: np.array         # collated batches

@dataclass
class Dataset(torch.Dataset):
    # Essential
    datadir: Path          # expected to contain files
    files: dict[str, Path] # init rel to datadir, if any missing, bail
    encoding: str          # tiktoken encoder used for encode, decode

    # Computed
    n_vocab: int      # from some metadata file
    n_train: int      # samples in train
    n_val: int        # samples in val
    __len__() -> int  # n_train + n_val
    __repr__() -> str # summarize above

    # Dataloader
    __get_item__(idx: int) -> Sample
    _collate(samples: list[Sample]) -> Batch
    # TODO: How to annotate torch Dataloader for dataset?
    get_dataloader(device: str,
                   batch_size: int,
                   block_size: int,
                   num_workers: int) -> torch.Dataloader[Dataset]

# Utilities to load:
load_dataset(name: str,
             run: str,
             datadir: Path,
             srcdir: Path) -> Dataset

# Scripts to create datasets should follow this signature
create_dataset(name: str,
               run: str,
               datadir: Path,
               srcdir: Path) -> Dataset
```
