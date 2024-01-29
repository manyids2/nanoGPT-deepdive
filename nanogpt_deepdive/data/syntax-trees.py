from pathlib import Path
import shutil
import pickle
import numpy as np

start_end = ["^", "$"]  # Do not appear in s-expressions
paranthesis = ["(", ")"]
whitespace = [" ", "\n", ":", "_", '"']
a_to_z = [chr(ord("a") + i) for i in range(26)]
missing_error = ["M", "I", "S", "N", "G", "E", "R", "O"]
chars = [*start_end, *paranthesis, *whitespace, *a_to_z, *missing_error]
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


def encode_char_repo(repo_dir: Path, train_bin: Path, val_bin: Path, meta_pkl: Path):
    if train_bin.exists() and val_bin.exists() and meta_pkl.exists():
        return
    # - all lowercase letters (26)
    # - paranthesis (2)
    # - space, newline, colon (3)
    # - start, end (2)
    # concat syntax trees from project dir
    data = ""

    files = sorted(repo_dir.glob("**/*.go"))
    pbar = tqdm(files, total=len(files), desc="Files")
    for f in pbar:
        if not f.is_file():
            continue
        data += "^" + f.read_text() + "$"

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(meta_pkl, "wb") as f:
        pickle.dump(meta, f)

    # length of dataset in characters: ...
    # all the unique characters:
    #  ...
    # vocab size: ...
    # train has ... tokens
    # val has ... tokens


def collate_bins(base_dir: Path, split: str, max_tokens: int = 10_000_000):
    print(f"Collating: {split}")
    bins = sorted(base_dir.glob(f"{split}-*.bin"))
    pbar = tqdm(bins, total=len(bins))
    n = 0
    data = []
    for bin in pbar:
        d = np.fromfile(bin, dtype=np.uint16)
        if n + len(d) > max_tokens:
            # Maybe there is a small dataset somewhere
            # Greedy fill
            continue
        data.append(d)
        n += len(d)
    data = np.concatenate(data)
    ids = np.array(data, dtype=np.uint16)
    ids.tofile(base_dir / f"{split}.bin")
    print(f"Saved: {split}.bin : {n} tokens ( {n/1e6:.2f} MB )")
    # full dataset : 5,532,719,741 ~ 5.5GB
    # max_tokens 10M: 9,999,970 ~ 10MB


if __name__ == "__main__":
    from tqdm import tqdm
    from nanogpt_deepdive.experiment import Experiment, dir_from_env

    DATADIR = "NANOGPT_DATADIR"
    expt = Experiment(
        name="syntax-trees-go-tokens",
        run="char",
        logdir=dir_from_env(DATADIR),
        create=True,
    )

    repos = [d for d in (expt.namedir / "source" / "go").iterdir() if d.is_dir()]
    # for repo in repos:
    #     print(f"\n> {repo.name}")
    #     try:
    #         encode_char_repo(
    #             repo_dir=repo,
    #             train_bin=expt.rundir / f"train-{repo.name}.bin",
    #             val_bin=expt.rundir / f"val-{repo.name}.bin",
    #             meta_pkl=expt.rundir / f"metadata-{repo.name}.pkl",
    #         )
    #     except KeyError:
    #         failed = expt.rundir / f"failed-{repo.name}"
    #         failed.touch()
    #         continue

    collate_bins(expt.rundir, "train", max_tokens=10_000_000)
    collate_bins(expt.rundir, "val", max_tokens=1_000_000)
    meta = expt.rundir / f"metadata-{repos[0].name}.pkl"
    shutil.copy(meta, expt.rundir / "meta.pkl")
