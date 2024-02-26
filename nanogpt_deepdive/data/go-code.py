from pathlib import Path
import pickle
import numpy as np
import tiktoken


def encode_tiktoken_files(
    name: str,
    files: list[Path],
    train_bin: Path,
    val_bin: Path,
    padding: int = 256,  # TODO: Use end tokens properly
):
    if train_bin.exists() and val_bin.exists():
        return
    data = ""

    pbar = tqdm(files, total=len(files), desc="Files")
    for f in pbar:
        if not f.is_file():
            continue
        data += f.read_text() + " " * padding  # TODO: use end token

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding(name)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)


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

    expt = Experiment(
        name="go-code",
        run="tiktoken-gpt2",
        logdir=dir_from_env("NANOGPT_DATADIR"),
        create=True,
    )

    # save the meta information as well, to help us encode/decode later
    enc = tiktoken.get_encoding("gpt2")
    meta = {
        "vocab_size": enc.max_token_value + 1,  # End token :/
        # "itos": {v: k for k, v in enc._mergeable_ranks.items()},  # Hopefully never used
        # "stoi": enc._mergeable_ranks,
    }
    with open(expt.rundir / f"meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    repos = [d for d in (expt.namedir / "source" / "go").iterdir() if d.is_dir()]
    repos = sorted(repos)

    skiplist = {"kubernetes"}
    repos = sorted(list(set(repos).difference(skiplist)))

    for idx, repo in enumerate(repos):
        print(f"\n> ({idx+1}/{len(repos)}) {repo.name}")
        partial = expt.rundir / f"partial-{repo.name}"
        if partial.exists():
            continue
        partial.touch()
        try:
            encode_tiktoken_files(
                name="gpt2",
                files=sorted(repo.glob("**/*.go")),
                train_bin=expt.rundir / f"train-{repo.name}.bin",
                val_bin=expt.rundir / f"val-{repo.name}.bin",
            )
        except KeyError:
            continue
        partial.unlink()

    collate_bins(expt.rundir, "train", max_tokens=10_000_000)
    collate_bins(expt.rundir, "val", max_tokens=1_000_000)
