from pathlib import Path
import pickle
import requests
import tiktoken
import numpy as np


def download_dataset(path: Path):
    if path.exists():
        return

    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(path, "w") as f:
        f.write(requests.get(data_url).text)
    print(f"Downloaded: {path}")


def encode_tiktoken(input_txt: Path, train_bin: Path, val_bin: Path):
    data = input_txt.read_text()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # --- Shakespeare tiktoken ---
    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    # train.bin has 301,966 tokens
    # val.bin has 36,059 tokens


def encode_char(input_txt: Path, train_bin: Path, val_bin: Path, meta_pkl: Path):
    data = input_txt.read_text()
    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

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

    # length of dataset in characters:  1115394
    # all the unique characters:
    #  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    # vocab size: 65
    # train has 1003854 tokens
    # val has 111540 tokens


if __name__ == "__main__":
    from nanogpt_deepdive.experiment import Experiment, dir_from_env

    DATADIR = "NANOGPT_DATADIR"

    expt = Experiment(
        name="shakespeare-tokens",
        run="source",
        logdir=dir_from_env(DATADIR),
        create=True,
    )
    path = expt.namedir / "input.txt"
    download_dataset(path)

    expt = Experiment(
        name="shakespeare-tokens",
        run="char",
        logdir=dir_from_env(DATADIR),
        create=True,
    )
    train_bin = expt.rundir / "train.bin"
    val_bin = expt.rundir / "val.bin"
    meta_pkl = expt.rundir / "meta.pkl"
    encode_char(path, train_bin, val_bin, meta_pkl)

    expt = Experiment(
        name="shakespeare-tokens",
        run="tiktoken",
        logdir=dir_from_env(DATADIR),
        create=True,
    )
    encode_tiktoken(path, train_bin, val_bin)
