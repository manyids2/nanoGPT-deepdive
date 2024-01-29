# TODO: pip install tabluate
import math
import tiktoken
from rich import inspect
import pandas as pd


ENCODINGS = ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base"]


def inspect_encodings():
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        inspect(enc)


def print_table():
    rows = []
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        row = {
            "name": enc.name,
            "n_vocab": enc.n_vocab,
            "max_token_value": enc.max_token_value,
            "eot_token": enc.eot_token,
            "special_tokens_set": enc.special_tokens_set,
            "pat_str": enc._pat_str,
            "mergeable_ranks": len(enc._mergeable_ranks),
            "mergeable_ranks_and_special_tokens": len(enc._mergeable_ranks)
            + len(enc.special_tokens_set),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_markdown())


def print_all_tokens():
    rows = []
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)

        # BUG: Wrong n_vocab reported by `cl100k_base`
        if name == "cl100k_base":
            n_vocab = 100261
        else:
            n_vocab = enc.n_vocab

        # Record to tsv in chunks
        CHUNK = 1000
        for i in range(math.ceil(n_vocab / float(CHUNK))):
            start = i * CHUNK
            end = min(n_vocab, (i + 1) * CHUNK)
            try:
                indices = list(range(start, end))
                tokens = enc.decode(indices)
            except:
                indices = [*list(range(start, 100256)), *list(range(100257, end))]
                # cl100k_base case : 100256 missing
                tokens = enc.decode(indices)
            row = {
                "name": enc.name,
                "start": start,
                "end": end,
                "tokens": tokens,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # BUG: _csv.Error: need to escape, but no escapechar set
    df.to_csv("/tmp/tokens.tsv")


def print_all_to_screen():
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        # BUG: Wrong n_vocab reported by `cl100k_base`
        if name == "cl100k_base":
            n_vocab = 100261
        else:
            n_vocab = enc.n_vocab

        # Record to tsv in chunks
        CHUNK = 1000
        for i in range(math.ceil(n_vocab / float(CHUNK))):
            start = i * CHUNK
            end = min(n_vocab, (i + 1) * CHUNK)
            if (name != "cl100k_base") | (end != n_vocab):
                indices = list(range(start, end))
                tokens = enc.decode(indices)
            else:
                indices = [*list(range(start, 100256)), *list(range(100257, end))]
                for i in indices[-10:]:
                    tokens = enc.decode(indices)
                # cl100k_base case : 100256 missing
                tokens = enc.decode(indices)

            # Remove all stuff that interferes with csv table parsing
            tokens = tokens.replace(",", "")
            tokens = tokens.replace("\n", "")
            print(enc.name, start, end, f'"{tokens}"', sep=",")


def check_encodings():
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        errors = []
        for i in range(enc.n_vocab):
            try:
                enc.decode([i])
            except:
                errors.append(i)
        print(f"{enc.name}: {errors} (len: {len(errors)})")


def save_mergeable_ranks():
    rows = []
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        m = enc._mergeable_ranks
        for k, v in m.items():
            row = {
                "name": name,
                "idx": v,
                "token": k,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("mergeable_ranks.csv", index=False)


if __name__ == "__main__":
    print_table()
    # print_all_to_screen()
    # save_mergeable_ranks()
