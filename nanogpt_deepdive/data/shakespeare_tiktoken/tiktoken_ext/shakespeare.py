from pathlib import Path
from tiktoken.load import load_tiktoken_bpe, dump_tiktoken_bpe


def dir_from_env(var: str = "NANOGPT_DATADIR"):
    import os

    _dir = os.environ.get(var)
    if _dir:
        return Path(_dir)
    else:
        raise KeyError(f"Env var not found: {var}")


INPUT_PATHS = {
    "shakespeare-tokens": dir_from_env() / "shakespeare-tokens" / "input.txt",
}

BPE_PATHS = {
    "shakespeare-tokens": dir_from_env()
    / "shakespeare-tokens"
    / "char"
    / "tokens.tiktoken",
}


def save_shakespeare_char_bpe():
    input_txt, bpe_path = (
        INPUT_PATHS["shakespeare-tokens"],
        BPE_PATHS["shakespeare-tokens"],
    )
    data = input_txt.read_text()
    chars = sorted(list(set(data)))  # get unique characters
    mergeable_ranks = {bytes(ch, "utf-8"): i for i, ch in enumerate(chars)}
    dump_tiktoken_bpe(mergeable_ranks, str(bpe_path))


PAT_STR = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
ENDOFTEXT = "<|endoftext|>"


def shakespeare_char():
    mergeable_ranks = load_tiktoken_bpe(
        str(dir_from_env() / "shakespeare-tokens" / "char" / "tokens.tiktoken")
    )
    return {
        "name": "shakespeare_char",
        "explicit_n_vocab": len(mergeable_ranks) + 1,
        "pat_str": PAT_STR,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: len(mergeable_ranks)},
    }


ENCODING_CONSTRUCTORS = {
    "shakespeare_char": shakespeare_char,
}

if __name__ == "__main__":
    save_shakespeare_char_bpe()

    # Load and check: Works!
    print(shakespeare_char())
