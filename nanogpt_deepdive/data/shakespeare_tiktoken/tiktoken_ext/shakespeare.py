from pathlib import Path
from tiktoken.load import load_tiktoken_bpe, dump_tiktoken_bpe


def dir_from_env(var: str = "NANOGPT_DATADIR"):
    import os

    _dir = os.environ.get(var)
    if _dir:
        return Path(_dir)
    else:
        raise KeyError(f"Env var not found: {var}")


def save_shakespeare_char_bpe():
    input_txt = dir_from_env() / "shakespeare-tokens" / "input.txt"
    bpe_path = dir_from_env() / "shakespeare-tokens" / "char" / "tokens.tiktoken"
    data = input_txt.read_text()
    chars = sorted(list(set(data)))  # get unique characters
    mergeable_ranks = {bytes(ch, "utf-8"): i for i, ch in enumerate(chars)}
    dump_tiktoken_bpe(mergeable_ranks, str(bpe_path))


def save_shakespeare_word_bpe():
    input_txt = dir_from_env() / "shakespeare-tokens" / "input.txt"
    bpe_path = dir_from_env() / "shakespeare-tokens" / "word" / "tokens.tiktoken"
    data = input_txt.read_text()
    lines = data.split("\n")
    words = []
    for line in lines:
        _words = line.split(" ")
        _words = [w for w in _words if len(w) > 0]
        # TODO: Further split by ["'", ",", "?", ...]
        words.extend(_words)
    words = sorted(list(set(words)))  # get unique words
    mergeable_ranks = {bytes(ch, "utf-8"): i for i, ch in enumerate(words)}
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


def shakespeare_word():
    mergeable_ranks = load_tiktoken_bpe(
        str(dir_from_env() / "shakespeare-tokens" / "word" / "tokens.tiktoken")
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
    # Save, load and check: Works!
    # save_shakespeare_char_bpe()
    # print(shakespeare_char())

    # Save, load and check: Works!
    save_shakespeare_word_bpe()
    print(shakespeare_word())
