---
title: "tiktoken"
date: 2024-01-27T13:21:21+01:00
draft: true
---

## Study of [tiktoken](https://github.com/openai/tiktoken)

`tiktoken` is a _Byte Pair Endcoding_(BPE) tokenizer. It is used to compress
text, i.e. an array of bytes, to an array of integers.

### Checking available tokenizers

Available tokenizers {{< sidenote-label available-tokenizers >}}:
{{< sidenote available-tokenizers >}}
[tiktoken_ext/openai_public](https://github.com/openai/tiktoken/blob/9e79899bc248d5313c7dd73562b5e211d728723d/tiktoken_ext/openai_public.py#L82C1-L88C2)
{{< /sidenote >}}

```python
#
ENCODING_CONSTRUCTORS = {
    "gpt2": gpt2,
    "r50k_base": r50k_base,
    "p50k_base": p50k_base,
    "p50k_edit": p50k_edit,
    "cl100k_base": cl100k_base,
}
```

Let's inspect each of the tokenizers:

```python
import tiktoken
import pandas as pd
rows = []
ENCODINGS = ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base"]
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
    }
    rows.append(row)
df = pd.DataFrame(rows)
print(df.to_markdown())
```

|     | name        | n_vocab | max_token_value | eot_token | special_tokens_set                                                                                   | pat_str                                                                                                                         | mergeable_ranks |
| --: | :---------- | ------: | --------------: | --------: | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | --------------: |
|   0 | gpt2        |   50257 |           50256 |     50256 | {'<\|endoftext\|>'}                                                                                  | 's\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+                                           |           50256 |
|   1 | r50k_base   |   50257 |           50256 |     50256 | {'<\|endoftext\|>'}                                                                                  | 's\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+                                           |           50256 |
|   2 | p50k_base   |   50281 |           50280 |     50256 | {'<\|endoftext\|>'}                                                                                  | 's\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+                                           |           50280 |
|   3 | p50k_edit   |   50284 |           50283 |     50256 | {'<\|fim_middle\|>', '<\|fim_suffix\|>', '<\|fim_prefix\|>', '<\|endoftext\|>'}                      | 's\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+                                           |           50280 |
|   4 | cl100k_base |  100277 |          100276 |    100257 | {'<\|fim_prefix\|>', '<\|fim_suffix\|>', '<\|endoftext\|>', '<\|endofprompt\|>', '<\|fim_middle\|>'} | (?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n]_\|\s_[\r\n]+\|\s+(?!\S)\|\s+ |          100256 |

So other than `cl100k_base`, 50k seems to be the vocab size. This handles all
languages, and unicode. Let's print all the tokens to see if we can make some
useful groupings.

TODO: List of all tokens for each encoding: [csv](./tokens.csv)

Wierdly, when we run it, `cl100k_base` seems to only have a vocab of 100261,

```python
import tiktoken
enc = tiktoken.get_encoding(name)
print(enc.decode([100260]))
# <|fim_middle|>
print(enc.decode([100261]))
# pyo3_runtime.PanicException: no entry found for key
```

Quickly checking all the vocabs, we see:

```python
import tiktoken
ENCODINGS = ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base"]
for name in ENCODINGS:
    enc = tiktoken.get_encoding(name)
    errors = []
    for i in range(enc.n_vocab):
        try:
            enc.decode([i])
        except:
            errors.append(i)
    print(f"{enc.name}: {errors} (len: {len(errors)})")

# gpt2: [] (len: 0)
# r50k_base: [] (len: 0)
# p50k_base: [] (len: 0)
# p50k_edit: [] (len: 0)
# cl100k_base: [100256, 100261, ..., 100275] (len: 16)
```

Not sure why this is the case {{< sidenote-label bug-report >}}.
{{< sidenote bug-report >}}
[Bug report is live since 4 March, 2023](https://github.com/openai/tiktoken/issues/47).
{{< /sidenote >}}

Let us see how to inspect **all** ( well, almost all ) the tokens from each
encoding. We save the rest in chunks of 1000 tokens to csv. Unfortunately, we
cannot save the table to a csv using `pandas`:

```bash
_csv.Error: need to escape, but no escapechar set
```

So, let's just do it manually and clean up later:

```python
import tiktoken, math
ENCODINGS = ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base"]
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
```

Now we need to also clean up the non-printable to present here,
and also to help `visidata` open the file.

```bash
# Dump all tokens to a temp csv file
python nanogpt_deepdive/data/tiktoken-summary.py > temp.csv

echo name,start,end,tokens > tokens.csv
for i in $(cat temp.csv);do
  # Clean up non-printable characters
  # BUG: Still cl100k_base is messed up in visidata
  echo $i | tr -cs [:print:] >> tokens.csv
done
rm temp.csv

vd tokens.csv
```

The full csv file is at [all-tokens.csv](all-tokens.csv).

Training on the whole internet certainly has bizarre consequences for our
tokenizers. Furthermore, the above tokenizers share a lot of their vocabulary.
However, `cl100k_base` behaves very wierdly.

### Update

`mergeable_ranks` actually has the whole dict we are looking for.

```python
import tiktoken
import pandas as pd
rows = []
ENCODINGS = ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base"]
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
df.to_csv("mergeable_ranks.csv")
```

This makes it much easier to visualize in `visidata`. Now we can check things
like how many tokens are common to which encodings, venn diagram of tokens,
encodings etc.

1. Right away, we see the discrepence in vocab size and dict of encodings:

   | name        |  count | n_vocab |
   | ----------- | -----: | ------: |
   | gpt2        |  50256 |   50257 |
   | r50k_base   |  50256 |   50257 |
   | p50k_base   |  50280 |   50281 |
   | p50k_edit   |  50280 |   50284 |
   | cl100k_base | 100256 |  100277 |

2. Tokens common to `n` vocabularies:

   | count | count |
   | ----: | ----: |
   |     5 | 43066 |
   |     4 |  7190 |
   |     3 |    24 |
   |     1 | 57166 |

3. Are all tokens from `gpt2` and `r50k_base` the same? Yes.

   Tokens common to `n` vocabularies among `gpt2` and `r50k_base`:

   | count | count |
   | ----: | ----: |
   |     2 | 50256 |

4. Are all tokens from `r50k_base` present in `p50k_base`? Yes.

   | count | count |
   | ----: | ----: |
   |     2 | 50256 |
   |     1 |    24 |

   Which are the extra 24 tokens? The answer is out there in space.

   ```txt
   |name       |idx    |token                         |
   |-----------|-------|------------------------------|
   |p50k\_base |50257  |b'  '                         |
   |p50k\_base |50258  |b'   '                        |
   |p50k\_base |50259  |b'    '                       |
   |p50k\_base |50260  |b'     '                      |
   |p50k\_base |50261  |b'      '                     |
   |p50k\_base |50262  |b'       '                    |
   |p50k\_base |50263  |b'        '                   |
   |p50k\_base |50264  |b'         '                  |
   |p50k\_base |50265  |b'          '                 |
   |p50k\_base |50266  |b'           '                |
   |p50k\_base |50267  |b'            '               |
   |p50k\_base |50268  |b'             '              |
   |p50k\_base |50269  |b'              '             |
   |p50k\_base |50270  |b'               '            |
   |p50k\_base |50271  |b'                '           |
   |p50k\_base |50272  |b'                 '          |
   |p50k\_base |50273  |b'                  '         |
   |p50k\_base |50274  |b'                   '        |
   |p50k\_base |50275  |b'                    '       |
   |p50k\_base |50276  |b'                     '      |
   |p50k\_base |50277  |b'                      '     |
   |p50k\_base |50278  |b'                       '    |
   |p50k\_base |50279  |b'                        '   |
   |p50k\_base |50280  |b'                         '  |
   ```

5. Ok! So then the disconnect between n_vocab and len(mergeable_ranks) is solved?

   It is from the special_tokens, mostly.

   | name        | n_vocab | max_token_value | eot_token | mergeable_ranks | mergeable_ranks + special_tokens |
   | :---------- | ------: | --------------: | --------: | --------------: | -------------------------------: |
   | gpt2        |   50257 |           50256 |     50256 |           50256 |                            50257 |
   | r50k_base   |   50257 |           50256 |     50256 |           50256 |                            50257 |
   | p50k_base   |   50281 |           50280 |     50256 |           50280 |                            50281 |
   | p50k_edit   |   50284 |           50283 |     50256 |           50280 |                            50284 |
   | cl100k_base |  100277 |          100276 |    100257 |          100256 |                           100261 |

   Though still not resolved for the `cl100k_base` case.

### Creating new encoding

Let's try to make an encoding for `shakespeare-char`.

```python
import tiktoken
from rich import inspect
PAT_STR = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
mergeable_ranks = {b'a': 65, b'b': 66}
enc = tiktoken.Encoding(
    name="shakespeare-char",
    pat_str=PAT_STR,
    mergeable_ranks=mergeable_ranks,
    special_tokens={'<|endoftext|>': len(mergeable_ranks)}
)
inspect(enc)

# Can only decode non-special tokens
print(enc.decode([65, 66])) # ab

# Can only decode non-special tokens
print(enc.encode("ab")) # [65, 66]
print(enc.encode("ab<|endoftext|>"))
# ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
# To allow ...
```

Thus, it is easy to set up our own encoding, so we dont have to keep track of
metadata, etc. So for our own encodings, let us just register it with `tiktoken`.

```python
from pathlib import Path
input_txt = Path("input.txt") # path to tiny shakespeare
data = input_txt.read_text()
chars = sorted(list(set(data))) # get unique characters
mergeable_ranks = {bytes(ch, 'utf-8'): i for i, ch in enumerate(chars)}

import tiktoken
PAT_STR = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
enc = tiktoken.Encoding(
    name="shakespeare-char",
    pat_str=PAT_STR,
    mergeable_ranks=mergeable_ranks,
    special_tokens={'<|endoftext|>': len(mergeable_ranks)} # Otherwise some keyerror shite
)

from rich import inspect
inspect(enc)
# ╭─── <class 'tiktoken.core.Encoding'> ────╮
# │ ╭─────────────────────────────────────╮ │
# │ │ <Encoding 'shakespeare-char'>       │ │
# │ ╰─────────────────────────────────────╯ │
# │                                         │
# │          eot_token = 65                 │
# │    max_token_value = 65                 │
# │            n_vocab = 66                 │
# │               name = 'shakespeare-char' │
# │ special_tokens_set = {'<|endoftext|>'}  │
# ╰─────────────────────────────────────────╯
```

### Barebones

Lets check how each of them encodes some simple examples:

```yaml
hello: []

hi: []

go: []
going: []
gone: []
gonna: []
golang: []

 go: []
 go : []
  "go": []
  'go': []
```

TODO: General observations:

### API

Extremely simple repo, just a dozen files.

```bash
 tiktoken
├──  Cargo.toml
├──  CHANGELOG.md
├──  LICENSE
├──  MANIFEST.in
├──  perf.svg
├──  pyproject.toml
├──  README.md
├──  scripts
│  ├──  benchmark.py
│  └──  redact.py
├──  setup.py
├──  src
│  └──  lib.rs
├──  tests
│  ├──  __init__.py
│  ├──  test_encoding.py
│  ├──  test_helpers.py
│  ├──  test_misc.py
│  ├──  test_offsets.py
│  └──  test_simple_public.py
├──  tiktoken
│  ├──  __init__.py
│  ├──  _educational.py
│  ├──  core.py
│  ├──  load.py
│  ├──  model.py
│  ├──  py.typed
│  └──  registry.py
└──  tiktoken_ext
   └──  openai_public.py
```

Zooming in on the `tiktoken` subdirectory, we have:

```bash
  tiktoken
├──  __init__.py
├──  _educational.py
├──  core.py
├──  load.py
├──  model.py
├──  py.typed
└──  registry.py
```

Let us check out the apis defined in each module.

```python
# --- __init__.py ---

# This is the public API of tiktoken
from .core import Encoding as Encoding

# All this is loading correct encoding given model name like gpt2, gpt-4, davinci-002, etc.
from .model import encoding_for_model as encoding_for_model # This is just `get_encoding(encoding_name_for_model(model_name))`
from .model import encoding_name_for_model as encoding_name_for_model
from .registry import get_encoding as get_encoding
from .registry import list_encoding_names as list_encoding_names

# --- core ---

# imports
general: regex, functools, concurrent.futures.ThreadPoolExecutor
typing: AbstractSet, Collection, Literal, NoReturn, Optional, Union
# TODO: what does this do?
from tiktoken import _tiktoken

# MVP
class Encoding:
    # Public
    name: str
    max_token_value: int

    # Defined as methods:
    #   eot_token: int
    #   n_vocab: int

    # Private
    _pat_str: str
    _mergeable_ranks: dict[bytes, int]
    _special_tokens: dict[str, int]

    # Implemented in rust (src/lib.rs)
    _core_bpe: _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)

    # Basics
    __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: Optional[int] = None,
    ) -> None
    __repr__(self) -> str

    # Encoding
    encode_ordinary(self, text: str) -> list[int]
    encode(self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]
    encode_ordinary_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
    encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[list[int]]
    encode_with_unstable(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> tuple[list[int], list[list[int]]]
    encode_single_token(self, text_or_bytes: Union[str, bytes]) -> int

    # Decoding
    decode_bytes(self, tokens: list[int]) -> bytes
    decode(self, tokens: list[int], errors: str = "replace") -> str
    decode_single_token_bytes(self, token: int) -> bytes
    decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]
    decode_with_offsets(self, tokens: list[int]) -> tuple[str, list[int]]
    decode_batch(
        self,
        batch: list[list[int]],
        *, errors: str = "replace",
        num_threads: int = 8
    ) -> list[str]
    decode_bytes_batch(self, batch: list[list[int]], *, num_threads: int = 8) -> list[bytes]

    # Miscellaneous
    token_byte_values(self) -> list[bytes]
    special_tokens_set(self) -> set[str]
    eot_token(self) -> int # property
    n_vocab(self) -> int # property

    # Private
    _encode_single_piece(self, text_or_bytes: Union[str, bytes]) -> list[int]
    _encode_only_native_bpe(self, text: str) -> list[int]
    _encode_bytes(self, text: bytes) -> list[int]
```

Lets add comprehensive documentation for each of these functions, and test them
out for each of the defined encodings.

```python
# Classes and functions in tiktoken codebase

```

### References

- [openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
