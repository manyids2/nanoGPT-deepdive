---
title: 'tokens'
date: 2024-01-27T13:21:21+01:00
draft: true
---

## Introduction to tokenizers


Influence of tokenizer on generation of shakespeare.

### tiktoken

```yaml
expt: shakespeare-tokens
runs:
  - tiktoken_small
  - tiktoken_medium
  - tiktoken_large
checkpoints:
  - epoch-100
  - epoch-500
  - epoch-1000
```

### char

```yaml
expt: shakespeare-tokens
runs:
  - char_small
  - char_medium
  - char_large
checkpoints:
  - epoch-100
  - epoch-500
  - epoch-1000
```

### word

```yaml
expt: shakespeare-tokens
runs:
  - word_small
  - word_medium
  - word_large
checkpoints:
  - epoch-100
  - epoch-500
  - epoch-1000
```
