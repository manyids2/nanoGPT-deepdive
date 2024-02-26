---
title: "dice"
date: 2024-01-27T13:21:21+01:00
draft: true
---

```python
metric_function(predicted, targets[, options]) -> TypedDict
```

```python
class StatsDice(TypedDict):
    n_tp: int
    n_fp: int
    n_tn: int
    n_fn: int
    dice: float

# Functional
dice(predicted, targets, options=None) -> StatsDice
```
