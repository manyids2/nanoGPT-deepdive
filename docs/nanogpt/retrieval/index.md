---
title: "retrieval"
date: 2024-01-27T13:21:21+01:00
draft: true
---

## What does GPT remember

Testing on tiny shakespeare, we train nanogpt with default params, and generate
samples given prompts from the training set itself.

We measure how often the generated sample is exactly the same as the
continuation in the text.

- Over 10 samples
  - prompt fraction from .25 to .75
    - temperature from 0.1 to 0.9
      - for 100 attempts
        - generate rest of sample
        - add to positives list with params

Sums for each case.

| expt          | fraction | temp | hit (/10) | total hits |
| ------------- | -------- | ---- | --------- | ---------- |
| char          | 0.9      | 0.1  | 3         | 23         |
| tiktoken-char | 0.9      | 0.1  | 3         | 23         |
| tiktoken-word | 0.9      | 0.1  | 3         | 23         |


| expt   | fraction | temp | hit (/10) | total hits |
| ------ | -------- | ---- | --------- | ---------- |
| sum    | 0.9      | 0.1  | 3         | 23         |
| concat | 0.9      | 0.1  | 3         | 23         |
| cart   | 0.9      | 0.1  | 3         | 23         |
