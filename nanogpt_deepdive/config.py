from typing import Optional
from pathlib import Path
import torch
from dataclasses import dataclass


def dir_from_env(var: str = "NANOGPT_SRCDIR"):
    import os

    _dir = os.environ.get(var)
    if _dir:
        dir = Path(_dir)
        dir.mkdir(parents=True, exist_ok=True)
        return dir
    else:
        raise KeyError(f"Env var not found: {var}")


@dataclass
class Config:
    # I/O
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    # if True, script exits right after the first eval
    eval_only: bool = False
    # if True, always save a checkpoint after each eval
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # data
    data_name: str = "shakespeare-tokens"
    data_run: str = "char"
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    # if gradient_accumulation_steps > 1, this is the micro-batch size
    batch_size: int = 12
    block_size: int = 256
    vocab_size: int = 65
    seed: int = 1337
    meta_vocab_size: Optional[int] = None
    tokens_per_iter: int = 0

    # model
    n_layer: int = 5
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    variant: str = "sum"

    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    min_lr: float = 6e-5

    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device: str = "cuda"
    device_type: str = "cuda"

    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

    # Training on multiple gpus -
    # TODO: Not supported for now, will understand and add later
    ddp: bool = False
    master_process: bool = True
    seed_offset: int = 0
    ddp_rank: int = 1
    ddp_local_rank: int = 0
    ddp_world_size: int = 0

    tokens_per_iter = 0


if __name__ == "__main__":
    from rich import print

    cfg = Config()
    print(cfg.__dict__)
