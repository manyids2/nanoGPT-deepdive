from typing import Optional
from pathlib import Path
import os
from contextlib import nullcontext

import torch
from torch.distributed import init_process_group

from nanoGPT.model import GPT
from nanogpt_deepdive.config import Config, GPTConfig
from nanogpt_deepdive.experiment import Experiment, dir_from_env
from nanogpt_deepdive.dataset import Dataset


class Trainer:
    expt: Experiment
    cfg: Config
    data: Dataset
    model: GPT

    def __init__(
        self,
        expt: Experiment,
        cfg: Optional[Config] = None,
        data: Optional[Dataset] = None,
        model: Optional[GPT] = None,
        datadir: Path = dir_from_env("NANOGPT_DATADIR"),
    ) -> None:
        self.expt = expt
        self.cfg = cfg if cfg else self.get_cfg()

        # Initialize stuff
        self.init_ddp()
        self.init_tokens_per_iter()
        self.init_cuda()

        # TODO: How to handle if above stuff is not yet initialized?
        # Make into class methods that accept config?
        self.data = data if data else self.get_data(datadir)
        self.model = model if model else self.get_model()

    def get_cfg(self) -> Config:
        return Config(**self.expt.get_cfg())

    def save_cfg(self) -> None:
        self.expt.save_cfg(self.cfg.__dict__)

    def get_data(self, datadir: Path) -> Dataset:
        cfg = self.cfg
        data = Dataset(
            name=cfg.data_name,
            run=cfg.data_run,
            datadir=datadir,
            block_size=cfg.block_size,
            batch_size=cfg.batch_size,
            device=cfg.device,
            device_type=cfg.device_type,
        )
        # NOTE: meh :/
        cfg.meta_vocab_size = data.meta_vocab_size
        return data

    def get_model(self) -> GPT:
        # TODO: support resume, etc
        cfg = self.cfg
        model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.block_size,
            bias=cfg.bias,
            vocab_size=None,
            dropout=cfg.dropout,
        )  # start with model_args from command line
        assert cfg.init_from == "scratch"
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if cfg.meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 65")
        model_args["vocab_size"] = (
            cfg.meta_vocab_size if cfg.meta_vocab_size is not None else 65
        )
        gptconf = GPTConfig(**model_args)  # type: ignore
        model = GPT(gptconf)
        # crop down the model block size if desired, using model surgery
        if cfg.block_size < model.config.block_size:
            model.crop_block_size(cfg.block_size)
            model_args[
                "block_size"
            ] = cfg.block_size  # so that the checkpoint will have the right value
        model.to(cfg.device)
        return model

    def init_ddp(self):
        cfg = self.cfg
        # various inits, derived attributes, I/O setup
        ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if ddp:
            init_process_group(backend=cfg.backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            cfg.ddp_world_size = int(os.environ["WORLD_SIZE"])
            cfg.device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(cfg.device)

            # this process will do logging, checkpointing etc.
            cfg.master_process = ddp_rank == 0
            cfg.seed_offset = ddp_rank  # each process gets a different seed

            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert cfg.gradient_accumulation_steps % cfg.ddp_world_size == 0
            cfg.gradient_accumulation_steps //= cfg.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            cfg.master_process = True
            cfg.seed_offset = 0
            cfg.ddp_world_size = 1

    def init_tokens_per_iter(self):
        cfg = self.cfg
        # Compute tokens per training iteration
        cfg.tokens_per_iter = (
            cfg.gradient_accumulation_steps
            * cfg.ddp_world_size
            * cfg.batch_size
            * cfg.block_size
        )
        assert cfg.tokens_per_iter > 0

    def init_cuda(self):
        cfg = self.cfg
        # Not sure what this is doing
        # if cfg.master_process:
        #     os.makedirs(cfg.logdir, exist_ok=True)
        torch.manual_seed(cfg.seed + cfg.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        cfg.device_type = (
            "cuda" if "cuda" in cfg.device else "cpu"
        )  # for later use in torch.autocast

    def get_ctx(self):
        cfg = self.cfg
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]

        return (
            nullcontext()
            if cfg.device_type == "cpu"
            else torch.amp.autocast(device_type=cfg.device_type, dtype=ptdtype)
        )


if __name__ == "__main__":
    from rich import print

    # expt = Experiment("debug", "barebones", create=True)
    # cfg = Config()
    # t = Trainer(expt=expt, cfg=cfg)
    # t.save_cfg()

    expt = Experiment("debug", "barebones")
    t = Trainer(expt=expt)
    print(t.cfg.__dict__)
