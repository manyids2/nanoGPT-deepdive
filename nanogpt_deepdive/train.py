from typing import Optional, Union, Dict, Any
from pathlib import Path
import time
import math
from contextlib import nullcontext

import torch

from nanoGPT.model import GPT
from nanogpt_deepdive.config import Config, GPTConfig
from nanogpt_deepdive.experiment import Experiment, dir_from_env
from nanogpt_deepdive.dataset import Dataset


class Trainer:
    expt: Experiment
    cfg: Config
    data: Dataset
    model: GPT
    model_args: Dict[str, Union[int, str, float]]
    ckpt_path: Path
    checkpoint: Any

    def __init__(
        self,
        expt: Experiment,
        cfg: Optional[Config] = None,
        data: Optional[Dataset] = None,
        model: Optional[GPT] = None,
        ckpt_path: Optional[Path] = None,
        datadir: Path = dir_from_env("NANOGPT_DATADIR"),
    ) -> None:
        self.expt = expt
        self.cfg = cfg if cfg else self.get_cfg()

        # Initialize stuff
        self.init_tokens_per_iter()
        self.init_cuda()

        # TODO: How to handle if above stuff is not yet initialized?
        # Make into class methods that accept config?
        self.ckpt_path = ckpt_path if ckpt_path else self.expt.rundir / "ckpt.pt"
        self.data = data if data else self.get_data(datadir)
        self.model = model if model else self.get_model()
        self.model.to(self.cfg.device)

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
        cfg = self.cfg
        model_args = dict(
            n_layer=int(cfg.n_layer),
            n_head=int(cfg.n_head),
            n_embd=int(cfg.n_embd),
            block_size=int(cfg.block_size),
            bias=bool(cfg.bias),
            vocab_size=int(cfg.vocab_size),  # Make sure we know it!
            dropout=float(cfg.dropout),
        )  # start with model_args from command line

        # Convenience: Switch to resume if ckpt_path exists
        if self.ckpt_path.exists():
            cfg.init_from = "resume"

        checkpoint = None
        if cfg.init_from == "scratch":
            print("Initializing model from scratch")
            gptconf = GPTConfig(**model_args)  # type: ignore
            model = GPT(gptconf)
        elif cfg.init_from == "resume":
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"File not found: {self.ckpt_path}")
            checkpoint = torch.load(self.ckpt_path, map_location=cfg.device)
            checkpoint_model_args = checkpoint["model_args"]

            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                model_args[k] = checkpoint_model_args[k]

            # init a new model from scratch
            print(f"Initializing model from resume: {self.ckpt_path}")
            gptconf = GPTConfig(**model_args)  # type: ignore
            model = GPT(gptconf)

            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, _ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.checkpoint = checkpoint
        elif cfg.init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
            override_args = dict(dropout=cfg.dropout)
            model = GPT.from_pretrained(cfg.init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                model_args[k] = getattr(model.config, k)
        else:
            raise KeyError(f"Unknown init_from: {cfg.init_from}")

        self.model_args = model_args  # type: ignore

        # crop down the model block size if desired, using model surgery
        if cfg.block_size < model.config.block_size:
            model.crop_block_size(cfg.block_size)
            model_args[
                "block_size"
            ] = cfg.block_size  # so that the checkpoint will have the right value
        model.to(cfg.device)

        # compile the model, requires PyTorch 2.0
        print("compiling the model... (takes a ~minute)")
        model: GPT = torch.compile(model)  # type: ignore

        return model

    def init_tokens_per_iter(self):
        cfg = self.cfg
        # Compute tokens per training iteration
        # Not based on size of dataset
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

    def get_scaler(self):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.dtype == "float16"))  # type: ignore
        return scaler

    def get_optimizer(self):
        cfg = self.cfg
        optimizer = self.model.configure_optimizers(
            cfg.weight_decay,
            cfg.learning_rate,
            (cfg.beta1, cfg.beta2),
            cfg.device_type,
        )
        if cfg.init_from == "resume":
            optimizer.load_state_dict(self.checkpoint["optimizer"])
        self.checkpoint = None  # free up memory
        return optimizer

    @torch.no_grad()
    def estimate_loss(self, ctx):
        # helps estimate an arbitrarily accurate loss over either split using many batches
        cfg = self.cfg
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = self.data.get_batch(split)
                with ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        # learning rate decay scheduler (cosine with warmup)
        cfg = self.cfg
        warmup_iters = cfg.warmup_iters
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return cfg.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (cfg.lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def train(self):
        cfg = self.cfg
        iter_num = 0  # Needs override in case of resume
        best_val_loss = 1e9

        optimizer = self.get_optimizer()
        scaler = self.get_scaler()
        ctx = self.get_ctx()

        # training loop
        X, Y = self.data.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        running_mfu = -1.0

        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % cfg.eval_interval == 0 and cfg.master_process:
                losses = self.estimate_loss(ctx)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": cfg.__dict__,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": cfg.__dict__,
                        }
                        print(f"saving checkpoint to {self.expt.rundir}")
                        torch.save(
                            checkpoint,
                            self.expt.rundir
                            / f"ckpt-{int(iter_num / cfg.eval_interval)}.pt",
                        )
            if iter_num == 0 and cfg.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for _ in range(cfg.gradient_accumulation_steps):
                with ctx:
                    _, loss = self.model(X, Y)
                    loss = (
                        loss / cfg.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.data.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()  # type: ignore
            # clip the gradient
            if cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)  # type: ignore
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.log_interval == 0 and cfg.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * cfg.gradient_accumulation_steps  # type: ignore
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.model.estimate_mfu(
                        cfg.batch_size * cfg.gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > cfg.max_iters:
                break


if __name__ == "__main__":
    import sys
    from pprint import pprint

    assert len(sys.argv) == 3
    expt = Experiment(sys.argv[1], sys.argv[2], create=True)
    cfg = Config(**expt.get_cfg_from_srcdir())
    t = Trainer(expt=expt, cfg=cfg)
    pprint(t.cfg.__dict__)
    t.save_cfg()

    print(f"\n\nUsing CUDA: {cfg.device == 'cuda'} ({cfg.device})\n\n")
    t.train()
