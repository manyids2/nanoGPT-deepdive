from pathlib import Path

import torch
from nanogpt_deepdive.experiment import Experiment
from nanogpt_deepdive.train import Trainer


def get_parser():
    parser = argparse.ArgumentParser(
        description="Sample predictions from saved checkpoint."
    )
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument("run", type=str, help="Run name of the experiment")
    parser.add_argument("ckpt_name", type=str, help="Name of checkpoint")
    parser.add_argument(
        "--startfile",
        type=Path,
        default=Path(),
        help="File with start for prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="^",
        help="Prompt to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples (default: 16)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature (default: 0.5)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=16,
        help="Use only top K tokens (default: 16)",
    )
    return parser


class Sampler:
    expt: Experiment
    t: Trainer

    def __init__(
        self,
        name: str,
        run: str,
        ckpt_name: str,
        startfile: Path,
        prompt: str,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> None:
        self.name = name
        self.run = run
        self.ckpt_name = ckpt_name
        self.startfile = startfile
        self.prompt = prompt
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        expt = Experiment(self.name, self.run)
        t = Trainer(expt=expt, ckpt_path=expt.rundir / f"{self.ckpt_name}.pt")
        t.model.eval()
        pprint(t.cfg)

        # Save references
        self.t = t
        self.expt = expt

        # Extract encode and decode from meta
        meta = t.data.meta
        assert meta
        stoi, itos = meta["stoi"], meta["itos"]
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join([itos[i] for i in l])

        # Load from startfile if necessary
        # self.start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        self.start = "^"  # Our start token
        if self.startfile.exists() & self.startfile.is_file():
            print(f"Using prompt from: {self.startfile}")
            self.start = self.startfile.read_text()
        elif self.prompt != "":
            print(f"Using prompt: {self.prompt}")
            self.start = self.prompt

        # Encode the beginning of the prompt
        start_ids = self.encode(self.start)
        self.x = torch.tensor(start_ids, dtype=torch.long, device=t.cfg.device)[
            None, ...
        ]

        # Run generation
        self.ctx = t.get_ctx()

    @torch.no_grad()
    def generate(self):
        with self.ctx:
            y = self.t.model.generate(
                self.x,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(self.decode(y[0].tolist()))
            print("---------------")


if __name__ == "__main__":
    from pprint import pprint
    import argparse

    # Load experiment and checkpoint
    args = get_parser().parse_args()
    s = Sampler(**args.__dict__)

    # Run for num_samples
    for _ in range(args.num_samples):
        s.generate()
