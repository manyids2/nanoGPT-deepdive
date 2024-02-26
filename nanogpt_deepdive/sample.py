from pathlib import Path
from pprint import pprint

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
        startfile: Path = Path(""),
        prompt: str = "",
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_k: int = 0,
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

        # Context - not sure why
        self.ctx = t.get_ctx()

        # Save references
        self.t = t
        self.expt = expt

        # Extract encode and decode from meta
        meta = t.data.meta
        assert meta
        stoi, itos = meta["stoi"], meta["itos"]
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda x: "".join([itos[i] for i in x])

        # Load from startfile if necessary
        if self.startfile.exists() & self.startfile.is_file():
            self.prompt = self.startfile.read_text()
        self.set_prompt(prompt)

    def set_prompt(self, prompt: str = "^"):
        self.start = prompt
        start_ids = self.encode(prompt)
        self.x = torch.tensor(start_ids, dtype=torch.long, device=self.t.cfg.device)[
            None, ...
        ]

    @torch.no_grad()
    def generate(self) -> str:
        with self.ctx:
            y = self.t.model.generate(
                self.x,
                self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k if self.top_k > 0 else None,
            )
            response = self.decode(y[0].tolist())
        return response

    @torch.no_grad()
    def generate_for_prompt(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_k: int = 0,
    ) -> str:
        start_ids = self.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.t.cfg.device)[
            None, ...
        ]
        with self.ctx:
            y = self.t.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )
            response = self.decode(y[0].tolist())
        return response


if __name__ == "__main__":
    import argparse

    # Load experiment and checkpoint
    args = get_parser().parse_args()
    s = Sampler(**args.__dict__)

    # Run for num_samples
    for _ in range(args.num_samples):
        s.generate()
