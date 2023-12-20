from rich import print
from nanoGPT.model import GPT, GPTConfig


class NumParams(GPTConfig):
    non_embedding: bool
    r: str
    s: float

    def __init__(self, **kwargs):
        relevant = {k: v for k, v in args.__dict__.items() if k in GPTConfig.__dict__}
        super().__init__(**relevant)

        self.non_embedding = kwargs["non_embedding"]
        self.set_precision(kwargs["precision"])

    def set_precision(self, precision):
        if precision == "0":
            s, r = 1e0, " "
        elif precision == "K":
            s, r = 1e3, "K"
        elif precision == "M":
            s, r = 1e6, "M"
        elif precision == "G":
            s, r = 1e9, "G"
        else:
            print(f"Unknown precision:{args.precision}, falling back to `M`")
            s, r = 1e6, "M"
        self.s, self.r = s, r

    @property
    def n_params(self) -> tuple[int, str]:
        wte = self.n_wte
        wpe = self.n_wpe
        block = self.n_block
        ln = self.n_ln
        n = wte[0] + wpe[0] + self.n_layer * block[0] + ln[0]
        if self.non_embedding:
            n -= self.n_wpe[0]

        return (
            n,
            (
                f"wte: {wte[1]}\n"
                f"wpe: {wpe[1]}\n"
                f"  h: {self.n_layer} * {block[0]/self.s}{self.r}{block[1]}\n"
                f" ln: {ln[1]}\n"
            ),
        )

    @property
    def n_lm_head(self) -> tuple[int, str]:
        return (
            self.vocab_size * self.n_embd,
            f"{self.vocab_size * self.n_embd/self.s}{self.r} (vocab_size, n_embd)",
        )

    @property
    def n_wte(self) -> tuple[int, str]:
        return (
            self.vocab_size * self.n_embd,
            f"{self.vocab_size * self.n_embd/self.s}{self.r} (vocab_size, n_embd)",
        )

    @property
    def n_wpe(self) -> tuple[int, str]:
        return (
            self.block_size * self.n_embd,
            f"{self.block_size * self.n_embd/self.s}{self.r} (block_size, n_embd)",
        )

    @property
    def n_ln(self) -> tuple[int, str]:
        n, r = self.n_embd, f"{self.n_embd /self.s}{self.r} (n_embd)"
        if self.bias:
            n += self.n_embd
            r = f"{n /self.s}{self.r} (n_embd + n_embd)"
        return (n, r)

    @property
    def n_block(self) -> tuple[int, str]:
        attn = self.n_block_attn
        mlp = self.n_block_mlp
        ln_1 = self.n_block_ln_1
        ln_2 = self.n_block_ln_2
        indent = "\t"
        return (
            attn[0] + ln_1[0] + mlp[0] + ln_2[0],
            (
                f"\n{indent}attn: {attn[1]}\n"
                f"{indent}ln_1: {ln_1[1]}\n"
                f"{indent} mlp: {mlp[1]}\n"
                f"{indent}ln_2: {ln_2[1]}"
            ),
        )

    @property
    def n_block_ln_1(self) -> tuple[int, str]:
        n, r = self.n_embd, f"{self.n_embd /self.s}{self.r} (n_embd)"
        if self.bias:
            n += self.n_embd
            r = f"{n /self.s}{self.r} (n_embd + n_embd)"
        return (n, r)

    @property
    def n_block_ln_2(self) -> tuple[int, str]:
        n, r = self.n_embd, f"{self.n_embd /self.s}{self.r} (n_embd)"
        if self.bias:
            n += self.n_embd
            r = f"{n /self.s}{self.r} (n_embd + n_embd)"
        return (n, r)

    @property
    def n_block_attn_attn(self) -> tuple[int, str]:
        n = self.n_embd * 3 * self.n_embd
        r = f"{n /self.s}{self.r} (n_embd, 3*n_embd)"
        if self.bias:
            n += 3 * self.n_embd
            r = f"{n /self.s}{self.r} ((n_embd, 3*n_embd) + 3*n_embd)"
        return (n, r)

    @property
    def n_block_attn_proj(self) -> tuple[int, str]:
        n = self.n_embd * self.n_embd
        r = f"{n / self.s}{self.r} (n_embd, n_embd)"
        if self.bias:
            n += self.n_embd
            r = f"{n / self.s}{self.r} ((n_embd, n_embd) + n_embd)"
        return (n, r)

    @property
    def n_block_attn(self) -> tuple[int, str]:
        attn = self.n_block_attn_attn
        proj = self.n_block_attn_proj
        indent = "\t\t"
        return (
            attn[0] + proj[0],
            (
                f"{(attn[0]+proj[0])/self.s}{self.r}\n"
                f"{indent}{attn[1]}\n"
                f"{indent}{proj[1]}"
            ),
        )

    @property
    def n_block_mlp_fc(self) -> tuple[int, str]:
        n = self.n_embd * 4 * self.n_embd
        r = f"{n / self.s}{self.r} (n_embd, 4*n_embd)"
        if self.bias:
            n += 4 * self.n_embd
            r = f"{n/self.s}{self.r} ((n_embd, 4*n_embd) + 4*n_embd)"
        return (n, r)

    @property
    def n_block_mlp_proj(self) -> tuple[int, str]:
        n = 4 * self.n_embd * self.n_embd
        r = f"{n /self.s}{self.r} (4*n_embd, n_embd)"
        if self.bias:
            n += self.n_embd
            r = f"{n /self.s}{self.r} ((4*n_embd, n_embd) + n_embd)"
        return (n, r)

    @property
    def n_block_mlp(self) -> tuple[int, str]:
        fc = self.n_block_mlp_fc
        proj = self.n_block_mlp_proj
        indent = "\t\t"
        return (
            fc[0] + proj[0],
            (
                f"{(fc[0]+proj[0])/self.s}{self.r}\n"
                f"{indent}{fc[1]}\n"
                f"{indent}{proj[1]}"
            ),
        )

    def __repr__(self) -> str:
        return (
            f"NumParams:\n"
            f"  block_size: {self.block_size}\n"
            f"  vocab_size: {self.vocab_size}\n"
            f"  n_layer   : {self.n_layer}\n"
            f"  n_head    : {self.n_head}\n"
            f"  n_embd    : {self.n_embd}\n"
            f"  dropout   : {self.dropout}\n"
            f"  bias      : {self.bias}\n\n"
            f"  non_embedding: {self.non_embedding}\n"
            f"  precision    : {self.r, self.s}\n"
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    s, r = 1e6, "M"

    parser = ArgumentParser()
    parser.add_argument
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=str, default="true")
    parser.add_argument("--precision", type=str, default="M")
    parser.add_argument("--non_embedding", type=str, default="true")

    args = parser.parse_args()

    args.bias = args.bias.lower() in ["true", "1"]
    args.non_embedding = args.non_embedding.lower() in ["true", "1"]

    n_params = NumParams(**args.__dict__)
    print(n_params)
    print(n_params.n_params[1])

    relevant = {k: v for k, v in args.__dict__.items() if k in GPTConfig.__dict__}
    gptconf = GPTConfig(**relevant)
    print(gptconf)
    model = GPT(gptconf)
    print()

    n_params = n_params.n_params[0]
    print()
    print(f"Total          : {n_params / s:.2f}{r}")
    print(f"Orig           : {model.get_num_params(args.non_embedding) / s:.2f}{r}")
    print(f"Total - Orig   : {n_params - model.get_num_params(args.non_embedding)}")
