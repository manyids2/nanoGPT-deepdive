import numpy as np
import pandas as pd
from tqdm import tqdm

from nanogpt_deepdive.dbs import save_db
from nanogpt_deepdive.sample import Sampler


def gen_combinations(
    prompts: list[str],
    fracs: list[float],
):
    rows = []
    for idx, prompt in enumerate(prompts):
        for frac in fracs:
            _prompt = prompt[: int(frac * len(prompt))]
            _expected = prompt[len(_prompt) :]
            max_new_tokens = len(prompt) - len(_prompt)
            row = {
                "idx": idx,
                "frac": frac,
                "prompt": prompt,
                "_prompt": _prompt,
                "_expected": _expected,
                "max_new_tokens": max_new_tokens,
            }
            rows.append(row)
    return rows


def get_tries(
    temps: list[float],
    n_tries: int,
):
    rows = []
    for t, temp in enumerate(temps):
        for i in range(n_tries):
            row = {
                "i": i,
                "t": t,
                "temp": temp,
            }
            rows.append(row)
    return rows


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 5
    s = Sampler(
        name=sys.argv[1],
        run=sys.argv[2],
        ckpt_name=sys.argv[3],
        max_new_tokens=64,
    )

    print(f"block_size: {s.t.cfg.block_size}")
    block_size = s.t.cfg.block_size  # 256

    # Generate and save list of samples
    seed = int(sys.argv[4])
    np.random.seed(seed)
    n_samples = 10
    data = s.t.data.train_data
    n_train = len(data)
    idxs = np.random.randint(low=0, high=n_train - block_size, size=(n_samples,))
    samples = np.stack([(data[i : i + block_size]).astype(np.int64) for i in idxs])
    print(samples.shape)  # 10, 256

    # Decode to prompt
    n_tries = 100
    temps = (np.arange(1, 10) * 0.1).tolist()
    fracs = [0.9, 0.8, 0.7]

    assert s.t.data.meta
    prompts = [
        "".join([s.t.data.meta["itos"][i] for i in sample]) for sample in samples
    ]
    metrics = {}
    (s.expt.rundir / "val").mkdir(exist_ok=True)

    combinations = gen_combinations(prompts, fracs)

    for c in combinations:
        # Name for output file
        idx, frac, max_new_tokens = (
            c["idx"],
            c["frac"],
            c["max_new_tokens"],
        )
        name = f"{s.ckpt_name}-{seed}-{idx}-{frac:.2f}"
        path = s.expt.rundir / "val" / f"{name}.db"
        if path.exists():
            continue
        print(name)

        # Initialize result
        prompt, _prompt, _expected = (
            c["prompt"],
            c["_prompt"],
            c["_expected"],
        )
        result = (
            f"--{idx}--\n"
            f"--{idx, frac}--\n"
            "----\n"
            + prompt
            + "\n--prompt--\n"
            + _prompt
            + "\n--expected--\n"
            + _expected
        )
        print(result)
        s.set_prompt(_prompt)
        s.max_new_tokens = max_new_tokens

        df = pd.DataFrame(
            [
                {
                    "filename": name,
                    "name": s.name,
                    "run": s.run,
                    "ckpt_name": s.ckpt_name,
                    "seed": seed,
                    "frac": frac,
                    "full_prompt": prompt,
                    "prompt": _prompt,
                    "expected": _expected,
                }
            ]
        )
        save_db(df, path, "metadata", verbose=True)

        # Try for many temps
        tries = get_tries(temps, n_tries)
        pbar = tqdm(tries, total=len(tries), ncols=80)
        res_list = []
        rows = []
        for p in pbar:
            i, t, temp = p["i"], p["t"], p["temp"]
            s.temperature = temp

            # Run over n_tries and acc result
            response = s.generate()
            _response = response[len(_prompt) :]  # Cut to only necessary
            # TODO: binary search
            match = 0
            for ii in range(len(_response)):
                if _expected[:ii] == _response[:ii]:
                    match = ii
                else:
                    break

            row = {
                "i": i,
                "temp": temp,
                "response": _response,
                "match": match,
            }
            rows.append(row)

        # Save text and stats
        df = pd.DataFrame(rows)
        save_db(df, path, "results", verbose=True)
