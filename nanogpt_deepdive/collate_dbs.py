from nanogpt_deepdive.dbs import load_db, save_db
from nanogpt_deepdive.experiment import Experiment
import pandas as pd

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 4
    e = Experiment(
        name=sys.argv[1],
        run=sys.argv[2],
    )
    ckpt_name = sys.argv[3]

    val_dir = e.rundir / "val"
    if not val_dir.exists():
        exit(1)

    db_path = e.rundir / "val.db"
    db_paths = val_dir.glob(f"{ckpt_name}*.db")

    metadata = []
    results = []
    for path in db_paths:
        _metadata = load_db(path, "metadata")
        _results = load_db(path, "results")
        _results["filename"] = path.stem

        # Add metrics to metadata
        _metadata["match_min"] = int(_results.match.min())
        _metadata["match_max"] = int(_results.match.max())
        _metadata["match_mean"] = int(_results.match.mean())
        _metadata["match_median"] = int(_results.match.median())

        metadata.append(_metadata)
        results.append(_results)

    metadata = pd.concat(metadata)
    save_db(metadata, db_path, "metadata", verbose=True)

    results = pd.concat(results)
    save_db(results, db_path, "results", verbose=True)
