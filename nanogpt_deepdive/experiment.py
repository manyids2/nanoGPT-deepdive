from pathlib import Path


def dir_from_env(var: str = "NANOGPT_LOGDIR"):
    import os

    _dir = os.environ.get(var)
    if _dir:
        dir = Path(_dir)
        dir.mkdir(parents=True, exist_ok=True)
        return dir
    else:
        raise KeyError(f"Env var not found: {var}")


def expts(logdir: Path) -> list[str]:
    return [d.stem for d in logdir.iterdir() if d.is_dir()]


def runs(expt: str, logdir: Path) -> list[str]:
    return [d.stem for d in (logdir / expt).iterdir() if d.is_dir()]


class Experiment:
    logdir: Path

    expt: str
    exptdir: Path

    run: str
    rundir: Path

    def __init__(
        self,
        name: str,
        run: str,
        logdir: Path = dir_from_env("NANOGPT_LOGDIR"),
        create: bool = False,
    ) -> None:
        self.name = name
        self.run = run
        self.logdir = Path(logdir)
        self.namedir = self.logdir / name
        self.rundir = self.logdir / name / run
        if create:
            self.rundir.mkdir(parents=True, exist_ok=True)
        else:
            assert self.rundir.exists()

    def init_dirs(self):
        (self.rundir / "checkpoints").mkdir(exist_ok=True)
        (self.rundir / "models").mkdir(exist_ok=True)
