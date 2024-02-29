import yaml
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


def runs(name: str, logdir: Path) -> list[str]:
    return [d.stem for d in (logdir / name).iterdir() if d.is_dir()]


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_shell_cmd(cmd: list[str]) -> tuple[int, list[str]]:
    from subprocess import Popen, PIPE

    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    output = stdout.decode("utf-8").splitlines()
    return p.returncode, output


class Experiment:
    logdir: Path

    name: str
    namedir: Path

    run: str
    rundir: Path

    def __init__(
        self,
        name: str,
        run: str,
        logdir: Path = dir_from_env("NANOGPT_LOGDIR"),
        create: bool = False,
        debug: bool = False,
    ) -> None:
        self.name = name
        self.run = run
        self.logdir = Path(logdir)
        self.namedir = self.logdir / name
        self.rundir = self.logdir / name / run
        self.debug = debug
        if create:
            self.rundir.mkdir(parents=True, exist_ok=True)
        else:
            assert self.rundir.exists(), f"No such dir: {self.rundir}"

    def init_dirs(self):
        (self.rundir / "checkpoints").mkdir(exist_ok=True)
        (self.rundir / "models").mkdir(exist_ok=True)

    def get_cfg(self) -> dict:
        with open(self.rundir / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def save_cfg(self, cfg: dict) -> None:
        # Get current git status
        status, _ = run_shell_cmd(["git", "diff-index", "--quiet", "HEAD", "--"])
        if (status != 0) & (not self.debug):
            raise RuntimeError("Git is dirty")
        _, commit = run_shell_cmd(["git", "rev-parse", "HEAD"])
        cfg["commit"] = commit

        # Dump yaml
        with open(self.rundir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    def get_cfg_from_srcdir(self):
        # Load baseline config
        srcdir = dir_from_env("NANOGPT_SRCDIR") / "experiments"
        base = load_yaml(srcdir / self.name / "baseline.yaml")

        # Update base by globals in runs file
        runs = load_yaml(srcdir / self.name / "runs.yaml")
        base.update(runs["__global__"]) if "__global__" in runs else 0

        # Update with run specific info
        assert (
            self.run in runs
        ), f"Could not find run: {self.run} ({srcdir / self.name / 'runs.yaml'})"
        base.update(runs[self.run])
        return base
