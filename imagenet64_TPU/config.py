# =============================================================================
# config.py — loads configs/teacher.yaml and configs/student.yaml.
#
#   from config import load
#   cfg = load("configs/teacher.yaml")
#   cfg.model.ch        -> 128
#   cfg.tag.w_max       -> 4.0
#   cfg.loss.lambda_im  -> 1.0   (student only)
#
# Nested keys are reachable by attribute; the raw dict stays at cfg.raw.
# A student config also loads its teacher as cfg.teacher.
# =============================================================================

import os

import yaml


class Cfg(dict):
    """dict with recursive attribute access."""

    def __init__(self, d):
        super().__init__(d)
        for k, v in d.items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def load(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(raw)
    cfg.raw = raw

    tpath = raw.get("teacher_config")
    if tpath:
        if not os.path.isabs(tpath):
            base = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            tpath = os.path.join(base, tpath)
        cfg.teacher = load(os.path.normpath(tpath))

    _check(cfg, path)
    return cfg


def _check(cfg, path):
    """Fail loudly on the values the paper's claims depend on."""
    m, d = cfg.model, cfg.diffusion
    assert d.timesteps == 1000, f"{path}: timesteps must be 1000"
    assert d.beta_schedule == "cosine", f"{path}: schedule must be cosine"
    assert tuple(m.ch_mult) == (1, 2, 3, 4), f"{path}: ch_mult must be (1,2,3,4)"
    assert tuple(m.attn_resolutions) == (16, 8), f"{path}: attn must be (16,8)"
    assert cfg.tirt.snr_gamma == 5.0, f"{path}: TIRT gamma must be 5.0"

    t = cfg.tag
    assert (t.w_min, t.w_max, t.beta) == (1.0, 4.0, 5.0), \
        f"{path}: TAG must be (1.0, 4.0, 5.0)"

    if cfg.role == "student":
        lo = cfg.loss
        assert lo.lambda_im == lo.lambda_un == 1.0, \
            f"{path}: symmetric branch weights required (identity Gram matrix)"
        assert cfg.tirt.weight_objective == "frozen", \
            f"{path}: TIRT Transfer requires a frozen schedule"
        assert cfg.train.objective in ("dash", "no_lun", "composite"), \
            f"{path}: unknown objective {cfg.train.objective}"


if __name__ == "__main__":
    for p in ("configs/teacher.yaml", "configs/student.yaml"):
        c = load(p)
        print(f"  {p}: {c.name}  ch={c.model.ch}  nrb={c.model.num_res_blocks}  "
              f"params={c.model.params}  epochs={c.train.epochs}")
