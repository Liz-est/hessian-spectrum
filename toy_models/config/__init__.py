"""
config: typed, named experiment configuration for the vanilla_transformer toys.

Usage
-----
    from config import load, apply_overrides, build_optimizer, make_lr_fn

    cfg = load("imbalance_s1_sgd")            # a deep-copied ExperimentConfig
    cfg = apply_overrides(cfg, sys.argv[1:])  # apply --group.key=value flags

The first positional (non "--") CLI token, if present, selects the preset by
name; everything of the form --group.key=value overrides one field. Examples:

    python train_vanilla_transformer.py imbalance_s1_adamw
    python train_vanilla_transformer.py --optim.name=adamw --lr.learning_rate=3e-4
    python train_vanilla_transformer.py --train.max_iters=40   # quick smoke test

Type coercion follows the existing field's type (int/float/bool/str). bool
accepts True/False/1/0/yes/no. Tuple fields (e.g. optim.betas) accept a
comma-separated value: --optim.betas=0.9,0.999.
"""

from dataclasses import fields, is_dataclass

from .schema import (ExperimentConfig, ModelConfig, DataConfig, OptimConfig,
                     LRConfig, TrainConfig, AnalyzeConfig)
from .presets import EXPERIMENTS, DEFAULT, get
from .build import build_optimizer, make_lr_fn

__all__ = [
    "ExperimentConfig", "ModelConfig", "DataConfig", "OptimConfig",
    "LRConfig", "TrainConfig", "AnalyzeConfig",
    "EXPERIMENTS", "DEFAULT", "load", "apply_overrides",
    "build_optimizer", "make_lr_fn",
]


def load(name=None):
    """Return a deep-copied ExperimentConfig for `name` (or the DEFAULT preset)."""
    return get(name or DEFAULT)


def _coerce(cur, val):
    """Coerce string `val` to the type of the current value `cur`."""
    if isinstance(cur, dict):
        raise TypeError(
            "dict fields (e.g. train.ckpt_fracs) can't be set from the CLI; "
            "define a preset in config/presets.py instead"
        )
    if isinstance(cur, bool):
        return str(val).lower() in ("true", "1", "yes", "y")
    if isinstance(cur, tuple):
        parts = [p for p in str(val).split(",") if p != ""]
        elem = type(cur[0]) if cur else float
        return tuple(elem(p) for p in parts)
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    return str(val)


def _set_dotted(cfg, dotted, val):
    """Set cfg.<a>.<b>...=val, coercing to the existing field's type."""
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not (is_dataclass(obj) and hasattr(obj, p)):
            raise KeyError(f"no config group/field '{p}' in override '{dotted}'")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config field '{dotted}'")
    setattr(obj, leaf, _coerce(getattr(obj, leaf), val))


def apply_overrides(cfg, argv):
    """Apply CLI tokens to `cfg` in place and return it.

    A bare token (no leading --) selects/replaces the preset by name; it must
    appear before any override and there may be at most one. Tokens of the form
    --group.key=value override individual fields.
    """
    # a leading bare token re-selects the preset
    i = 0
    if argv and not argv[0].startswith("--"):
        cfg = get(argv[0])
        i = 1
    for arg in argv[i:]:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(
                f"bad config arg {arg!r} (want --group.key=value or a preset name)"
            )
        key, val = arg[2:].split("=", 1)
        _set_dotted(cfg, key, val)
    return cfg
