"""Authoritative environment construction for training and evaluation."""

from generals.core.env import GeneralsEnv


def make_env(cfg, *, pool_size=None, distance=None, exact_competition=False):
    """Build an environment without accidentally dropping ruleset modifiers.

    Competition mode is always constructed from the named preset. Curriculum
    may then override only the general-distance distribution. Passing
    ``exact_competition=True`` ignores curriculum overrides.
    """
    size = cfg.pool_size if pool_size is None else pool_size
    if cfg.mode == "competition":
        env = GeneralsEnv(mode="competition", pool_size=size)
        if distance is not None and not exact_competition:
            env.min_generals_distance, env.max_generals_distance = distance
        return env

    kwargs = dict(
        min_grid_size=cfg.min_grid_size,
        max_grid_size=cfg.max_grid_size,
        pad_to=cfg.pad_to,
        min_generals_distance=cfg.min_generals_distance,
        max_generals_distance=cfg.max_generals_distance,
        truncation=cfg.truncation,
        pool_size=size,
        castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
        num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
    )
    if distance is not None:
        kwargs["min_generals_distance"], kwargs["max_generals_distance"] = distance
    return GeneralsEnv(**kwargs)
