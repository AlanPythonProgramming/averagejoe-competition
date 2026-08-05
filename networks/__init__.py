"""Network architectures for generals.io PPO agent."""

import inspect
from functools import partial

from networks import transformer, common
from networks.transformer import HistoryTransformer, greedy_action_transformer
from networks.common import (
    obs_to_array,
    random_action,
    reset_done_envs,
)

NETWORK_REGISTRY = {
    "history_transformer": {
        "cls": HistoryTransformer,
        "init_obs_state": common.init_obs_state,
        "augment_obs": common.augment_obs,
        "reset_obs_state": common.reset_obs_state,
        "greedy_action": transformer.greedy_action_transformer,
    },
}


def get_network_bundle(name: str, cfg=None) -> dict:
    """Look up a network bundle (cls + obs state functions) by name."""
    if name not in NETWORK_REGISTRY:
        available = ", ".join(NETWORK_REGISTRY.keys())
        raise ValueError(f"Unknown network '{name}'. Available: {available}")
    bundle = dict(NETWORK_REGISTRY[name])
    if cfg is not None:
        bundle["init_obs_state"] = partial(
            common.init_obs_state,
            history_size=cfg.history_size,
            temporal_window=cfg.temporal_window,
        )
        bundle["augment_obs"] = partial(
            common.augment_obs,
            competition_features=cfg.competition_features,
        )
    return bundle


# Config fields forwarded to a network constructor when present in its signature.
_NET_CFG_FIELDS = [
    "depth", "embed_dim", "n_head", "ff_factor", "patch_size", "conv_dim",
    "use_bf16", "value_loss", "num_bins", "v_min", "v_max",
    "history_size", "temporal_window", "temporal_hidden", "action_planes",
    "competition_features",
]


def build_network(cfg, key):
    """Instantiate the configured network, forwarding only the config fields
    that match its constructor signature."""
    cls = get_network_bundle(cfg.network)["cls"]
    params = inspect.signature(cls).parameters
    kwargs = {"grid_size": cfg.pad_to, "pad_to": cfg.pad_to, "key": key}
    kwargs.update({f: getattr(cfg, f) for f in _NET_CFG_FIELDS if f in params})
    return cls(**kwargs)
