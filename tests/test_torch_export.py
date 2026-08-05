import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import torch
from dataclasses import replace

from config import Config
from export_torch import export_network
from networks import build_network
from networks.common import normalize_observations


def test_torch_export_matches_equinox(tmp_path):
    cfg = Config.from_yaml("configs/competition_cpu.yaml")
    network = build_network(cfg, jrandom.PRNGKey(7))
    model = export_network(network, cfg, tmp_path)
    obs = jrandom.normal(jrandom.PRNGKey(8), (34, 21, 21))
    temporal = jrandom.normal(jrandom.PRNGKey(9), (2, 64))
    mask = jnp.ones((10, 21, 21), dtype=bool).at[9].set(False).at[9, 0, 0].set(True)
    jax_logits, jax_value, _ = network._forward(obs, mask, temporal)
    with torch.inference_mode():
        torch_logits, torch_value = model(
            torch.from_numpy(np.asarray(normalize_observations(obs)).copy()[None]),
            torch.from_numpy(np.asarray(temporal).copy()[None]),
            torch.from_numpy(np.asarray(mask).copy()[None]),
        )
    np.testing.assert_allclose(np.asarray(jax_logits), torch_logits[0].numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(jax_value), torch_value[0].numpy(), atol=1e-4, rtol=1e-4)
    assert int(jnp.argmax(jax_logits)) == int(torch.argmax(torch_logits[0]))


def test_categorical_value_export(tmp_path):
    cfg = replace(
        Config.from_yaml("configs/competition_cpu.yaml"),
        depth=1,
        embed_dim=24,
        n_head=4,
        history_size=1,
        temporal_window=8,
        temporal_hidden=16,
        value_loss="ce",
        num_bins=7,
    )
    network = build_network(cfg, jrandom.PRNGKey(10))
    model = export_network(network, cfg, tmp_path)
    obs = jrandom.normal(jrandom.PRNGKey(11), (28, 21, 21))
    temporal = jrandom.normal(jrandom.PRNGKey(12), (2, 8))
    mask = jnp.zeros((10, 21, 21), dtype=bool).at[9, 0, 0].set(True)
    jax_logits, jax_value, _ = network._forward(obs, mask, temporal)
    with torch.inference_mode():
        torch_logits, torch_value = model(
            torch.from_numpy(np.asarray(normalize_observations(obs)).copy()[None]),
            torch.from_numpy(np.asarray(temporal).copy()[None]),
            torch.from_numpy(np.asarray(mask).copy()[None]),
        )
    np.testing.assert_allclose(np.asarray(jax_logits), torch_logits[0].numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(jax_value), torch_value[0].numpy(), atol=1e-4, rtol=1e-4)
