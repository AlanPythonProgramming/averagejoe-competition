import jax
import jax.numpy as jnp
import jax.random as jrandom

from config import Config
from networks import build_network, get_network_bundle
from networks.common import (
    compute_action_mask,
    compute_build_cost_grid_obs,
    decode_action,
    encode_action,
    obs_to_array,
    reset_done_envs,
)
from train.environment import make_env
from train.ppo import compute_gae
from train.evaluations import result_metrics
from generals.core.game import create_initial_state, get_observation
from generals.modifiers.build_castles import build_cost_grid


def _state():
    grid = jnp.zeros((8, 8), jnp.int32).at[0, 0].set(1).at[7, 7].set(2)
    state = create_initial_state(grid)
    own = state.ownership.at[0, 3, 3].set(True)
    neutral = state.ownership_neutral.at[3, 3].set(False)
    return state._replace(ownership=own, ownership_neutral=neutral,
                          armies=state.armies.at[3, 3].set(80))


def test_action_round_trip_and_canonical_pass():
    actions = [
        jnp.array([0, 3, 4, 2, 0]),
        jnp.array([0, 5, 6, 1, 1]),
        jnp.array([2, 2, 7, 0, 0]),
        jnp.array([1, 0, 0, 0, 0]),
    ]
    for action in actions:
        got = decode_action(encode_action(action, 21, 10), 21, 10)
        assert jnp.array_equal(got, action)


def test_build_cost_and_mask_match_engine():
    state = _state()
    obs = get_observation(state, 0)
    assert jnp.array_equal(compute_build_cost_grid_obs(obs), build_cost_grid(state, 0))
    mask = compute_action_mask(obs, 10, True)
    assert int(mask[9].sum()) == 1 and bool(mask[9, 0, 0])
    assert bool(mask[8, 3, 3])
    assert not bool(mask[8, 0, 0])


def test_competition_factory_preserves_rules_during_curriculum():
    cfg = Config.from_yaml("configs/competition_cpu.yaml")
    env = make_env(cfg, pool_size=16, distance=(2, 6))
    assert env.mode == "competition"
    assert env.build_castles and env.deathtouch_turn == 800
    assert env.truncation == 1200 and env.pad_to == 21
    assert (env.min_generals_distance, env.max_generals_distance) == (2, 6)


def test_observation_shape_and_memory_reset():
    cfg = Config.from_yaml("configs/competition_cpu.yaml")
    bundle = get_network_bundle(cfg.network, cfg)
    state = bundle["init_obs_state"](21, 21)
    obs = get_observation(_state(), 0)
    augmented, new_state = bundle["augment_obs"](obs_to_array(obs), state)
    assert augmented.shape == (34, 21, 21)
    batched = jax.tree.map(lambda x: x[None], new_state)
    reset = reset_done_envs(batched, jnp.array([True]))
    assert all(bool(jnp.all(x == 0)) for x in jax.tree.leaves(reset))


def test_standard_config_retains_averagejoe_channel_shape():
    cfg = Config(history_size=7, competition_features=False)
    bundle = get_network_bundle(cfg.network, cfg)
    memory = bundle["init_obs_state"](15, 15)
    augmented, _ = bundle["augment_obs"](obs_to_array(get_observation(_state(), 0)), memory)
    network = build_network(cfg, jrandom.PRNGKey(3))
    assert augmented.shape == (38, 15, 15)
    assert network.n_channels == 38


def test_gae_and_network_forward_are_finite():
    rewards = jnp.array([[0.0, 1.0], [1.0, -1.0]])
    values = jnp.zeros_like(rewards)
    adv = compute_gae(rewards, values, values, jnp.array([[False, False], [True, True]]),
                      jnp.zeros_like(rewards, dtype=bool), 1.0, 0.9)
    assert bool(jnp.all(jnp.isfinite(adv)))

    cfg = Config.from_yaml("configs/competition_cpu.yaml")
    network = build_network(cfg, jrandom.PRNGKey(0))
    obs = jnp.zeros((34, 21, 21))
    mask = jnp.zeros((10, 21, 21), dtype=bool).at[9, 0, 0].set(True)
    temporal = jnp.zeros((2, 64))
    logits, value, _ = network._forward(obs, mask, temporal)
    assert logits.shape == (4410,)
    assert bool(jnp.all(jnp.isfinite(logits))) and bool(jnp.isfinite(value))


def test_evaluation_metrics_report_draws_separately():
    score, decisive_wr, decisive_fraction = result_metrics(85, 1, 170)
    assert abs(score - 0.6640625) < 1e-9
    assert decisive_wr > 0.98
    assert abs(decisive_fraction - 86 / 256) < 1e-9
    assert result_metrics(0, 0, 256)[0] == 0.5
