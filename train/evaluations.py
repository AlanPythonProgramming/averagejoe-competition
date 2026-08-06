"""Evaluation functions: play network vs random or vs opponent."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import NamedTuple

from generals.core.game import get_observation

from networks import (
    obs_to_array,
    random_action,
    reset_done_envs,
)
from networks.common import compute_action_mask
from evals.agent import Agent
from evals.ref_eval import ref_eval
from generals.core.action import DIRECTIONS, compute_valid_move_mask
from generals.agents import HunterAgent


_COMPETITION_HUNTER = HunterAgent(id="CompetitionHunter")


def competition_hunter_action(obs, key):
    """Use the pinned engine's aggressive Hunter policy for exact evaluation."""
    return _COMPETITION_HUNTER.act(obs, key)


def competition_expander_action(obs, key):
    """JAX-exact equivalent of competition/agents/expander_python."""
    del key
    h, w = obs.armies.shape
    valid = compute_valid_move_mask(
        obs.armies, obs.owned_cells, obs.mountains | obs.structures_in_fog)
    positions = jnp.argwhere(valid, size=h * w * 4, fill_value=-1)
    position_valid = jnp.all(positions >= 0, axis=-1)

    def score(position):
        r, c, direction = position
        r = jnp.clip(r, 0, h - 1)
        c = jnp.clip(c, 0, w - 1)
        nr = jnp.clip(r + DIRECTIONS[direction, 0], 0, h - 1)
        nc = jnp.clip(c + DIRECTIONS[direction, 1], 0, w - 1)
        can_capture = obs.armies[r, c] > obs.armies[nr, nc] + 1
        is_opp = obs.opponent_cells[nr, nc]
        is_visible_neutral = obs.neutral_cells[nr, nc] & ~obs.fog_cells[nr, nc]
        is_expansion = is_opp | is_visible_neutral
        value = obs.armies[r, c].astype(jnp.float32)
        value = jnp.where(is_expansion, value * 10.0, value)
        value = jnp.where(is_opp, value * 2.0, value)
        return jnp.where(can_capture, value, -1.0)

    scores = jax.vmap(score)(positions)
    best_idx = jnp.argmax(jnp.where(position_valid, scores, -1.0))
    first_idx = jnp.argmax(position_valid)
    has_capture = jnp.any(position_valid & (scores >= 0))
    has_valid = jnp.any(position_valid)
    choice = positions[jnp.where(has_capture, best_idx, first_idx)]
    return jnp.array([
        jnp.where(has_valid, 0, 1),
        jnp.where(has_valid, choice[0], 0),
        jnp.where(has_valid, choice[1], 0),
        jnp.where(has_valid, choice[2], 0),
        0,
    ], dtype=jnp.int32)


def opponent_action(kind, obs, key, action_planes, build_enabled):
    """Dispatch one JAX-native fixed opponent action."""
    if kind == "hunter":
        return competition_hunter_action(obs, key)
    if kind == "expander":
        return competition_expander_action(obs, key)
    if kind == "random":
        return random_action(key, obs, action_planes, build_enabled)
    raise ValueError(f"unknown evaluation opponent: {kind}")


def result_metrics(wins, losses, draws):
    """Return match score, decisive win rate, and decisive fraction.

    Match score awards 1 for a win, 0.5 for a draw, and 0 for a loss. It is a
    useful diagnostic alongside the raw win rate used by the original
    AverageJoe curriculum.
    """
    total = wins + losses + draws
    decisive = wins + losses
    score = (wins + 0.5 * draws) / max(total, 1)
    decisive_wr = wins / max(decisive, 1)
    decisive_fraction = decisive / max(total, 1)
    return score, decisive_wr, decisive_fraction


@jax.jit(static_argnames=["env", "truncation", "n_maps", "grid_size", "augment_fn", "reset_fn", "greedy_fn", "opponent_kind"])
def evaluate(env, network, key, truncation, n_maps, grid_size,
             obs_state, augment_fn, reset_fn, greedy_fn, pool=None,
             opponent_kind="random"):
    """Play n_maps maps as both player positions vs random.

    Each map is played twice: once with network as p0, once as p1.
    Returns (wins, losses, draws, finished, won_both, lost_both, split, key).
    """
    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step
    key, key_fwd, key_rev = jrandom.split(key, 3)
    key, *init_keys = jrandom.split(key, n_maps + 1)
    init_keys_arr = jnp.stack(init_keys)

    # ---- Forward: network=p0, random=p1 ----
    states_fwd = (jax.tree.map(lambda x: x[:n_maps], pool)
                  if pool is not None else jax.vmap(env.init_state)(init_keys_arr))

    def scan_fwd(carry, _):
        states, rng, finished, net_won, net_lost, drew, obs_st = carry
        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        obs_arr = jax.vmap(obs_to_array)(obs_p0)
        masks = jax.vmap(lambda o: compute_action_mask(o, network.action_planes, env.build_castles))(obs_p0)
        obs_aug, obs_st = jax.vmap(augment_fn)(obs_arr, obs_st)
        temporal = jnp.stack([obs_st.opponent_army_history, obs_st.opponent_land_history], axis=1)
        a0 = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0))(network, obs_aug, masks, temporal)

        rng, *ks = jrandom.split(rng, n_maps + 1)
        a1 = jax.vmap(
            lambda o, k: opponent_action(
                opponent_kind, o, k, network.action_planes, env.build_castles)
        )(obs_p1, jnp.stack(ks))

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(step_fn)(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == 0))
        net_lost = net_lost | (new_done & (timesteps.info.winner == 1))
        drew = drew | (new_done & (timesteps.info.winner < 0))
        finished = finished | dones
        obs_st = reset_done_envs(obs_st, dones)

        return (new_states, rng, finished, net_won, net_lost, drew, obs_st), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, _, fwd_fin, fwd_won, fwd_lost, fwd_drew, _), _ = jax.lax.scan(
        scan_fwd, (states_fwd, key_fwd, z, z, z, z, obs_state), None, length=truncation)

    # ---- Reverse: random=p0, network=p1 ----
    states_rev = (jax.tree.map(lambda x: x[:n_maps], pool)
                  if pool is not None else jax.vmap(env.init_state)(init_keys_arr))

    def scan_rev(carry, _):
        states, rng, finished, net_won, net_lost, drew, obs_st = carry
        obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

        obs_arr = jax.vmap(obs_to_array)(obs_p1)
        masks = jax.vmap(lambda o: compute_action_mask(o, network.action_planes, env.build_castles))(obs_p1)
        obs_aug, obs_st = jax.vmap(augment_fn)(obs_arr, obs_st)
        temporal = jnp.stack([obs_st.opponent_army_history, obs_st.opponent_land_history], axis=1)
        a1 = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0))(network, obs_aug, masks, temporal)

        rng, *ks = jrandom.split(rng, n_maps + 1)
        a0 = jax.vmap(
            lambda o, k: opponent_action(
                opponent_kind, o, k, network.action_planes, env.build_castles)
        )(obs_p0, jnp.stack(ks))

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(step_fn)(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == 1))
        net_lost = net_lost | (new_done & (timesteps.info.winner == 0))
        drew = drew | (new_done & (timesteps.info.winner < 0))
        finished = finished | dones
        obs_st = reset_done_envs(obs_st, dones)

        return (new_states, rng, finished, net_won, net_lost, drew, obs_st), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, _, rev_fin, rev_won, rev_lost, rev_drew, _), _ = jax.lax.scan(
        scan_rev, (states_rev, key_rev, z, z, z, z, obs_state), None, length=truncation)

    # ---- Aggregate ----
    total_wins = jnp.sum(fwd_won) + jnp.sum(rev_won)
    total_losses = jnp.sum(fwd_lost) + jnp.sum(rev_lost)
    total_draws = jnp.sum(fwd_drew) + jnp.sum(rev_drew)
    total_finished = jnp.sum(fwd_fin) + jnp.sum(rev_fin)

    # Paired map stats (maps where both games had decisive outcomes)
    both_decisive = (fwd_won | fwd_lost) & (rev_won | rev_lost)
    won_both = jnp.sum(both_decisive & fwd_won & rev_won)
    lost_both = jnp.sum(both_decisive & fwd_lost & rev_lost)
    split = jnp.sum(both_decisive & ((fwd_won & rev_lost) | (fwd_lost & rev_won)))

    return total_wins, total_losses, total_draws, total_finished, won_both, lost_both, split, key


class EvalCtx(NamedTuple):
    """Loop-invariant context for periodic_eval (network plumbing + reference setup)."""
    bundle: dict
    single_state: object
    augment_fn: object
    reset_fn: object
    greedy_fn: object
    ref_agents: object
    ref_h2h: object
    ref_eval_env: object
    ref_eval_pool: object


def periodic_eval(it, cfg, eval_freq, network, ema_params, static,
                  eval_env, eval_pool, ev, logger, key, last_eval_wr):
    """Eval vs random (+ reference ELO) on eval iters; logs results.

    Returns (eval_ran, last_eval_wr, key). ``last_eval_wr`` is raw wins divided
    by all completed games, matching the original AverageJoe curriculum.
    """
    eval_ran = False
    if eval_freq > 0 and (it == 0 or (it + 1) % eval_freq == 0):
        key, eval_key = jrandom.split(key)
        n_maps = cfg.eval_games // 2
        eval_obs_state = jax.tree.map(
            lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), ev.single_state)
        ew, el, ed, edone, wb, lb, sp, _ = evaluate(
            eval_env, network, eval_key, cfg.truncation, n_maps, cfg.pad_to,
            eval_obs_state, ev.augment_fn, ev.reset_fn, ev.greedy_fn, pool=eval_pool)
        ew, el, ed, edone = int(ew), int(el), int(ed), int(edone)
        wb, lb, sp = int(wb), int(lb), int(sp)
        match_score, decisive_wr, decisive_fraction = result_metrics(ew, el, ed)
        last_eval_wr = ew / max(edone, 1)
        eval_ran = True
        print(
            f"  EVAL: {ew}W/{el}L/{ed}D vs random | "
            f"WR={last_eval_wr:.0%}, score={match_score:.0%}, "
            f"decisive WR={decisive_wr:.0%}, "
            f"decisive={decisive_fraction:.0%} | "
            f"Maps({n_maps}): {wb}WB/{lb}LB/{sp}S"
        )
        logger.log_eval(it, ew, el, ed, edone)
        logger.log(it, {
            "eval/match_score": match_score,
            "eval/decisive_win_rate": decisive_wr,
            "eval/decisive_fraction": decisive_fraction,
            "eval/won_both": wb / max(n_maps, 1),
            "eval/lost_both": lb / max(n_maps, 1),
            "eval/split": sp / max(n_maps, 1),
        })

    strength_freq = cfg.strength_eval_every
    strength_opponents = cfg.strength_eval_opponents or (
        [cfg.strength_eval_opponent] if cfg.strength_eval_opponent else [])
    if (strength_opponents and strength_freq > 0
            and (it == 0 or (it + 1) % strength_freq == 0)):
        for opponent_name in strength_opponents:
            opponent = opponent_name.lower()
            if opponent not in ("hunter", "expander", "random"):
                raise ValueError(
                    "strength opponents must be hunter, expander, or random")
            key, strength_key = jrandom.split(key)
            n_maps = cfg.strength_eval_games // 2
            strength_obs_state = jax.tree.map(
                lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), ev.single_state)
            strength_network = (
                eqx.combine(ema_params, static)
                if cfg.strength_eval_use_ema else network
            )
            sw, sl, sd, sdone, swb, slb, ssp, _ = evaluate(
                eval_env, strength_network, strength_key, cfg.truncation, n_maps,
                cfg.pad_to, strength_obs_state, ev.augment_fn, ev.reset_fn,
                ev.greedy_fn, pool=eval_pool, opponent_kind=opponent)
            sw, sl, sd, sdone = int(sw), int(sl), int(sd), int(sdone)
            swb, slb, ssp = int(swb), int(slb), int(ssp)
            match_score, decisive_wr, decisive_fraction = result_metrics(sw, sl, sd)
            weights = "EMA" if cfg.strength_eval_use_ema else "current"
            print(
                f"  STRENGTH: {sw}W/{sl}L/{sd}D vs {opponent} ({weights}) | "
                f"score={match_score:.0%}, decisive WR={decisive_wr:.0%}, "
                f"decisive={decisive_fraction:.0%} | "
                f"Maps({n_maps}): {swb}WB/{slb}LB/{ssp}S"
            )
            logger.log(it, {
                f"strength/{opponent}_wins": sw,
                f"strength/{opponent}_losses": sl,
                f"strength/{opponent}_draws": sd,
                f"strength/{opponent}_match_score": match_score,
                f"strength/{opponent}_decisive_wr": decisive_wr,
                f"strength/{opponent}_decisive_fraction": decisive_fraction,
                f"strength/{opponent}_won_both": swb / max(n_maps, 1),
                f"strength/{opponent}_lost_both": slb / max(n_maps, 1),
                f"strength/{opponent}_split": ssp / max(n_maps, 1),
            })

    if ev.ref_agents and (it == 0 or (it + 1) % cfg.ref_eval_every == 0):
        ema_network = eqx.combine(ema_params, static)
        ema_agent = Agent(ema_network, cfg, ev.bundle, name="_ema")
        if cfg.eval_ema_only:
            candidates = [ema_agent]
        else:
            candidates = [Agent(network, cfg, ev.bundle, name="_current"), ema_agent]
        key, ref_key = jrandom.split(key)
        ratings, full_h2h = ref_eval(
            candidates=candidates,
            ref_agents=ev.ref_agents, ref_h2h=ev.ref_h2h,
            env=ev.ref_eval_env, pool=ev.ref_eval_pool,
            num_games=cfg.ref_eval_games,
            truncation=ev.ref_eval_env.truncation,
            key=ref_key,
        )
        ref_names = [a.name for a in ev.ref_agents]
        max_ref_elo = max(ratings[n] for n in ref_names)
        if cfg.eval_ema_only:
            print(f"  REF_EVAL ELO (iter {it+1}): ema={ratings['_ema']:.0f}")
        else:
            print(f"  REF_EVAL ELO (iter {it+1}): current={ratings['_current']:.0f} ema={ratings['_ema']:.0f}")
        ref_metrics = {
            "ref_elo/ema": ratings["_ema"],
            "ref_elo/ema_vs_max": ratings["_ema"] - max_ref_elo,
        }
        if not cfg.eval_ema_only:
            ref_metrics["ref_elo/current"] = ratings["_current"]
            ref_metrics["ref_elo/current_vs_max"] = ratings["_current"] - max_ref_elo
        for rn in ref_names:
            for cand_tag in [c.name for c in candidates]:
                r = full_h2h.get(cand_tag, {}).get(rn, {})
                w, l, d = r.get("wins", 0), r.get("losses", 0), r.get("draws", 0)
                tot = w + l + d
                wr = w / tot if tot > 0 else 0.0
                tag = "current" if cand_tag == "_current" else "ema"
                ref_metrics[f"ref_wr/{tag}_vs_{rn}"] = wr
        logger.log(it, ref_metrics)

    return eval_ran, last_eval_wr, key
