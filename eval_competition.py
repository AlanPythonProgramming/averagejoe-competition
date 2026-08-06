"""Fixed-seed, both-seat competition strength gate against a JAX bot."""

import argparse
import sys

import jax
import jax.numpy as jnp
import jax.random as jrandom

from config import Config
from evals.agent import Agent
from train.environment import make_env
from train.evaluations import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/competition_cpu.yaml")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument(
        "--opponent", choices=("hunter", "expander", "random"),
        default="hunter",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--min-decisive-win-rate", type=float, default=0.70)
    parser.add_argument("--min-decisive-fraction", type=float, default=0.50)
    args = parser.parse_args()
    if args.games % 2:
        parser.error("--games must be even so every map is played from both seats")

    cfg = Config.from_yaml(args.config)
    agent = Agent.from_config(args.checkpoint, cfg)
    n_maps = args.games // 2
    pool_size = max(16, ((n_maps + 15) // 16) * 16)
    env = make_env(cfg, pool_size=pool_size, exact_competition=True)
    key = jrandom.PRNGKey(args.seed)
    key, pool_key, eval_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    single = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    obs_state = jax.tree.map(
        lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), single)
    wins, losses, draws, _, wb, lb, split, _ = evaluate(
        env, agent.network, eval_key, env.truncation, n_maps, cfg.pad_to,
        obs_state, agent.augment_fn, agent.bundle["reset_obs_state"],
        agent.greedy_fn, pool=pool, opponent_kind=args.opponent)
    wins, losses, draws = int(wins), int(losses), int(draws)
    decisive = wins + losses
    decisive_wr = wins / max(decisive, 1)
    decisive_fraction = decisive / args.games
    print(
        f"{args.opponent.title()}: {wins}W/{losses}L/{draws}D; "
        f"decisive WR={decisive_wr:.1%}; decisive={decisive_fraction:.1%}"
    )
    print(f"Paired maps: {int(wb)} won-both / {int(lb)} lost-both / {int(split)} split")
    passed = decisive_wr >= args.min_decisive_win_rate and decisive_fraction >= args.min_decisive_fraction
    print("STRENGTH GATE: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
