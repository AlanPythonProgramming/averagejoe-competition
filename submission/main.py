"""Competition stdin/stdout runner."""

import sys
from dataclasses import dataclass

import torch

from agent import Agent, PASS


@dataclass
class Observation:
    H: int
    W: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: list
    owner_grid: list
    army_grid: list


def _grid(stream, h):
    return [[int(x) for x in stream.readline().split()] for _ in range(h)]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    line = sys.stdin.readline()
    if not line:
        return
    player_id, h, w = map(int, line.split())
    try:
        agent = Agent(player_id, h, w)
    except Exception as exc:
        print(f"model initialization failed: {exc}", file=sys.stderr, flush=True)
        agent = None
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        turn, ml, ma, ol, oa = map(int, line.split())
        obs = Observation(h, w, turn, ml, ma, ol, oa,
                          _grid(sys.stdin, h), _grid(sys.stdin, h), _grid(sys.stdin, h))
        try:
            action = agent.act(obs) if agent is not None else PASS
        except Exception as exc:
            print(f"inference failed on turn {turn}: {exc}", file=sys.stderr, flush=True)
            action = PASS
        sys.stdout.write("%d %d %d %d %d\n" % action)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
