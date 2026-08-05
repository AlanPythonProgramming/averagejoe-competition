"""Stateful protocol observation adapter and Torch policy agent."""

from pathlib import Path

import numpy as np
import torch

from model import TorchCompetitionPolicy

PASS = (1, 0, 0, 0, 0)
PAD = 21


def _max_pool3(x):
    padded = np.pad(x.astype(bool), 1)
    h, w = x.shape
    out = np.zeros((h, w), dtype=bool)
    for di in range(3):
        for dj in range(3):
            out |= padded[di:di + h, dj:dj + w]
    return out


def build_cost_grid(generals, castles, owned):
    structures = (generals | castles) & owned
    padded = np.pad(structures.astype(np.int32), 6)
    h, w = structures.shape
    cost = np.full((h, w), 35, dtype=np.float32)
    for di in range(-6, 7):
        for dj in range(-6, 7):
            surcharge = 14 - 2 * (abs(di) + abs(dj))
            if surcharge > 0:
                cost += surcharge * padded[6 + di:6 + di + h, 6 + dj:6 + dj + w]
    return cost


class Agent:
    def __init__(self, player_id, H, W):
        del player_id
        self.H, self.W = H, W
        self.model = TorchCompetitionPolicy.load(Path(__file__).parent)
        metadata = self.model.metadata
        if metadata["pad_to"] != PAD or not metadata.get("competition_features", True):
            raise ValueError("submission requires the 21x21 competition observation codec")
        self.history_size = metadata["history_size"]
        self.temporal_window = metadata["temporal_window"]
        self.army_stack = np.zeros((self.history_size, PAD, PAD), np.float32)
        self.enemy_stack = np.zeros_like(self.army_stack)
        self.last_army = np.zeros((PAD, PAD), np.float32)
        self.last_enemy = np.zeros((PAD, PAD), np.float32)
        self.castles = np.zeros((PAD, PAD), bool)
        self.generals = np.zeros((PAD, PAD), bool)
        self.mountains = np.zeros((PAD, PAD), bool)
        self.seen = np.zeros((PAD, PAD), bool)
        self.enemy_seen = np.zeros((PAD, PAD), bool)
        self.last_enemy_value = np.zeros((PAD, PAD), np.float32)
        self.last_enemy_age = np.zeros((PAD, PAD), np.float32)
        self.opp_army_history = np.zeros(self.temporal_window, np.float32)
        self.opp_land_history = np.zeros(self.temporal_window, np.float32)
        with torch.inference_mode():
            dummy_obs = torch.zeros((1, metadata["n_channels"], PAD, PAD))
            dummy_temporal = torch.zeros((1, 2, self.temporal_window))
            dummy_mask = torch.zeros((1, 10, PAD, PAD), dtype=torch.bool)
            dummy_mask[0, 9, 0, 0] = True
            self.model(dummy_obs, dummy_temporal, dummy_mask)

    def _encode(self, obs):
        h, w = obs.H, obs.W
        types = np.asarray(obs.type_grid, dtype=np.int32)
        owners = np.asarray(obs.owner_grid, dtype=np.int32)
        armies_small = np.asarray(obs.army_grid, dtype=np.float32)
        pad = lambda x, value=0: np.pad(x, ((0, PAD - h), (0, PAD - w)), constant_values=value)

        armies = pad(armies_small)
        generals_now = pad(types == 4)
        castles_now = pad(types == 3)
        mountains_now = pad(types == 2)
        neutral = pad((owners == 0) & ~np.isin(types, (0, 5)))
        owned = pad(owners == 1)
        opponent = pad(owners == 2)
        fog = pad(types == 0)
        structures_fog = pad(types == 5)
        pad_mask = np.zeros((PAD, PAD), bool)
        pad_mask[h:, :] = True
        pad_mask[:, w:] = True
        structures_fog[pad_mask] = True

        visible = _max_pool3(owned)
        seen_pad_mountains = self.mountains | (pad_mask & visible)
        mountains_now[pad_mask & seen_pad_mountains] = True
        structures_fog[pad_mask & seen_pad_mountains] = False

        own_army = armies * owned
        enemy_army = armies * opponent
        self.army_stack = np.concatenate(((own_army - self.last_army)[None], self.army_stack[:-1]))
        self.enemy_stack = np.concatenate(((enemy_army - self.last_enemy)[None], self.enemy_stack[:-1]))
        self.last_army, self.last_enemy = own_army, enemy_army
        self.seen |= visible
        self.enemy_seen |= _max_pool3(opponent)
        self.castles |= castles_now
        self.generals |= generals_now
        self.mountains |= mountains_now | seen_pad_mountains
        enemy_present = enemy_army > 0
        self.last_enemy_value = np.where(enemy_present, enemy_army, self.last_enemy_value)
        self.last_enemy_age = np.where(enemy_present, 0.0, self.last_enemy_age + 1.0)
        self.opp_army_history = np.roll(self.opp_army_history, -1)
        self.opp_land_history = np.roll(self.opp_land_history, -1)
        self.opp_army_history[-1] = obs.opp_army
        self.opp_land_history[-1] = obs.opp_land

        yy, xx = np.mgrid[:PAD, :PAD]
        ones = np.ones((PAD, PAD), np.float32)
        cost = build_cost_grid(generals_now, castles_now, owned)
        channels = np.stack([
            armies, own_army, enemy_army, armies * neutral,
            self.seen, self.enemy_seen, self.generals, self.castles, self.mountains,
            neutral, owned, opponent, fog, structures_fog,
            obs.turn * ones, (obs.turn % 50) / 50.0 * ones,
            obs.my_land * ones, obs.my_army * ones, obs.opp_land * ones, obs.opp_army * ones,
            self.last_enemy_value, np.log1p(self.last_enemy_age) / 5.0,
            xx / (PAD - 1), yy / (PAD - 1), cost,
            (obs.turn >= 800) * ones,
        ], axis=0).astype(np.float32)
        augmented = np.concatenate((channels, self.army_stack, self.enemy_stack), axis=0)
        augmented[[0, 1, 2, 3, 17, 19, 20] + list(range(26, augmented.shape[0]))] /= 50.0
        augmented[14] /= 50.0
        augmented[[16, 18]] /= 50.0
        augmented[24] /= 100.0

        legal = np.zeros((10, PAD, PAD), bool)
        blocked = pad(np.isin(types, (2, 5)), True)
        for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if owned[r, c] and armies[r, c] > 1 and 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                        legal[d, r, c] = legal[d + 4, r, c] = True
        legal[8] = owned & ~generals_now & ~castles_now & (armies >= cost)
        legal[9, 0, 0] = True
        temporal = np.stack((self.opp_army_history, self.opp_land_history), axis=0)
        return augmented, temporal, legal

    def act(self, obs):
        augmented, temporal, legal = self._encode(obs)
        with torch.inference_mode():
            logits, _ = self.model(
                torch.from_numpy(augmented[None]),
                torch.from_numpy(temporal[None]),
                torch.from_numpy(legal[None]),
            )
        idx = int(torch.argmax(logits[0]))
        plane, pos = divmod(idx, PAD * PAD)
        row, col = divmod(pos, PAD)
        if plane < 4:
            return 0, row, col, plane, 0
        if plane < 8:
            return 0, row, col, plane - 4, 1
        if plane == 8:
            return 2, row, col, 0, 0
        return PASS
