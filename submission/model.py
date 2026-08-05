"""CPU PyTorch mirror of the Equinox competition transformer."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


class Attention(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.head_dim**0.5, dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(b, n, d))


class Block(torch.nn.Module):
    def __init__(self, dim, heads, ff_factor):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ff_linear1 = torch.nn.Linear(dim, ff_factor * dim)
        self.ff_linear2 = torch.nn.Linear(ff_factor * dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff_linear2(torch.nn.functional.silu(self.ff_linear1(self.norm2(x))))


class TemporalEncoder(torch.nn.Module):
    def __init__(self, window, hidden, dim):
        super().__init__()
        self.army_l1 = torch.nn.Linear(window, hidden)
        self.army_l2 = torch.nn.Linear(hidden, dim)
        self.land_l1 = torch.nn.Linear(window, hidden)
        self.land_l2 = torch.nn.Linear(hidden, dim)

    def forward(self, temporal):
        army = self.army_l2(torch.nn.functional.silu(self.army_l1(temporal[:, 0] / 50.0)))
        land = self.land_l2(torch.nn.functional.silu(self.land_l1(temporal[:, 1] / 50.0)))
        return torch.stack((army, land), dim=1)


class TorchCompetitionPolicy(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        self.pad_to = metadata["pad_to"]
        self.patch_size = metadata["patch_size"]
        self.action_planes = metadata["action_planes"]
        dim = metadata["embed_dim"]
        patch_dim = metadata["n_channels"] * self.patch_size**2
        self.embedder = torch.nn.Linear(patch_dim, dim)
        tokens = (self.pad_to // self.patch_size) ** 2 + 3
        self.value_token = torch.nn.Parameter(torch.empty(1, dim))
        self.pos_encoding = torch.nn.Parameter(torch.empty(tokens, dim))
        self.layers = torch.nn.ModuleList([
            Block(dim, metadata["n_head"], metadata["ff_factor"])
            for _ in range(metadata["depth"])
        ])
        self.norm_out = torch.nn.LayerNorm(dim)
        self.policy_head = torch.nn.Linear(
            dim, self.action_planes * self.patch_size**2)
        self.num_bins = metadata.get("num_bins", 0)
        self.value_head = torch.nn.Linear(dim, self.num_bins or 1)
        if self.num_bins:
            self.register_buffer(
                "bin_centers",
                torch.linspace(metadata["v_min"], metadata["v_max"], self.num_bins),
            )
        self.temporal_encoder = TemporalEncoder(
            metadata["temporal_window"], metadata["temporal_hidden"], dim)
        self.temporal_type_embed = torch.nn.Parameter(torch.empty(2, dim))

    def forward(self, obs, temporal, legal_mask=None):
        b, c, p, _ = obs.shape
        m = self.patch_size
        gp = p // m
        patches = obs.reshape(b, c, gp, m, gp, m)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, gp * gp, -1)
        spatial = self.embedder(patches)
        temporal_tokens = self.temporal_encoder(temporal) + self.temporal_type_embed.unsqueeze(0)
        value_token = self.value_token.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat((value_token, temporal_tokens, spatial), dim=1) + self.pos_encoding.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        value_raw = self.value_head(x[:, 0])
        if self.num_bins:
            value = (torch.softmax(value_raw, dim=-1) * self.bin_centers).sum(dim=-1)
        else:
            value = value_raw.squeeze(-1)
        logits = self.policy_head(x[:, 3:]).reshape(b, gp, gp, self.action_planes, m, m)
        logits = logits.permute(0, 3, 1, 4, 2, 5).reshape(b, self.action_planes, p, p)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        return logits.flatten(1), value

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        model = cls(metadata)
        model.load_state_dict(load_file(str(directory / "weights.safetensors")))
        return model.eval()
