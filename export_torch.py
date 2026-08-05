"""Export an Equinox checkpoint to the CPU PyTorch submission format."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import torch
from safetensors.torch import save_file

from config import Config
from evals.agent import Agent
from submission.model import TorchCompetitionPolicy


def _tensor(x):
    return torch.from_numpy(np.asarray(jax.device_get(x)).copy())


def export_network(network, cfg, output_dir):
    metadata = {
        "format_version": 1,
        "pad_to": cfg.pad_to,
        "patch_size": cfg.patch_size,
        "action_planes": cfg.action_planes,
        "history_size": cfg.history_size,
        "temporal_window": cfg.temporal_window,
        "temporal_hidden": cfg.temporal_hidden,
        "n_channels": network.n_channels,
        "competition_features": cfg.competition_features,
        "depth": cfg.depth,
        "embed_dim": cfg.embed_dim,
        "n_head": cfg.n_head,
        "ff_factor": cfg.ff_factor,
        "num_bins": network.num_bins,
        "v_min": cfg.v_min,
        "v_max": cfg.v_max,
        "deathtouch_turn": 800,
        "normalization_version": 1,
    }
    torch_model = TorchCompetitionPolicy(metadata)
    state = {
        "value_token": _tensor(network.value_token),
        "pos_encoding": _tensor(network.pos_encoding),
        "temporal_type_embed": _tensor(network.temporal_type_embed),
    }
    if network.num_bins:
        state["bin_centers"] = _tensor(network.bin_centers)

    def linear(prefix, layer):
        state[f"{prefix}.weight"] = _tensor(layer.weight)
        state[f"{prefix}.bias"] = _tensor(layer.bias)

    def norm(prefix, layer):
        state[f"{prefix}.weight"] = _tensor(layer.weight)
        state[f"{prefix}.bias"] = _tensor(layer.bias)

    linear("embedder", network.embedder)
    linear("policy_head", network.policy_head)
    linear("value_head", network.value_head)
    for name in ("army_l1", "army_l2", "land_l1", "land_l2"):
        linear(f"temporal_encoder.{name}", getattr(network.temporal_encoder, name))
    norm("norm_out", network.norm_out)
    for i, block in enumerate(network.transformer_layers):
        base = f"layers.{i}"
        norm(f"{base}.norm1", block.norm1)
        norm(f"{base}.norm2", block.norm2)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{base}.attn.{name}", getattr(block.attn, name))
        linear(f"{base}.ff_linear1", block.ff_linear1)
        linear(f"{base}.ff_linear2", block.ff_linear2)
    torch_model.load_state_dict(state, strict=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file(torch_model.state_dict(), str(output / "weights.safetensors"))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return torch_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="submission")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    agent = Agent.from_config(args.checkpoint, cfg)
    export_network(agent.network, cfg, args.output)
    print(f"Exported {args.checkpoint} to {args.output}")


if __name__ == "__main__":
    main()
