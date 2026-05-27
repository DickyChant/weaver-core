"""Example weaver network-config for generative MeanFlow training.

Drop into your project as `--network-config networks/example_MeanFlow.py` and
feed weaver a data-config that exposes `pf_features` (the data to generate)
and at least one conditioning group (e.g. `jet_kinematics`). Labels are not
required by the loss but weaver still wants them in the yaml; you can use a
dummy `labels: { type: simple, value: [is_signal] }`.

A minimal pf_features input could be the constituent (E, px, py, pz, ...)
log-normalised; conditioning is e.g. the per-jet (pt, eta, mass).
"""

import math

import torch
import torch.nn as nn

from weaver.nn.loss.meanflow import MeanFlowLoss, MeanFlowSEALLoss
from weaver.utils.nn.tools_generative import (
    make_train_generative,
    make_evaluate_generative,
)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class MeanFlowParT(nn.Module):
    """Toy MeanFlow-conditioned set transformer over particle constituents.

    forward signature: model(z, t, r, cond) -> velocity (same shape as z).

    `z` has shape (B, F, N) — same layout as weaver's pf_features input (channels x particles).
    `cond` is whichever extra input groups are passed (each shaped (B, Cg, M); reduced via mean).
    """

    def __init__(self, num_features, cond_dims=(), embed_dim=128, num_heads=4, num_layers=4, time_dim=128):
        super().__init__()
        self.num_features = num_features
        self.time_dim = time_dim
        self.feat_in = nn.Linear(num_features, embed_dim)
        self.t_embed = SinusoidalTimeEmbed(time_dim)
        self.r_embed = SinusoidalTimeEmbed(time_dim)
        cond_in = sum(cond_dims)
        self.tr_mlp = nn.Sequential(
            nn.Linear(2 * time_dim + cond_in, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        enc = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4 * embed_dim,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_features)

    def forward(self, z, t, r, *cond):
        # z: (B, F, N) -> (B, N, F)
        x = z.transpose(1, 2)
        h = self.feat_in(x)
        tr = torch.cat([self.t_embed(t), self.r_embed(r)], dim=-1)
        if cond:
            cond_flat = torch.cat([c.mean(dim=-1) if c.ndim == 3 else c for c in cond], dim=-1)
            tr = torch.cat([tr, cond_flat], dim=-1)
        tr = self.tr_mlp(tr).unsqueeze(1)  # (B, 1, embed)
        h = h + tr  # FiLM-style additive conditioning
        h = self.blocks(h)
        v = self.head(h).transpose(1, 2)  # back to (B, F, N)
        return v


def _shape_dims(data_config, key):
    """Return the feature dim (channels) of a weaver input group."""
    # input_shapes maps to (-1, channels, length)
    return data_config.input_shapes[key][1]


def get_model(data_config, **kwargs):
    primary = kwargs.get("data_key", data_config.input_names[0])
    cond_keys = kwargs.get("cond_keys", list(data_config.input_names[1:]))
    cond_dims = tuple(_shape_dims(data_config, k) for k in cond_keys)
    num_features = _shape_dims(data_config, primary)
    model = MeanFlowParT(
        num_features=num_features,
        cond_dims=cond_dims,
        embed_dim=kwargs.get("embed_dim", 128),
        num_heads=kwargs.get("num_heads", 4),
        num_layers=kwargs.get("num_layers", 4),
        time_dim=kwargs.get("time_dim", 128),
    )
    model_info = {
        "input_names": (primary, *cond_keys),
        "input_shapes": {k: data_config.input_shapes[k] for k in (primary, *cond_keys)},
        "output_names": ["velocity"],
        "dynamic_axes": None,
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    """If `seal_lambda` is set (and > 0), use MeanFlow + deltaSEAL on the
    1-NFE generation (kinematics carry the Lorentz / SO(3) rep, detector
    features are declared invariant). Otherwise plain MeanFlow.
    """
    if kwargs.get("seal_lambda", 0.0):
        return MeanFlowSEALLoss(
            flow_ratio=kwargs.get("flow_ratio", 0.75),
            time_dist=tuple(kwargs.get("time_dist", ("lognorm", -0.4, 1.0))),
            jvp_api=kwargs.get("jvp_api", "func"),
            seal_lambda=float(kwargs["seal_lambda"]),
            kin_start=int(kwargs.get("kin_start", 0)),
            kin_dim=int(kwargs.get("kin_dim", 4)),
            group=kwargs.get("seal_group", "lorentz"),
            num_gens_per_step=int(kwargs.get("seal_num_gens_per_step", -1)),
            normalize_per_generator=bool(kwargs.get("seal_normalize", True)),
        )
    return MeanFlowLoss(
        flow_ratio=kwargs.get("flow_ratio", 0.75),
        time_dist=tuple(kwargs.get("time_dist", ("lognorm", -0.4, 1.0))),
        jvp_api=kwargs.get("jvp_api", "func"),
    )


def get_train_fn(data_config, **kwargs):
    return make_train_generative(
        input_key=kwargs.get("data_key", data_config.input_names[0]),
        cond_keys=kwargs.get("cond_keys", list(data_config.input_names[1:])),
    )


def get_evaluate_fn(data_config, **kwargs):
    return make_evaluate_generative(
        input_key=kwargs.get("data_key", data_config.input_names[0]),
        cond_keys=kwargs.get("cond_keys", list(data_config.input_names[1:])),
    )
