"""ParT as a MeanFlow-compatible generative backbone.

Wraps weaver.nn.model.ParticleTransformer.ParticleTransformer in segmentation
mode (per-particle output) and adds the (t, r, *cond) conditioning that the
MeanFlow loss expects.

Forward signature: model(z, t, r, *cond) -> v  with `v` shape == `z` shape
(B, num_features, num_particles).

Pair features are computed from z's kinematic block (slice cols
[kin_start..kin_start+4]). These are analytic Lorentz invariants of the
noisy state, so the attention pattern is by-construction Lorentz-equivariant,
but the per-particle output is not — SEAL trains the latter.

Conditioning is concatenated to each particle's feature vector (FiLM-style
modulation could replace this; concat is the simplest thing that works).
"""

import math

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer


def _replace_batchnorm_with_identity(module: nn.Module) -> None:
    """Walk a module tree and swap every nn.BatchNorm1d for nn.Identity.

    Why: torch.func.jvp rejects modules that mutate captured tensors in
    place, and BatchNorm updates `num_batches_tracked` even in eval mode.
    For weaver-preprocessed inputs the BN was already a near-identity, so
    swapping it out is essentially free.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.Identity())
        else:
            _replace_batchnorm_with_identity(child)


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int, max_freq: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even"
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_freq) * torch.arange(half, device=device) / half)
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class MeanFlowParT(nn.Module):
    """ParT wrapped for MeanFlow generation.

    Args mirror the most-used ParT knobs and add:
        num_features: F (number of features per particle to generate)
        cond_dims: tuple of per-cond-input feature dims. Each cond input has
            shape (B, C_g, M) with M either 1 (broadcast) or N (per-particle).
        time_dim: dimension of sinusoidal embeddings for t and r.
        kin_start, kin_dim: feature columns used as the (E, px, py, pz) 4-vec
            for ParT's pair-feature computation. Standard JetClass-like layout
            is kin_start=0, kin_dim=4.
        use_pair_features: default False because ParT's `pp` pair features
            compute log((p_i + p_j)^2) which is undefined for noisy 4-vectors
            (the input `z` to a MeanFlow generator). Turn back on once you
            have a stable on-shell parameterisation, or feed clean pair
            features from the conditioning instead.
        pair_input_type: as in ParT ('pp', 'ee', 'xyzt', ...). Default 'pp'.
        version: 1, 2 or 3 (default 3 = RMSNorm + SwiGLU + DropPath, ie the
            new ParT brought in by the recent upstream merge).
    """

    def __init__(
        self,
        num_features: int,
        cond_dims=(),
        time_dim: int = 128,
        kin_start: int = 0,
        kin_dim: int = 4,
        use_pair_features: bool = False,
        pair_input_type: str = "pp",
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads: int = 8,
        num_layers: int = 8,
        version: int = 3,
        for_inference: bool = False,
        use_amp: bool = False,
        compile_model: bool = False,
        **part_kwargs,
    ):
        super().__init__()
        self.num_features = num_features
        self.kin_start = int(kin_start)
        self.kin_dim = int(kin_dim)
        self.use_pair_features = use_pair_features and (kin_dim == 4)

        self.t_embed = SinusoidalEmbed(time_dim)
        self.r_embed = SinusoidalEmbed(time_dim)

        cond_dim_total = int(sum(cond_dims))
        # Per-particle input to ParT: features + t_emb + r_emb + broadcast(cond)
        input_dim = num_features + 2 * time_dim + cond_dim_total

        self.part = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_features,
            pair_input_type=pair_input_type if self.use_pair_features else "pp",
            pair_input_dim=None if self.use_pair_features else 0,
            pair_embed_dims=pair_embed_dims if self.use_pair_features else None,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=0,            # no class token, we're segmenting
            include_global_token=False,
            fc_params=(),                # final Linear added by num_classes
            for_segmentation=True,       # per-particle output
            version=version,
            for_inference=for_inference,
            use_amp=use_amp,
            compile_model=compile_model,
            trim=False,                  # generative use: keep all positions
            **part_kwargs,
        )
        # ParT's Embed/PairEmbed use BatchNorm1d on the inputs to normalise
        # raw HEP features. We rely on weaver's data-config standardisation
        # instead, and BatchNorm1d's in-place updates to `num_batches_tracked`
        # are incompatible with torch.func.jvp (the MeanFlow training mode).
        _replace_batchnorm_with_identity(self.part)

    def _build_input(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, conds):
        # z: (B, F, N); t, r: (B,); each cond: (B, C, M) or (B, C)
        B, F, N = z.shape
        t_emb = self.t_embed(t).unsqueeze(-1).expand(-1, -1, N)  # (B, time_dim, N)
        r_emb = self.r_embed(r).unsqueeze(-1).expand(-1, -1, N)
        parts = [z, t_emb, r_emb]
        for c in conds:
            if c.ndim == 3:
                if c.shape[-1] == N:
                    parts.append(c)
                else:
                    parts.append(c.expand(-1, -1, N))
            elif c.ndim == 2:
                parts.append(c.unsqueeze(-1).expand(-1, -1, N))
            else:
                raise ValueError(f"cond must be 2D or 3D, got shape {tuple(c.shape)}")
        return torch.cat(parts, dim=1)  # (B, input_dim, N)

    def forward(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, *cond,
                mask: torch.Tensor = None):
        """z: (B, F, N); t, r: (B,); cond: list of conditioning tensors.

        Returns velocity prediction of shape (B, F, N).
        """
        x = self._build_input(z, t, r, cond)
        v_4vec = None
        if self.use_pair_features:
            v_4vec = z[:, self.kin_start:self.kin_start + self.kin_dim, :].contiguous()
        out = self.part(x, v=v_4vec, mask=mask)  # (B, num_classes, N)
        return out
