"""Hard SO(3)-equivariant EGNN velocity field for MeanFlow, drop-in for
MeanFlowParT on molecular conformer generation.

Diagnosis (Codex, 2026-06-08): the non-equivariant raw-Cartesian ParT velocity
field is the dominant bottleneck for molecular generation (AMR ~2.1A, COV 0,
gen_div > ref_div = over-dispersed). The cure is a velocity field that is
SO(3)-equivariant *by construction*: scalar (invariant) messages drive coordinate
updates only through sums of relative vectors (x_i - x_j), so f(Rx) = R f(x)
exactly (Satorras et al., E(n)-GNN, 2021; cf. ET-Flow).

Interface matches MeanFlowParT exactly:
    forward(z, t, r, *cond, mask=None) -> velocity, all (B, F, N).
The kinematic block z[:, 0:kin_dim] is the 3D coordinates (kin_start MUST be 0 for
molecules); any remaining generated channels (SPICE charge / element one-hots) are
treated as invariant scalars with their own (non-equivariant) output head.

JVP-safe: only Linear / SiLU / LayerNorm-free sums / einsum-style broadcasting.
NO BatchNorm, dropout, in-place mutation, scatter, or eigendecomposition -- all of
which break torch.func.jvp (MeanFlow's training path).
"""
import torch
import torch.nn as nn
from .ParT_MeanFlow import SinusoidalEmbed


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_extra: int = 0):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden + 1 + edge_extra, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, h, pair_mask, edge_attr=None):
        # x: (B,N,3), h: (B,N,H), pair_mask: (B,N,N,1)
        B, N, H = h.shape
        diff = x.unsqueeze(2) - x.unsqueeze(1)              # (B,N,N,3)
        d2 = (diff * diff).sum(-1, keepdim=True)            # (B,N,N,1) invariant
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        eij = [hi, hj, d2] + ([edge_attr] if edge_attr is not None else [])
        m = self.phi_e(torch.cat(eij, dim=-1)) * pair_mask  # (B,N,N,H), padded pairs -> 0
        # coordinate update (equivariant): sum_j (x_i - x_j) * gate, mean over real j
        gate = self.phi_x(m) * pair_mask                    # (B,N,N,1)
        denom = pair_mask.sum(dim=2).clamp_min(1.0)         # (B,N,1)
        x = x + (diff * gate).sum(dim=2) / denom            # (B,N,3)
        # node update (invariant)
        m_i = m.sum(dim=2)                                  # (B,N,H)
        h = h + self.phi_h(torch.cat([h, m_i], dim=-1))
        return x, h


class MeanFlowEGNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        cond_dims=(),
        time_dim: int = 64,
        kin_start: int = 0,
        kin_dim: int = 3,
        hidden: int = 128,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        assert kin_start == 0, "MeanFlowEGNN assumes coordinates are the first kin_dim channels"
        assert kin_dim == 3, "EGNN coordinate block must be 3D"
        self.num_features = int(num_features)
        self.kin_dim = int(kin_dim)
        self.n_scalar = self.num_features - self.kin_dim     # extra generated (invariant) channels
        self.t_embed = SinusoidalEmbed(time_dim)
        self.r_embed = SinusoidalEmbed(time_dim)
        cond_total = int(sum(cond_dims))
        h_in = cond_total + 2 * time_dim + max(self.n_scalar, 0)
        self.embed = nn.Sequential(nn.Linear(h_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(num_layers)])
        if self.n_scalar > 0:
            self.scalar_out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                            nn.Linear(hidden, self.n_scalar))

    def forward(self, z, t, r, *cond, mask=None):
        B, F, N = z.shape
        x = z[:, :self.kin_dim].transpose(1, 2).contiguous()     # (B,N,3) coordinates
        m = mask
        if m is None:
            m = torch.ones(B, 1, N, device=z.device, dtype=z.dtype)
        m = m[:, 0, :] if m.ndim == 3 else m                     # (B,N)
        pair_mask = (m.unsqueeze(2) * m.unsqueeze(1)).unsqueeze(-1)  # (B,N,N,1)

        te = self.t_embed(t).unsqueeze(1).expand(-1, N, -1)      # (B,N,time_dim)
        re = self.r_embed(r).unsqueeze(1).expand(-1, N, -1)
        feats = [te, re]
        for c in cond:
            if c.ndim == 3:
                cc = c.transpose(1, 2) if c.shape[-1] == N else c.expand(-1, -1, N).transpose(1, 2)
            elif c.ndim == 2:
                cc = c.unsqueeze(1).expand(-1, N, -1)
            else:
                raise ValueError(f"cond must be 2D/3D, got {tuple(c.shape)}")
            feats.append(cc)
        if self.n_scalar > 0:
            feats.append(z[:, self.kin_dim:].transpose(1, 2))    # invariant extra channels
        h = self.embed(torch.cat(feats, dim=-1)) * m.unsqueeze(-1)

        x0 = x
        for layer in self.layers:
            x, h = layer(x, h, pair_mask)
        v_coord = (x - x0).transpose(1, 2)                       # (B,3,N) equivariant
        if self.n_scalar > 0:
            v_scalar = self.scalar_out(h).transpose(1, 2)        # (B,n_scalar,N)
            v = torch.cat([v_coord, v_scalar], dim=1)            # no in-place (JVP-safe)
        else:
            v = v_coord
        return v * m.unsqueeze(1)
