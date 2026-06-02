"""Pairwise (2-point) physics-informed residual for MOLECULAR generation.

The molecular analog of the jet energy-energy correlator (residuals_pairwise.py).
For molecules the canonical 2-point structure is the set of pairwise interatomic
distances D_ij = ||r_i - r_j||: bond lengths (C-C ~1.5A, C-H ~1.1A, ...) are
sharply peaked and a physical conformer must reproduce them. Distances are
SO(3)+translation invariant, so a residual on them is automatically
symmetry-respecting (a synergy with the SEAL equivariance theme).

Computed on the physical 1-NFE generation x_hat = z - u(z,1,0) (real positions,
no JVP through it). Matches the generated conformer's distance distribution to
that of the REAL conformers in the same batch via a differentiable soft
histogram. Two observables:
  - "alldist" : histogram of ALL pairwise distances (the full 2-point shape)
  - "bonded"  : histogram of SHORT distances < bond_cutoff (the bond-length
                spectrum -- the sharply-peaked, most diagnostic part)

Contract (matches residuals_jet / residuals_pairwise):
    MolecularPairwiseResiduals(...)(x_hat, mask, cond, real=...) -> (scalar, info)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pair_distances(pos, mask, max_atoms=None, eps=1e-8):
    """pos: (B, 3, N) positions; mask: (B, 1, N). Returns (D, w) each (B, N, N):
    pairwise Euclidean distances and a 0/1 valid-pair weight (real-real, no self)."""
    p = pos.transpose(1, 2)                       # (B, N, 3)
    m = mask.squeeze(1)                           # (B, N)
    if max_atoms and p.shape[1] > max_atoms:
        # keep first max_atoms (positions have no natural energy ordering;
        # masking already removed pads, so this just bounds N^2 memory).
        p = p[:, :max_atoms]; m = m[:, :max_atoms]
    diff = p.unsqueeze(2) - p.unsqueeze(1)        # (B, N, N, 3)
    D = torch.sqrt((diff * diff).sum(-1) + eps)   # (B, N, N)
    w = m.unsqueeze(2) * m.unsqueeze(1)           # (B, N, N) real-real
    eye = torch.eye(w.shape[-1], device=w.device, dtype=w.dtype).unsqueeze(0)
    w = w * (1.0 - eye)                           # drop self-pairs
    return D, w


def _soft_hist(values, weights, edges, tau=0.5, chunk=2_000_000):
    """Differentiable soft histogram (chunked); returns normalised (K,)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]).clamp(min=1e-6)
    v = values.reshape(-1); w = weights.reshape(-1)
    K = centers.numel()
    hist = values.new_zeros(K)
    for i in range(0, v.numel(), chunk):
        vi = v[i:i + chunk].unsqueeze(1); wi = w[i:i + chunk].unsqueeze(1)
        d = (vi - centers.view(1, -1)) / (tau * width)
        soft = torch.exp(-d * d)
        soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-12)
        hist = hist + (soft * wi).sum(dim=0)
    return hist / hist.sum().clamp(min=1e-12)


class MolecularPairwiseResiduals(nn.Module):
    """2-point PIDM residual on interatomic distances, gen-vs-real in batch.

    Args:
        kin_slice: slice selecting the 3 position channels (default 0:3).
        pos_scale_inv: inverse of the yaml position scaling -> Angstrom
            (SPICE scales positions by 0.5 -> pos_scale_inv=2.0;
             GEOM-QM9 by 0.333 -> pos_scale_inv=3.0).
        observables: subset of {"alldist", "bonded"}.
        bond_cutoff: Angstrom; distances below this count as the bond-length
            spectrum for the "bonded" observable.
        metric: "chi2" (symmetric) or "l1" on normalised histograms.
    """

    def __init__(
        self,
        kin_slice: slice = slice(0, 3),
        pos_scale_inv: float = 2.0,
        observables=("alldist", "bonded"),
        n_bins: int = 40,
        alldist_range=(0.0, 12.0),
        bonded_range=(0.5, 2.2),
        bond_cutoff: float = 2.2,
        metric: str = "chi2",
        weights: dict | None = None,
        max_atoms: int = 64,
    ):
        super().__init__()
        self.kin_slice = kin_slice
        self.pos_scale_inv = float(pos_scale_inv)
        self.observables = tuple(observables)
        self.bond_cutoff = float(bond_cutoff)
        self.metric = metric
        self.max_atoms = max_atoms
        self.loss_weights = dict(weights or {"alldist": 1.0, "bonded": 1.0})
        self.register_buffer("alldist_edges", torch.linspace(*alldist_range, n_bins + 1))
        self.register_buffer("bonded_edges", torch.linspace(*bonded_range, n_bins + 1))

    def _hist(self, x, mask, which):
        pos = x[:, self.kin_slice] * self.pos_scale_inv     # -> Angstrom
        D, w = _pair_distances(pos, mask, self.max_atoms)
        if which == "bonded":
            # weight only short (bonded) pairs
            wb = w * (D < self.bond_cutoff).to(w.dtype)
            return _soft_hist(D, wb, self.bonded_edges)
        return _soft_hist(D, w, self.alldist_edges)

    def forward(self, x_hat, mask, cond, real=None, **kwargs):
        if real is None:
            return x_hat.new_zeros(()), {}
        if mask is None:
            mask = torch.ones(x_hat.shape[0], 1, x_hat.shape[2],
                              device=x_hat.device, dtype=x_hat.dtype)
        if self.alldist_edges.device != x_hat.device:
            self.alldist_edges = self.alldist_edges.to(x_hat.device)
            self.bonded_edges = self.bonded_edges.to(x_hat.device)

        total = x_hat.new_zeros(())
        info = {}
        for obs in self.observables:
            hg = self._hist(x_hat, mask, obs)
            with torch.no_grad():
                hr = self._hist(real, mask, obs)
            if self.metric == "l1":
                d = (hg - hr).abs().sum()
            else:
                d = ((hg - hr) ** 2 / (hg + hr + 1e-8)).sum()
            total = total + self.loss_weights.get(obs, 1.0) * d
            info[f"{obs}_dist"] = float(d.detach())
        info["n_obs"] = len(self.observables)
        return total, info
