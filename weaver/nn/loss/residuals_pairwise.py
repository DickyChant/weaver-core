"""Pairwise (2-point) physics-informed residual for jet generation.

Where `residuals_jet.JetResiduals` enforces 1-point aggregate sum rules
(total E / pT / mass), this enforces the 2-point STRUCTURE of the generated
jet: the spectrum of pairwise quantities (pairwise invariant mass m_ij^2 and
angular separation Delta-R_ij), and the Energy-Energy Correlator (EEC), a
standard jet-substructure observable.

Key idea (the "pairwise feature as a PIDM loss, not an attention bias" angle):
ParT puts pairwise features (m_ij^2, Delta-R, kt, z) into the attention bias.
That breaks MeanFlow because forward-mode AD (torch.func.jvp) of log/sqrt of
clamped near-zero pairwise masses on NOISY z blows up. Here we instead compute
pairwise observables on the PHYSICAL 1-NFE generation
    x_hat = z - u(z, t=1, r=0, cond)
(a real jet -> positive masses -> well-defined, no JVP through it), and match
their batch distribution to that of the REAL jets in the same batch. So the
pairwise physics enters the loss, never the noisy forward.

Contract (matches residuals_jet): callable
    PairwiseResiduals(...)(x_hat, mask, cond, real=...) -> (scalar, info)
The extra `real` kwarg (the batch's real data tensor) is what we match against;
MeanFlowSEALLoss passes it through.

Observables (each a fixed-bin histogram, compared gen-vs-real by a symmetric
chi2 / L1 on the normalised histograms):
  - lnm2  : log pairwise invariant mass^2,  weighted by E_i E_j (EEC-like mass)
  - lndR  : log Delta-R_ij,                  weighted by E_i E_j  (the EEC proper)
Both are scale/orientation robust and dominated by real jet substructure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pair_quantities(kin: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """kin: (B, 4, N) = (E, px, py, pz) in GeV. mask: (B, 1, N).
    Returns per-pair (lnm2, lndR, w) each (B, N, N), w = E_i E_j * mask_i mask_j.
    """
    E = kin[:, 0]                      # (B, N)
    px, py, pz = kin[:, 1], kin[:, 2], kin[:, 3]
    m = mask.squeeze(1)               # (B, N)

    # pairwise invariant mass^2: (p_i + p_j)^2 = m_i^2 + m_j^2 + 2(E_iE_j - p_i.p_j)
    Ei, Ej = E.unsqueeze(2), E.unsqueeze(1)
    pxi, pxj = px.unsqueeze(2), px.unsqueeze(1)
    pyi, pyj = py.unsqueeze(2), py.unsqueeze(1)
    pzi, pzj = pz.unsqueeze(2), pz.unsqueeze(1)
    dot = Ei * Ej - (pxi * pxj + pyi * pyj + pzi * pzj)   # E_iE_j - p_i.p_j
    m2_i = (E * E - (px * px + py * py + pz * pz)).clamp(min=0)
    mij2 = (m2_i.unsqueeze(2) + m2_i.unsqueeze(1) + 2 * dot).clamp(min=eps)
    lnm2 = torch.log(mij2)

    # angular separation Delta-R = sqrt(d_eta^2 + d_phi^2)
    pt = torch.sqrt(px * px + py * py + eps)
    eta = 0.5 * torch.log(((torch.sqrt(px*px+py*py+pz*pz+eps) + pz).clamp(min=eps)) /
                          ((torch.sqrt(px*px+py*py+pz*pz+eps) - pz).clamp(min=eps)))
    phi = torch.atan2(py, px)
    deta = eta.unsqueeze(2) - eta.unsqueeze(1)
    dphi = torch.atan2(torch.sin(phi.unsqueeze(2) - phi.unsqueeze(1)),
                       torch.cos(phi.unsqueeze(2) - phi.unsqueeze(1)))
    dR = torch.sqrt(deta * deta + dphi * dphi + eps)
    lndR = torch.log(dR.clamp(min=eps))

    # EEC weight: E_i E_j (normalised per jet later), masked, no self-pairs.
    # Clamp energies to >= 0: physical energies are non-negative, but early in
    # training x_hat can have negative E. Negative weights would make the
    # histogram normaliser near-zero/negative and blow the chi2 up.
    Ei_p, Ej_p = Ei.clamp(min=0), Ej.clamp(min=0)
    w = (Ei_p * Ej_p) * (m.unsqueeze(2) * m.unsqueeze(1))
    eye = torch.eye(w.shape[-1], device=w.device, dtype=w.dtype).unsqueeze(0)
    w = w * (1.0 - eye)
    return lnm2, lndR, w


def _soft_hist(values: torch.Tensor, weights: torch.Tensor, edges: torch.Tensor,
               tau: float = 0.5, chunk: int = 1_000_000) -> torch.Tensor:
    """Differentiable soft histogram. values,weights: (B,N,N); edges: (K+1,).
    Returns (K,) batch-summed soft-binned weight, normalised to sum 1.
    Soft assignment via a Gaussian kernel so gradients flow to values.

    Chunked over the flattened pair dimension so the (M, K) soft-assignment
    matrix never materialises all at once -- M = B*N*N can be ~4M for JetClass
    (N=128, batch 256), which OOMs a 40 GB GPU at K=24 bins. `chunk` caps the
    rows processed per step; gradients are identical to the unchunked version.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])           # (K,)
    width = (edges[1] - edges[0]).clamp(min=1e-6)
    v = values.reshape(-1)                              # (M,)
    w = weights.reshape(-1)                             # (M,)
    K = centers.numel()
    hist = values.new_zeros(K)
    for i in range(0, v.numel(), chunk):
        vi = v[i:i + chunk].unsqueeze(1)               # (m, 1)
        wi = w[i:i + chunk].unsqueeze(1)               # (m, 1)
        d = (vi - centers.view(1, -1)) / (tau * width)
        soft = torch.exp(-d * d)                       # (m, K)
        soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-12)
        hist = hist + (soft * wi).sum(dim=0)
    return hist / hist.sum().clamp(min=1e-12)


class PairwiseResiduals(nn.Module):
    """2-point PIDM residual: match gen vs real pairwise observables in-batch.

    Args:
        kin_slice, kin_scale_inv: as in JetResiduals (cols 0:4, *100 -> GeV).
        observables: subset of {"lnm2", "lndR"}.
        n_bins, ranges: histogram config per observable.
        metric: "chi2" (symmetric) or "l1" on normalised histograms.
    """

    def __init__(
        self,
        kin_slice: slice = slice(0, 4),
        kin_scale_inv: float = 100.0,
        observables=("lnm2", "lndR"),
        n_bins: int = 24,
        lnm2_range=(-4.0, 8.0),
        lndR_range=(-6.0, 1.0),
        metric: str = "chi2",
        weights: dict | None = None,
        max_particles: int = 64,   # cap pairwise to top-|E| particles (memory)
    ):
        super().__init__()
        self.max_particles = max_particles
        self.kin_slice = kin_slice
        self.kin_scale_inv = float(kin_scale_inv)
        self.observables = tuple(observables)
        self.metric = metric
        self.loss_weights = dict(weights or {"lnm2": 1.0, "lndR": 1.0})
        self.register_buffer("lnm2_edges", torch.linspace(*lnm2_range, n_bins + 1))
        self.register_buffer("lndR_edges", torch.linspace(*lndR_range, n_bins + 1))

    def _topk_by_energy(self, x, mask):
        """Keep only the top `max_particles` by |E| per jet -- bounds the (N,N)
        pairwise memory and is physically motivated (EEC is dominated by
        high-energy particles). No-op if N <= max_particles."""
        N = x.shape[-1]
        if self.max_particles is None or N <= self.max_particles:
            return x, mask
        E = x[:, self.kin_slice.start].abs()           # (B, N) energy magnitude
        if mask is not None:
            E = E * mask.squeeze(1)
        idx = E.topk(self.max_particles, dim=-1).indices  # (B, k)
        xg = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
        mg = torch.gather(mask, 2, idx.unsqueeze(1)) if mask is not None else None
        return xg, mg

    def _hist_pair(self, x, mask, which):
        x, mask = self._topk_by_energy(x, mask)
        kin = x[:, self.kin_slice] * self.kin_scale_inv
        lnm2, lndR, w = _pair_quantities(kin, mask)
        if which == "lnm2":
            return _soft_hist(lnm2, w, self.lnm2_edges)
        return _soft_hist(lndR, w, self.lndR_edges)

    def forward(self, x_hat, mask, cond, real=None):
        if real is None:
            return x_hat.new_zeros(()), {}
        if mask is None:
            mask = torch.ones(x_hat.shape[0], 1, x_hat.shape[2],
                              device=x_hat.device, dtype=x_hat.dtype)
        # ensure buffers on the right device (loss module isn't .to(dev)'d)
        if self.lnm2_edges.device != x_hat.device:
            self.lnm2_edges = self.lnm2_edges.to(x_hat.device)
            self.lndR_edges = self.lndR_edges.to(x_hat.device)

        total = x_hat.new_zeros(())
        info = {}
        for obs in self.observables:
            hg = self._hist_pair(x_hat, mask, obs)             # gen (grad)
            with torch.no_grad():
                hr = self._hist_pair(real, mask, obs)          # real target
            if self.metric == "l1":
                d = (hg - hr).abs().sum()
            else:  # symmetric chi2
                d = ((hg - hr) ** 2 / (hg + hr + 1e-8)).sum()
            total = total + self.loss_weights.get(obs, 1.0) * d
            info[f"{obs}_dist"] = float(d.detach())
        info["n_obs"] = len(self.observables)
        return total, info
