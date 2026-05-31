"""Physics residuals for jet generation, mirroring the PIDM pattern.

For a jet generated as a point cloud `x_hat[B, F, N]`, the residual is the
violation of aggregated-quantity sum rules against per-event targets carried
in the conditioning. Plugged into `MeanFlowSEALLoss` via the `residual_func`
argument, it forms the analog of PIDM's `c_residual * residual_log_likelihood`
(src/denoising_utils.py:680-691) in our MeanFlow setup.

The contract:
    JetResiduals(...)(x_hat, mask, cond) -> (scalar_loss, info_dict)

`x_hat` is in the data-config's scaled units; the residual module knows the
inverse scaling (`kin_scale_inv`, default 100 for the JetClass yaml that
multiplies by 0.01) and the cond layout, so it can undo both sides and
compare in physical units (GeV). All residuals are relative (divided by the
target) so weights are interpretable across pT / E / mass.

Typical channel/cond layout for `runs/configs/jetclass_gen_hbb_ttbar.yaml`:

    pf_features cols 0:4 = (E, px, py, pz) * 0.01   ->   kin_scale_inv = 100
    pf_cond
        index 0   jet_pt_log,     center=5.5, scale=0.5, transform=log
        index 1   jet_e_log,      center=6.0, scale=0.5, transform=log
        index 2   jet_eta,        center=0.0, scale=2.0, transform=identity
        index 3   jet_sdmass_log, center=4.0, scale=0.7, transform=log_offset (+1e-3)
        index 4   cls_hbb         (class flag, no transform)
        index 5   cls_ttbar

`cond_layout` is a dict observable-name -> {index, center, scale, transform[, offset]}.
The provided default matches that yaml exactly; pass a custom layout from the
network-config if the yaml channels differ.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


DEFAULT_JETCLASS_LAYOUT: dict = {
    "E":    {"index": 1, "center": 6.0, "scale": 0.5, "transform": "log"},
    "pt":   {"index": 0, "center": 5.5, "scale": 0.5, "transform": "log"},
    "eta":  {"index": 2, "center": 0.0, "scale": 2.0, "transform": "identity"},
    "mass": {"index": 3, "center": 4.0, "scale": 0.7, "transform": "log_offset", "offset": 1e-3},
}


class JetResiduals(nn.Module):
    """Compute jet-level sum-rule residuals on a generated point cloud.

    Each enabled observable contributes a relative-squared residual:
        r_obs = ((sum_gen - target_phys) / max(target_phys, 1.0))^2

    The 1.0-clamp on the denominator avoids blow-up on rare low-pT jets and
    keeps the loss dimensionally consistent across observables (it is the
    Huber-free squared relative error in [pT or E units]/[pT or E units]).

    Args:
        kin_slice: slice over the F axis selecting (E, px, py, pz).
        kin_scale_inv: inverse of the yaml scaling factor on the kin block.
        cond_layout: dict observable-name -> param dict (see DEFAULT_JETCLASS_LAYOUT).
        loss_weights: dict observable-name -> weight. Set to {} to disable that
            observable. Default uses E, pt, mass at equal weight.
        reduce: 'mean' -> scalar; 'none' -> per-event tensor (B,).
    """

    def __init__(
        self,
        kin_slice: slice = slice(0, 4),
        kin_scale_inv: float = 100.0,
        cond_layout: dict | None = None,
        loss_weights: dict | None = None,
        reduce: str = "mean",
    ):
        super().__init__()
        self.kin_slice = kin_slice
        self.kin_scale_inv = float(kin_scale_inv)
        self.cond_layout = dict(cond_layout if cond_layout is not None else DEFAULT_JETCLASS_LAYOUT)
        self.loss_weights = dict(loss_weights if loss_weights is not None
                                 else {"E": 1.0, "pt": 1.0, "mass": 0.5})
        assert reduce in ("mean", "none"), reduce
        self.reduce = reduce

    @staticmethod
    def _untransform(c: torch.Tensor, layout: dict) -> torch.Tensor:
        """Recover the physical value from a standardized cond column.

        Weaver standardizes inputs as `standardized = (raw - center) * scale`
        (see weaver/utils/data/tools.py::_batched_fused_*_pad, `val = (val -
        center) * scale_v`). The correct inverse is therefore
            raw = standardized / scale + center
        NOT `standardized * scale + center` (the original bug, which compressed
        every target toward exp(center) and trained the residual to wrong
        sum-rule targets).
        """
        i = layout["index"]
        center = layout["center"]
        scale = layout["scale"]
        kind = layout["transform"]
        u = c[:, i] / scale + center
        if kind == "log":
            return torch.exp(u)
        if kind == "log_offset":
            return torch.exp(u) - layout.get("offset", 0.0)
        if kind == "identity":
            return u
        raise ValueError(f"unknown transform: {kind}")

    def _jet_sums(self, x_hat: torch.Tensor, mask: torch.Tensor):
        """Return (E_sum, px_sum, py_sum, pz_sum, pt_sum, p_tot, mass) in GeV."""
        kin = x_hat[:, self.kin_slice] * self.kin_scale_inv  # (B, 4, N) in GeV
        if mask is not None:
            m = mask.float().squeeze(1)  # (B, N)
        else:
            m = torch.ones_like(kin[:, 0])
        E_sum = (kin[:, 0] * m).sum(dim=-1)
        px_sum = (kin[:, 1] * m).sum(dim=-1)
        py_sum = (kin[:, 2] * m).sum(dim=-1)
        pz_sum = (kin[:, 3] * m).sum(dim=-1)
        pt_sum = torch.sqrt(px_sum.pow(2) + py_sum.pow(2) + 1e-12)
        p_tot = torch.sqrt(pt_sum.pow(2) + pz_sum.pow(2) + 1e-12)
        m2 = (E_sum.pow(2) - p_tot.pow(2)).clamp_min(0.0)
        mass = m2.sqrt()
        return E_sum, px_sum, py_sum, pz_sum, pt_sum, p_tot, mass

    def forward(self, x_hat: torch.Tensor, mask: torch.Tensor | None,
                cond: Sequence[torch.Tensor]):
        """x_hat: (B, F, N) generated, mask: (B, 1, N) or None,
           cond: list/tuple of cond tensors (B, Ck, Mk). Uses cond[0] (pf_cond)."""
        if not cond:
            return x_hat.new_zeros(()), {}
        c = cond[0]
        if c.dim() == 3:
            c = c.squeeze(-1)  # (B, Ck)

        E_sum, px_sum, py_sum, pz_sum, pt_sum, p_tot, mass = self._jet_sums(x_hat, mask)

        components = []
        info = {}
        if "E" in self.loss_weights and "E" in self.cond_layout:
            t = self._untransform(c, self.cond_layout["E"])
            r = (E_sum - t) / t.clamp_min(1.0)
            components.append(self.loss_weights["E"] * r.pow(2))
            info["E_abs"] = float(r.detach().abs().mean())
            info["E_target_mean"] = float(t.detach().mean())
            info["E_gen_mean"] = float(E_sum.detach().mean())
        if "pt" in self.loss_weights and "pt" in self.cond_layout:
            t = self._untransform(c, self.cond_layout["pt"])
            r = (pt_sum - t) / t.clamp_min(1.0)
            components.append(self.loss_weights["pt"] * r.pow(2))
            info["pt_abs"] = float(r.detach().abs().mean())
            info["pt_target_mean"] = float(t.detach().mean())
            info["pt_gen_mean"] = float(pt_sum.detach().mean())
        if "mass" in self.loss_weights and "mass" in self.cond_layout:
            t = self._untransform(c, self.cond_layout["mass"]).clamp_min(0.0)
            r = (mass - t) / t.clamp_min(1.0)
            components.append(self.loss_weights["mass"] * r.pow(2))
            info["mass_abs"] = float(r.detach().abs().mean())
            info["mass_target_mean"] = float(t.detach().mean())
            info["mass_gen_mean"] = float(mass.detach().mean())
        if "eta" in self.loss_weights and "eta" in self.cond_layout:
            t = self._untransform(c, self.cond_layout["eta"])
            # eta from sum: only meaningful for nontrivial p_tot
            eta_gen = 0.5 * (
                (p_tot + pz_sum).clamp_min(1e-6) / (p_tot - pz_sum).clamp_min(1e-6)
            ).log()
            r = eta_gen - t  # eta is unbounded around 0, use absolute
            components.append(self.loss_weights["eta"] * r.pow(2))
            info["eta_abs"] = float(r.detach().abs().mean())

        if not components:
            return x_hat.new_zeros(()), info

        per_event = torch.stack(components, dim=0).sum(dim=0)
        info["n_observables"] = int(len(components))
        if self.reduce == "mean":
            return per_event.mean(), info
        return per_event, info
