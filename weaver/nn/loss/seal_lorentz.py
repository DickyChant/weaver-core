"""SEAL primitives for generative models in weaver.

Ports the Lorentz / SO(3) / SO(2) generators from
``SEAL/SEAL/SEAL_cartesian.py`` and provides a vector-output deltaSEAL via
``torch.func.jvp``.  The original scalar-output deltaSEAL was designed for
classifier scores; the vector form is what we need for a generator whose
output transforms under the same group as the input.

Key entry points:
    Lorentz_gens()  -> (6, 4, 4)   3 boosts + 3 rotations on (E, px, py, pz)
    SO3_gens()      -> (3, 3, 3)   spatial rotations on (px, py, pz)
    SO2_gens()      -> (1, 2, 2)   rotation about z in (px, py)
    block_pad_generator(L, full_dim, block_slice)
                    -> embed a small generator inside a full feature axis
    delta_seal_vector(f, x, gens_in, gens_out)
                    -> enforce  df/dx . (L_in x) == L_out . f(x)  per generator
"""

import torch


def SO2_gens(dtype=torch.float32):
    Lz = torch.tensor([[0, 1], [-1, 0]], dtype=dtype)
    return torch.stack([Lz])  # (1, 2, 2)


def SO3_gens(dtype=torch.float32):
    Lx = torch.tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=dtype)
    Ly = torch.tensor([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=dtype)
    Lz = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=dtype)
    return torch.stack([Lx, Ly, Lz])  # (3, 3, 3)


def Lorentz_gens(dtype=torch.float32):
    """Generators of SO(1,3) acting on column vector (E, px, py, pz)."""
    # spatial rotations (act on (px, py, pz))
    Lz = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 0]], dtype=dtype)
    Ly = torch.tensor([[0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=dtype)
    Lx = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)
    # boosts (mix E with one spatial direction)
    Kz = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=dtype)
    Ky = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=dtype)
    Kx = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=dtype)
    return torch.stack([Kx, Ky, Kz, Lx, Ly, Lz])  # (6, 4, 4)


def block_pad_generator(L: torch.Tensor, full_dim: int, block_start: int) -> torch.Tensor:
    """Embed an MxM generator L into a (full_dim, full_dim) zero matrix at
    block_start..block_start+M on both axes. Outside the block is zero, which
    is the right rep for features we declare 'invariant'."""
    M = L.shape[-1]
    out = torch.zeros(full_dim, full_dim, dtype=L.dtype, device=L.device)
    out[block_start:block_start + M, block_start:block_start + M] = L
    return out


def block_pad_generators(gens: torch.Tensor, full_dim: int, block_start: int) -> torch.Tensor:
    """Vectorised version: (G, M, M) -> (G, full_dim, full_dim)."""
    G, M, _ = gens.shape
    out = torch.zeros(G, full_dim, full_dim, dtype=gens.dtype, device=gens.device)
    out[:, block_start:block_start + M, block_start:block_start + M] = gens
    return out


def _apply_generator(L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply L on the *feature* axis (axis -2) of x with layout (B, F, N).

    For a different layout, override the einsum string upstream.
    """
    return torch.einsum("ij,...jn->...in", L, x)


def delta_seal_vector(model_fn, x, gens_in, gens_out=None, take_mean=True,
                      reduce="mean", normalize=False):
    """Vector-output deltaSEAL via JVP.

    Enforces ``df/dx . (L_in @ x) == L_out @ f(x)`` per generator pair.

    Args:
        model_fn: callable ``x -> f(x)``. Must be pure (no in-place updates to
            module state); calling a torch ``nn.Module`` is fine.
        x: input tensor (B, F, N) (or any shape where the feature axis is -2).
        gens_in: (G, F, F) generators acting on x's feature axis.
        gens_out: (G, F_out, F_out) generators acting on f(x)'s feature axis.
            Defaults to ``gens_in`` (i.e. f shares the input rep — useful when
            f maps the noise space to a same-shape generated jet).
        take_mean: if True, return one scalar; else return per-generator
            losses of shape (G,).
        reduce: 'mean' or 'sum' across non-batch axes.
        normalize: if True, divide each per-generator violation by
            ``mean(|L_out @ f(x)|**2) + eps`` so different generators contribute
            comparably even when their magnitudes differ wildly (e.g. Lorentz
            boosts dwarf rotations at high pT).
    """
    if gens_out is None:
        gens_out = gens_in
    eps = 1e-8

    losses = []
    for L_in, L_out in zip(gens_in, gens_out):
        tangent = _apply_generator(L_in, x)
        fx, jvp = torch.func.jvp(model_fn, (x,), (tangent,))
        L_fx = _apply_generator(L_out, fx)
        diff = (jvp - L_fx)
        if reduce == "sum":
            diff_sq = diff.pow(2).flatten(1).sum(dim=1).mean()
        else:
            diff_sq = diff.pow(2).mean()
        if normalize:
            scale = L_fx.pow(2).mean() + eps
            diff_sq = diff_sq / scale
        losses.append(diff_sq)
    stacked = torch.stack(losses)
    return stacked.mean() if take_mean else stacked


def delta_seal_scalar(model_fn, x, gens_in, take_mean=True):
    """Scalar-output deltaSEAL via JVP.

    Enforces ``ds/dx . (L_in @ x) == 0`` per generator -- i.e. the output is
    invariant. This is the analogue of the SEAL paper's original deltaSEAL for
    classifier scores. Use when ``model_fn`` returns a scalar / vector of
    scalars per event that should be Lorentz-invariant by physics, but is *not*
    invariant by construction (e.g. a learned classifier head, or a learned
    estimate of jet mass before any explicit invariant computation).
    """
    losses = []
    for L_in in gens_in:
        tangent = _apply_generator(L_in, x)
        _, jvp = torch.func.jvp(model_fn, (x,), (tangent,))
        losses.append(jvp.pow(2).mean())
    stacked = torch.stack(losses)
    return stacked.mean() if take_mean else stacked
