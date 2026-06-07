"""
MeanFlow loss for generative training inside the weaver framework.

Ported from CaloDiffu_HGCAL/scripts/CaloDiffu.py (sample_t_r, adaptive_l2_loss,
compute_loss_meanflow_old). Trimmed to the minimum needed for a weaver-style
get_loss/get_train_fn contract.

Conventions:
    z(t) = (1 - t) * x_data + t * eps,   eps ~ N(0, I)
    target velocity: v = eps - x_data
    model is called as model(z, t, r, *cond) where cond carries per-event
    conditioning (e.g. energy, jet kinematics). For r == t we get the
    instantaneous flow-matching mode; for r < t we get the MeanFlow target
    u_tgt = v - (t - r) * dudt, with dudt obtained via JVP.
"""

import contextlib

import numpy as np
import torch
import torch.nn as nn

from weaver.nn.loss.seal_lorentz import (
    Lorentz_gens,
    SO3_gens,
    block_pad_generators,
    delta_seal_vector,
)


def _sdpa_jvp_safe_ctx():
    """Force scaled_dot_product_attention onto the math backend.

    Flash-attention and memory-efficient SDPA don't implement forward-mode AD
    (torch.func.jvp), which is what MeanFlow + SEAL both depend on. The math
    backend is slower but JVP-compatible; we enter it only around the JVP
    region so non-generative paths are unaffected.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """sg(w) * ||delta||^2 with w = 1 / (||delta||^2 + c)^(1-gamma)."""
    dims = tuple(range(1, error.ndim))
    delta_sq = (error ** 2).mean(dim=dims) if dims else (error ** 2).mean(dim=0)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (stopgrad(w) * delta_sq).mean()


def sample_t_r(batch_size, device, flow_ratio=0.75, time_dist=("lognorm", -0.4, 1.0)):
    """Sample (t, r) pairs with t >= r and with probability flow_ratio set r = t."""
    kind = time_dist[0]
    if kind == "uniform":
        samples = np.random.rand(batch_size, 2).astype(np.float32)
    elif kind == "lognorm":
        mu, sigma = time_dist[1], time_dist[2]
        z = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unknown time_dist kind {kind!r}")
    t_np = np.maximum(samples[:, 0], samples[:, 1])
    r_np = np.minimum(samples[:, 0], samples[:, 1])
    if flow_ratio > 0:
        idx = np.random.permutation(batch_size)[: int(flow_ratio * batch_size)]
        r_np[idx] = t_np[idx]
    return (
        torch.as_tensor(t_np, device=device),
        torch.as_tensor(r_np, device=device),
    )


def _expand_like(t, like):
    return t.view(t.size(0), *([1] * (like.ndim - 1)))


class MeanFlowLoss(nn.Module):
    """Callable wrapper holding hyperparameters for the MeanFlow training loss.

    The model passed at forward time must accept (z, t, r, *cond) and return
    the predicted velocity u(z, t, r | cond).
    """

    def __init__(
        self,
        flow_ratio: float = 0.75,
        time_dist=("lognorm", -0.4, 1.0),
        jvp_api: str = "func",
        adaptive_gamma: float = 0.5,
        adaptive_c: float = 1e-3,
    ):
        super().__init__()
        assert jvp_api in ("func", "autograd"), jvp_api
        self.flow_ratio = float(flow_ratio)
        self.time_dist = tuple(time_dist)
        self.jvp_api = jvp_api
        self.adaptive_gamma = float(adaptive_gamma)
        self.adaptive_c = float(adaptive_c)

    def forward(self, model, data, *cond, mask=None):
        # mask is unused by the velocity loss itself but accepted so the
        # generative trainer can call all loss variants with the same kwargs.
        del mask  # explicit "intentionally unused"
        device = data.device
        bsz = data.shape[0]
        eps = torch.randn_like(data)
        t, r = sample_t_r(bsz, device, self.flow_ratio, self.time_dist)
        t_ = _expand_like(t, data)
        r_ = _expand_like(r, data)
        z = (1.0 - t_) * data + t_ * eps
        v = eps - data

        def fn(z_, t_in, r_in):
            return model(z_, t_in.flatten(), r_in.flatten(), *cond)

        with _sdpa_jvp_safe_ctx():
            if self.jvp_api == "func":
                u, dudt = torch.func.jvp(
                    fn, (z, t_, r_), (v, torch.ones_like(t_), torch.zeros_like(r_))
                )
            else:
                u, dudt = torch.autograd.functional.jvp(
                    fn, (z, t_, r_), (v, torch.ones_like(t_), torch.zeros_like(r_)),
                    create_graph=True,
                )

        u_tgt = v - (t_ - r_) * dudt
        loss = adaptive_l2_loss(u - stopgrad(u_tgt), gamma=self.adaptive_gamma, c=self.adaptive_c)
        info = {
            "mse": (stopgrad(u - u_tgt) ** 2).mean().item(),
            "t_mean": t.mean().item(),
            "r_mean": r.mean().item(),
        }
        return loss, info


class MeanFlowSEALLoss(nn.Module):
    """MeanFlow + deltaSEAL on the 1-NFE generation.

    Two terms:
    1. Standard MeanFlow loss on the full feature vector (as in `MeanFlowLoss`).
    2. Vector-output deltaSEAL on the 1-NFE generation
           f(z, cond) = z - u(z, t=1, r=0, cond)
       with block-diagonal Lorentz/SO(3) generators on the kinematic columns
       (cols `kin_start..kin_start+kin_dim`) and zero generators on the rest
       of the feature vector. Per the design discussion: kinematics carry the
       symmetry constraint; detector features are declared invariant and the
       data likelihood (the MeanFlow term) is responsible for getting their
       correlations with the kinematics right.

    The SEAL term is non-trivial *because* it goes through the model u: the
    constraint  df/dz . (L z) = L . f(z)  is an equivariance statement on the
    learned generator, not a statement about the analytic structure of
    invariants. Computed via `torch.func.jvp` so each generator costs one
    extra forward; with `group='so3'` only 3 JVPs, with `group='lorentz'` 6.

    For cost control, set `num_gens_per_step` < total to randomly sample a
    subset of generators each training step.
    """

    def __init__(
        self,
        flow_ratio: float = 0.75,
        time_dist=("lognorm", -0.4, 1.0),
        jvp_api: str = "func",
        seal_lambda: float = 1.0,
        kin_start: int = 0,
        kin_dim: int = 4,
        group: str = "lorentz",
        num_gens_per_step: int = -1,
        normalize_per_generator: bool = True,
        # --- Augmented-Lagrangian / MDMM auto-balancing ---
        # When `adaptive_lambda=True` we treat SEAL as a soft constraint
        # `SEAL <= target_violation` and run dual ascent on `seal_lambda`:
        #   lambda <- clip( lambda + lambda_lr * (EMA(SEAL) - target), [0, lambda_max] )
        # The fixed `seal_lambda` arg above is used as the initial value.
        # Set `target_violation` to a small positive number (the level of
        # equivariance violation you'll tolerate at convergence). The EMA
        # smooths the SEAL value to avoid lambda oscillation.
        adaptive_lambda: bool = False,
        target_violation: float = 1e-3,
        lambda_lr: float = 1e-2,
        lambda_max: float = 100.0,
        seal_ema_alpha: float = 0.95,
        # --- PIDM-style residual on the 1-NFE generation -----------------
        # `residual_func(x_hat, mask, cond) -> (scalar, info_dict)` is called
        # on the same 1-NFE estimate f(z) used by SEAL. Use it to inject
        # physics-derived constraints on aggregated observables (jet E/pT/mass
        # sum rules, charge neutrality, ...). See weaver.nn.loss.residuals_jet
        # for the JetResiduals helper that matches the JetClass yaml layout.
        residual_func=None,
        lam_residual: float = 0.0,
        # Where the residual is enforced. By default (=1) the residual acts on
        # the 1-NFE generation f(z) = z - u(z, 1, 0) -- the same state mismatch
        # SEAL suffers: it constrains the one-shot sample, NOT the states the
        # K-step Euler sampler actually visits. Set > 1 to instead generate via
        # a `residual_sample_steps`-step Euler rollout (WITH grad) and enforce
        # the residual on THAT multi-step sample, so the physics constraint
        # matches what we evaluate. Costs `residual_sample_steps` extra forward
        # passes with backprop through the whole rollout.
        residual_sample_steps: int = 1,
        # --- Where SEAL enforces equivariance --------------------------------
        # "generator_endpoint" (default, original behaviour): constrain the
        #   1-NFE generator f(z) = z - u(z, t=1, r=0, cond). This makes the
        #   model equivariant ONLY at the (t=1, r=0) operating point. Multi-step
        #   Euler sampling calls u at intermediate (t, r) that are NOT
        #   constrained, so sampled-output equivariance erodes with step count.
        # "velocity_sampled" (recommended for integrator-agnostic equivariance):
        #   constrain the velocity field u(z, t, r, cond) itself at (t, r) drawn
        #   from the sampling/time distribution. For LINEAR group actions R, an
        #   equivariant velocity field gives EXACTLY equivariant explicit-Euler
        #   sampling at any step count / schedule:
        #       g(Rz) = Rz - h u(Rz,t,r) = R(z - h u(z,t,r)) = R g(z).
        #   Over SGD steps the sampled (t, r) cover the whole trajectory.
        seal_target: str = "generator_endpoint",
        # (t,r) sampling for seal_target="velocity_sampled":
        #   "training"        -> sample_t_r (MeanFlow training distribution)
        #   "sampler_matched" -> the Euler grid the sampler uses (recommended;
        #                        constrains the velocity along the actual
        #                        sampling trajectory). seal_sampler_steps sets
        #                        the grid resolution (use the eval sample_steps).
        seal_time_dist: str = "sampler_matched",
        seal_sampler_steps: int = 10,
        # State at which to enforce velocity SEAL (see forward()):
        #   "gaussian" (default, original) | "interpolant" | "trajectory".
        # The diagnostic showed "gaussian" leaves the sampler's actual low-t
        # states unconstrained; "interpolant"/"trajectory" fix the state mismatch.
        seal_state: str = "gaussian",
        seal_traj_steps: int = 5,
        combined_w_endpoint: float = 1.0,
        combined_w_velocity: float = 1.0,
    ):
        super().__init__()
        assert group in ("lorentz", "so3"), group
        assert kin_dim in (3, 4), kin_dim
        if group == "lorentz" and kin_dim != 4:
            raise ValueError("Lorentz generators are 4x4; set kin_dim=4 or use group='so3'")
        if group == "so3" and kin_dim not in (3, 4):
            raise ValueError("SO(3) wants kin_dim=3 (px,py,pz) or kin_dim=4 (E,px,py,pz, rotates spatial part)")
        assert seal_target in ("generator_endpoint", "velocity_sampled", "endpoint_sampled", "combined"), seal_target
        assert seal_time_dist in ("training", "sampler_matched"), seal_time_dist
        assert seal_state in ("gaussian", "interpolant", "trajectory"), seal_state
        self.seal_target = seal_target
        self._seal_time_dist = seal_time_dist
        self._seal_sampler_steps = int(seal_sampler_steps)
        self._seal_state = seal_state
        self._seal_traj_steps = int(seal_traj_steps)
        self._combined_w_endpoint = float(combined_w_endpoint)
        self._combined_w_velocity = float(combined_w_velocity)
        self.meanflow = MeanFlowLoss(flow_ratio=flow_ratio, time_dist=time_dist, jvp_api=jvp_api)
        self._flow_ratio = float(flow_ratio)
        self._time_dist = tuple(time_dist)
        self.kin_start = int(kin_start)
        self.kin_dim = int(kin_dim)
        self.group = group
        self.num_gens_per_step = int(num_gens_per_step)
        self.normalize_per_generator = bool(normalize_per_generator)
        self._cached_gens = None  # built lazily once we know the feature dim

        # SEAL weight: buffer so it persists in checkpoints + DDP sync.
        # We update it in-place during forward() without grad if adaptive.
        self.register_buffer("seal_lambda", torch.tensor(float(seal_lambda)))
        self.adaptive_lambda = bool(adaptive_lambda)
        self.target_violation = float(target_violation)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.seal_ema_alpha = float(seal_ema_alpha)
        # EMA of SEAL violation; initialised to 0 (will warm up over first ~10 steps)
        self.register_buffer("seal_ema", torch.tensor(0.0))
        # Counter to detect first step (skip EMA bias correction beyond a few steps)
        self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long))

        # PIDM-style residual hook. Kept as an attribute (not a buffer / module
        # registration via add_module) because callers may pass a plain
        # callable; if it is an nn.Module it gets registered via setattr.
        self.residual_func = residual_func
        self.lam_residual = float(lam_residual)
        self.residual_sample_steps = int(residual_sample_steps)

    def _build_generators(self, feature_dim: int, device, dtype):
        if self.group == "lorentz":
            small = Lorentz_gens(dtype=dtype).to(device)  # (6, 4, 4)
            block_start = self.kin_start
        else:  # so3
            small = SO3_gens(dtype=dtype).to(device)  # (3, 3, 3)
            # If kin_dim == 4 (E + 3-momentum), rotate only the spatial part
            # (offset by 1). If kin_dim == 3, rotate the whole block.
            block_start = self.kin_start + (1 if self.kin_dim == 4 else 0)
        return block_pad_generators(small, feature_dim, block_start)

    def forward(self, model, data, *cond, mask=None):
        # term 1: MeanFlow on full features (cond is the same)
        mf_loss, info = self.meanflow(model, data, *cond)
        info = dict(info)
        info["mf"] = float(mf_loss.detach())

        # term 2: deltaSEAL on the 1-NFE generation w.r.t. fresh Gaussian noise
        device = data.device
        B, F, N = data.shape[0], data.shape[1], data.shape[2] if data.ndim == 3 else None
        if self._cached_gens is None or self._cached_gens.shape[-1] != F or self._cached_gens.device != device:
            self._cached_gens = self._build_generators(F, device, data.dtype)
        gens = self._cached_gens  # (G, F, F)

        # subsample generators if requested (random subset each call)
        G = gens.shape[0]
        if 0 < self.num_gens_per_step < G:
            idx = torch.randperm(G, device=device)[: self.num_gens_per_step]
            gens_use = gens[idx]
        else:
            gens_use = gens

        # Choose the STATE at which to enforce SEAL. The diagnostic showed the
        # velocity is equivariant at Gaussian z (~0.09) but the actual sampler
        # visits states where violation explodes to ~0.55 at low t -- so
        # constraining at Gaussian z (the default) leaves the sampler's true
        # states unconstrained. seal_state picks a more representative state:
        #   "gaussian"    : z ~ N(0,I)  (original; only matches the t=1 endpoint)
        #   "interpolant" : z = (1-t)*data + t*eps  (the MeanFlow training state
        #                   at time t -- the distribution the velocity is trained
        #                   on, far closer to the sampler's states than Gaussian)
        #   "trajectory"  : run a few no-grad Euler steps from Gaussian to reach
        #                   the model's own z_k, then enforce SEAL there (closest
        #                   to what the sampler actually evaluates).
        # Build (seal_fn, z) for one target spec. Factored out so the "combined"
        # target can enforce two operating points in one step.
        def _build_seal(target):
            Bb = data.shape[0]
            if target == "velocity_sampled":
                # Velocity-field equivariance du/dz.(Lz)==L.u(z,t,r) at the (t,r)
                # the SAMPLER evaluates. sampler_matched: t on the Euler grid,
                # r = t - 1/steps (the actual step). This is the variant that
                # empirically recovers multi-step equivariance.
                if self._seal_time_dist == "sampler_matched":
                    steps = self._seal_sampler_steps
                    k = torch.randint(0, steps, (Bb,), device=device)
                    t_s = (1.0 - k.to(data.dtype) / steps)
                    r_s = (t_s - 1.0 / steps).clamp(min=0.0)
                else:
                    t_s, r_s = sample_t_r(Bb, device, self._flow_ratio, self._time_dist)
                t_col = _expand_like(t_s, data)
                if self._seal_state == "interpolant":
                    eps = torch.randn_like(data)
                    zz = (1.0 - t_col) * data + t_col * eps
                elif self._seal_state == "trajectory":
                    with torch.no_grad(), _sdpa_jvp_safe_ctx():
                        zt = torch.randn_like(data)
                        n_pre = int(self._seal_traj_steps)
                        tv = torch.linspace(1.0, 0.0, n_pre + 1, device=device)
                        for j in range(n_pre):
                            tj = torch.full((Bb,), tv[j].item(), device=device, dtype=data.dtype)
                            rj = torch.full((Bb,), tv[j + 1].item(), device=device, dtype=data.dtype)
                            v = model(zt, tj, rj, *cond)
                            step_mask = (t_s < tj).view(-1, *([1] * (data.ndim - 1))).to(data.dtype)
                            zt = zt - step_mask * (tj - rj).view(-1, *([1] * (data.ndim - 1))) * v
                    zz = zt.detach()
                else:
                    zz = torch.randn_like(data)

                def fn(z_):
                    return model(z_, t_s, r_s, *cond)
                return fn, zz
            elif target == "endpoint_sampled":
                # Endpoint estimate at every t: x_hat = z_t - t*u(z_t,t,0). Helps
                # the 1-NFE (r=0) operating point across all t; does NOT transfer
                # to the multi-step sampler (wrong r) -- see the combined target.
                if self._seal_time_dist == "sampler_matched":
                    steps = self._seal_sampler_steps
                    k = torch.randint(0, steps, (Bb,), device=device)
                    t_s = (1.0 - k.to(data.dtype) / steps).clamp(min=1.0 / steps)
                else:
                    t_s, _r = sample_t_r(Bb, device, self._flow_ratio, self._time_dist)
                zeros = torch.zeros(Bb, device=device, dtype=data.dtype)
                t_col = _expand_like(t_s, data)
                if self._seal_state == "gaussian":
                    zz = torch.randn_like(data)
                else:
                    eps = torch.randn_like(data)
                    zz = (1.0 - t_col) * data + t_col * eps

                def fn(z_):
                    v = model(z_, t_s, zeros, *cond)
                    return z_ - t_col * v
                return fn, zz
            else:  # generator_endpoint: 1-NFE map f(z) = z - u(z,1,0,cond)
                ones = torch.ones(Bb, device=device, dtype=data.dtype)
                zeros = torch.zeros(Bb, device=device, dtype=data.dtype)

                def fn(z_):
                    return z_ - model(z_, ones, zeros, *cond)
                return fn, torch.randn_like(data)

        seal_parts = {}
        if self.seal_target == "combined":
            # Operating-point locality showed each single constraint helps only
            # its own operating point (endpoint -> 1-step; sampler-matched
            # velocity -> multi-step). Enforce BOTH so the generator can be
            # equivariant at 1-NFE AND under multi-step sampling. The velocity
            # part uses seal_state (interpolant) + sampler_matched time.
            with _sdpa_jvp_safe_ctx():
                fn_e, z_e = _build_seal("generator_endpoint")
                sl_e = delta_seal_vector(fn_e, z_e, gens_use, gens_out=gens_use,
                                         take_mean=True, normalize=self.normalize_per_generator)
                fn_v, z_v = _build_seal("velocity_sampled")
                sl_v = delta_seal_vector(fn_v, z_v, gens_use, gens_out=gens_use,
                                         take_mean=True, normalize=self.normalize_per_generator)
            seal_loss = self._combined_w_endpoint * sl_e + self._combined_w_velocity * sl_v
            seal_parts = {"seal_endpoint": float(sl_e.detach()), "seal_velocity": float(sl_v.detach())}
        else:
            seal_fn, z = _build_seal(self.seal_target)
            with _sdpa_jvp_safe_ctx():
                seal_loss = delta_seal_vector(
                    seal_fn, z, gens_use, gens_out=gens_use,
                    take_mean=True, normalize=self.normalize_per_generator,
                )

        # Pull the current lambda as a plain Python float so it doesn't
        # accumulate into the autograd graph; we apply gradients to model
        # parameters only, the multiplier is updated by dual ascent below.
        cur_lambda = float(self.seal_lambda.detach().item())
        total = mf_loss + cur_lambda * seal_loss

        if self.adaptive_lambda:
            with torch.no_grad():
                # The loss module's buffers (seal_lambda/seal_ema/_step_count)
                # are not moved to the GPU by weaver (it never calls .to(dev)
                # on the loss). Migrate them to the loss tensor's device on
                # first use so the in-place EMA update doesn't mix cpu+cuda.
                if self.seal_ema.device != seal_loss.device:
                    self.seal_ema = self.seal_ema.to(seal_loss.device)
                    self.seal_lambda = self.seal_lambda.to(seal_loss.device)
                    self._step_count = self._step_count.to(seal_loss.device)
                seal_val = seal_loss.detach()
                # EMA of the SEAL violation
                self._step_count.add_(1)
                if int(self._step_count.item()) == 1:
                    # initialise EMA to first value to avoid a long warmup
                    self.seal_ema.copy_(seal_val)
                else:
                    self.seal_ema.mul_(self.seal_ema_alpha).add_(
                        (1.0 - self.seal_ema_alpha) * seal_val
                    )
                # Dual ascent on the multiplier: grow when constraint
                # violated (EMA > target), shrink when satisfied. Project to
                # [0, lambda_max].
                update = self.lambda_lr * (self.seal_ema - self.target_violation)
                new_lambda = (self.seal_lambda + update).clamp_(0.0, self.lambda_max)
                self.seal_lambda.copy_(new_lambda)
                info["seal_ema"] = float(self.seal_ema.item())

        info["seal"] = float(seal_loss.detach())
        info["seal_n_gens"] = int(gens_use.shape[0])
        info.update(seal_parts)
        info["seal_lambda"] = cur_lambda

        # --- PIDM-style residual on the 1-NFE generation -----------------
        # The residual always acts on the 1-NFE generation f(z') = z' - u(z',1,0)
        # (the actual one-shot sample), independent of where SEAL is enforced.
        # Fresh noise z' (independent of the SEAL z) so the residual sees an
        # unbiased sample. The model is differentiated through, so the residual
        # gradient flows back into the velocity head. One extra forward/step.
        if self.residual_func is not None and self.lam_residual > 0:
            z_res = torch.randn_like(data)
            K = self.residual_sample_steps
            with _sdpa_jvp_safe_ctx():
                if K <= 1:
                    # 1-NFE generation -- the operating point the residual was
                    # historically enforced at. Works at 1-step but does NOT
                    # transfer to multi-step sampling (state mismatch).
                    ones_r = torch.ones(data.shape[0], device=device, dtype=data.dtype)
                    zeros_r = torch.zeros(data.shape[0], device=device, dtype=data.dtype)
                    x_hat = z_res - model(z_res, ones_r, zeros_r, *cond)
                else:
                    # Sampler-matched: run a K-step Euler rollout WITH grad so
                    # the residual constrains the actual multi-step sample. Same
                    # time grid as euler_sample (1 -> 0 over K+1 points).
                    t_grid = torch.linspace(1.0, 0.0, K + 1, device=device, dtype=data.dtype)
                    x_hat = z_res
                    for i in range(K):
                        ti = torch.full((data.shape[0],), float(t_grid[i]), device=device, dtype=data.dtype)
                        ri = torch.full((data.shape[0],), float(t_grid[i + 1]), device=device, dtype=data.dtype)
                        v = model(x_hat, ti, ri, *cond)
                        x_hat = x_hat - _expand_like(ti - ri, x_hat) * v
            # Pass the real batch through too. 1-point residuals (JetResiduals)
            # ignore it via **kwargs; 2-point residuals (PairwiseResiduals)
            # match the generated pairwise structure against `real`.
            res_loss, res_info = self.residual_func(x_hat, mask, cond, real=data)
            total = total + self.lam_residual * res_loss
            info["residual"] = float(res_loss.detach())
            info["lam_residual"] = float(self.lam_residual)
            for k, v in res_info.items():
                info[f"res_{k}"] = v

        info["loss_total"] = float(total.detach())
        return total, info


@torch.no_grad()
def euler_sample(model, shape, device, sample_steps=10, cond=()):
    """Minimal Euler-style sampling loop: z_{t-dt} = z_t - (t - r) * model(z_t, t, r, cond).

    Time grid runs from 1 -> 0 over (sample_steps + 1) points. Returns the
    final sample z_0.
    """
    z = torch.randn(shape, device=device)
    t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
    for i in range(sample_steps):
        t = torch.full((shape[0],), t_vals[i].item(), device=device)
        r = torch.full((shape[0],), t_vals[i + 1].item(), device=device)
        v = model(z, t, r, *cond)
        z = z - _expand_like(t - r, z) * v
    return z
