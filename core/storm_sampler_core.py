# storm_sampler_core.py
# STORM -- Stabilized Taylor Oscillation with Runge-Kutta Memory
# Adaptive hybrid solver: STORK4 (stiff) + DPM++3M (stable), per-step dispatch
#
# © 2026 Alexander Allan (MDMAchine) | A&E Concepts
# GPL v3 -- Public version. Gradient norm stiffness detection only.
# Enterprise stiffness extensions are not included in this GPL build.
#
# Version: 2.1.1
# Created: 2026-05-08 | Revised: 2026-06-20
# Compatible: ACE-Step XL Turbo, ACE-Step SFT, flow-matching ODEs, any step count
#
# Changelog:
#   2.1.1 -- BUGFIX: sub_step_stork v_cache mutation during recursion.
#             Micro-step velocities were appended to the global deque by reference,
#             corrupting RK3/4/5 coefficient math on subsequent macro-steps.
#             Fix: sub-stepper now operates on a cloned cache. Only the main loop
#             appends the final velocity to the real v_cache.
#   2.1.0 -- EMA-weighted velocity for aligned restarts
#             Batch-invariant norms (flatten+per-element, not global torch.norm)
#             Sub-depth tracking in profiler (sub_depth per step)
#             In-place lerp_ for look-back (zero allocation, wrapper-level)
#   2.0.0 -- RK2/3/4/5 adaptive multi-order (single NFE, cached derivatives)
#             Adaptive sub-stepping (manifold fracture defense, HPC-style)
#             SDE restarts: isotropic + velocity-aligned Langevin noise
#             Adaptive calibration steps (CALIB_FRAC of total steps)
#             Profiler support: x_mag/v_mag/cos_sim/rk_order per step
#   1.2.0 -- Look-Back SNR smoother. Validated: lambda=0.35, snr_power=1.5
#   1.1.0 -- Correct flow-matching velocity, curvature damping, deque cache
#   1.0.0 -- Initial release

import torch
import torch.nn.functional as F
from collections import deque


# ─────────────────────────────────────────────
# LOOK-BACK SMOOTHER (arXiv:2602.09449)
# ─────────────────────────────────────────────

def look_back_smooth(x_curr, x_prev, sigma_curr, sigma_max, lambda_base=0.35, snr_power=1.5):
    """
    SNR-adaptive EMA smoothing of latent trajectory.
    λ(σ) = lambda_base * (σ / σ_max)^snr_power
    Heavy at high sigma (shearing zone). Zero at sigma=0 (detail preserved).

    Validated:
        35-step simple:       lambda=0.35, snr_power=1.5
        25-step ddim_uniform: lambda=0.55, snr_power=1.3
    """
    if x_prev is None:
        return x_curr, 0.0
    sigma_ratio = min(sigma_curr / max(sigma_max, 1e-8), 1.0)
    lam         = lambda_base * (sigma_ratio ** snr_power)
    return (1.0 - lam) * x_curr + lam * x_prev, lam


# ─────────────────────────────────────────────
# STIFFNESS DETECTION
# ─────────────────────────────────────────────

def compute_stiffness(v_curr, v_cache, step_idx, baseline, threshold=0.15,
                      ema_alpha=0.3, n_calib=4):
    """
    Auto-calibrating stiffness detector with EMA smoothing.
    Returns (stiff, baseline, cos_sim_mean).
    cos_sim_mean: velocity trajectory curvature metric (None if no history).
    """
    if len(v_cache) < 1:
        return True, baseline, None

    v_prev, _ = v_cache[-1]
    norm_delta = torch.norm(v_curr - v_prev).item()
    norm_curr  = torch.norm(v_curr).item() + 1e-8
    raw_ratio  = norm_delta / norm_curr

    prev_ema        = baseline.get("ema", raw_ratio)
    smoothed        = ema_alpha * raw_ratio + (1.0 - ema_alpha) * prev_ema
    baseline["ema"] = smoothed

    # Cosine similarity -- velocity curvature metric surfaced for telemetry
    v_c_flat     = v_curr.flatten(1)
    v_p_flat     = v_prev.flatten(1)
    cos_sim_mean = F.cosine_similarity(v_c_flat, v_p_flat, dim=1).mean().item()

    if step_idx < n_calib:
        baseline["sum"]        = baseline.get("sum", 0.0) + smoothed
        baseline["count"]      = baseline.get("count", 0) + 1
        baseline["last_ratio"] = smoothed
        return True, baseline, cos_sim_mean

    baseline_mean      = baseline["sum"] / max(baseline["count"], 1)
    adaptive_threshold = threshold * (baseline_mean / 0.15)
    adaptive_threshold = max(0.05, min(0.50, adaptive_threshold))

    stiff = smoothed > adaptive_threshold
    baseline["last_ratio"]     = smoothed
    baseline["last_threshold"] = adaptive_threshold
    return stiff, baseline, cos_sim_mean


# ─────────────────────────────────────────────
# STORK MULTI-ORDER (RK2/3/4/5)
# Single NFE per step -- cached velocities as virtual NFEs
# ─────────────────────────────────────────────

def stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, order="auto"):
    """
    Single-NFE stabilized RK step with adaptive order selection.
    All orders use 1 actual model call -- cached velocities act as virtual NFEs.
    Curvature damping (cosine similarity) applied to all orders.

    order="auto": highest order cache supports (RK2→3→4→5 as cache fills)
    order=2/3/4/5: force specific order
    """
    dt     = sigma_next - sigma_curr
    v_curr = model_fn(x, sigma_curr)
    n_cache = len(v_cache)

    if order == "auto":
        actual_order = min(n_cache + 1, 5) if n_cache >= 1 else 1
    else:
        actual_order = min(int(order), n_cache + 1) if n_cache >= 1 else 1
    actual_order = max(actual_order, 1)

    if n_cache < 1 or actual_order <= 1:
        return x + dt * v_curr, v_curr, 1

    v_prev_0, sigma_prev_0 = v_cache[-1]
    v_c_flat = v_curr.flatten(1)
    v_p_flat = v_prev_0.flatten(1)
    cos_sim  = F.cosine_similarity(v_c_flat, v_p_flat, dim=1)
    damping  = torch.clamp(cos_sim, 0.0, 1.0).view([-1] + [1] * (v_curr.ndim - 1))

    denom_base = sigma_curr - sigma_prev_0
    if abs(denom_base) < 1e-8:
        return x + dt * v_curr, v_curr, 2

    alpha = (sigma_next - sigma_curr) / denom_base

    if actual_order == 2:
        v_extrap = v_curr + (alpha * damping) * (v_curr - v_prev_0)
        x_next   = x + dt * (0.5 * v_curr + 0.5 * v_extrap)

    elif actual_order == 3 and n_cache >= 2:
        v1, s1 = v_cache[-1]; v2, s2 = v_cache[-2]
        h, h1  = sigma_curr - s1, s1 - s2
        if abs(h) < 1e-8 or abs(h1) < 1e-8:
            v_extrap = v_curr + (alpha * damping) * (v_curr - v1)
            x_next   = x + dt * (0.5 * v_curr + 0.5 * v_extrap)
            actual_order = 2
        else:
            c0 = 1.0 + (dt/(2.0*h)) + (dt**2/(3.0*h*h1))
            c1 = -(dt/(2.0*h)) * (1.0 + dt/h1)
            c2 = (dt**2)/(3.0*h*h1)
            v_pred = c0*v_curr + c1*v1 + c2*v2
            x_next = x + dt * (v_curr + damping * (v_pred - v_curr))

    elif actual_order == 4 and n_cache >= 3:
        v1,s1=v_cache[-1]; v2,s2=v_cache[-2]; v3,s3=v_cache[-3]
        h,h1,h2 = sigma_curr-s1, s1-s2, s2-s3
        if abs(h)<1e-8 or abs(h1)<1e-8 or abs(h2)<1e-8:
            c0=1.0+(dt/(2.0*h))+(dt**2/(3.0*h*h1))
            c1=-(dt/(2.0*h))*(1.0+dt/h1)
            c2=(dt**2)/(3.0*h*h1)
            v_pred=c0*v_curr+c1*v1+c2*v2
            x_next=x+dt*(v_curr+damping*(v_pred-v_curr))
            actual_order=3
        else:
            c0=(1.0+(dt/(2.0*h))+(dt**2/(3.0*h*h1))+(dt**3/(4.0*h*h1*h2)))
            c1=(-(dt/(2.0*h))*(1.0+dt/h1+dt**2/(2.0*h1*h2)))
            c2=((dt**2)/(3.0*h*h1))*(1.0+dt/(2.0*h2))
            c3=-(dt**3)/(4.0*h*h1*h2)
            v_pred=c0*v_curr+c1*v1+c2*v2+c3*v3
            x_next=x+dt*(v_curr+damping*(v_pred-v_curr))

    elif actual_order >= 5 and n_cache >= 4:
        v1,s1=v_cache[-1]; v2,s2=v_cache[-2]; v3,s3=v_cache[-3]; v4,s4=v_cache[-4]
        h,h1,h2,h3 = sigma_curr-s1, s1-s2, s2-s3, s3-s4
        if abs(h)<1e-8 or abs(h1)<1e-8 or abs(h2)<1e-8 or abs(h3)<1e-8:
            c0=(1.0+dt/(2.0*h)+dt**2/(3.0*h*h1)+dt**3/(4.0*h*h1*h2))
            c1=-(dt/(2.0*h))*(1.0+dt/h1+dt**2/(2.0*h1*h2))
            c2=(dt**2/(3.0*h*h1))*(1.0+dt/(2.0*h2))
            c3=-(dt**3)/(4.0*h*h1*h2)
            v_pred=c0*v_curr+c1*v1+c2*v2+c3*v3
            x_next=x+dt*(v_curr+damping*(v_pred-v_curr))
            actual_order=4
        else:
            c0=(1.0+dt/(2.0*h)+dt**2/(3.0*h*h1)+dt**3/(4.0*h*h1*h2)+dt**4/(5.0*h*h1*h2*h3))
            c1=-(dt/(2.0*h))*(1.0+dt/h1+dt**2/(2.0*h1*h2)+dt**3/(3.0*h1*h2*h3))
            c2=(dt**2/(3.0*h*h1))*(1.0+dt/(2.0*h2)+dt**2/(3.0*h2*h3))
            c3=-(dt**3/(4.0*h*h1*h2))*(1.0+dt/(2.0*h3))
            c4=dt**4/(5.0*h*h1*h2*h3)
            v_pred=c0*v_curr+c1*v1+c2*v2+c3*v3+c4*v4
            x_next=x+dt*(v_curr+damping*(v_pred-v_curr))
            actual_order=5
    else:
        v_extrap = v_curr + (alpha * damping) * (v_curr - v_prev_0)
        x_next   = x + dt * (0.5 * v_curr + 0.5 * v_extrap)
        actual_order = 2

    return x_next, v_curr, actual_order


# ─────────────────────────────────────────────
# ADAPTIVE SUB-STEPPING (Manifold Fracture Defense)
# ─────────────────────────────────────────────

def sub_step_stork(v_cache, x, sigma_curr, sigma_next, model_fn,
                   rk_order="auto", threshold=0.0, depth=0, max_depth=2):
    """
    Recursive sub-stepper. U-turn detected (cos_sim < threshold) -> split step in half.
    Inspired by HPC adaptive timestep rejection in fluid dynamics.
    max_depth prevents infinite recursion.

    CRITICAL: operates on a CLONED cache so micro-step velocities do not
    pollute the global v_cache used by macro-step RK3/4/5 coefficient math.
    Only the caller (main loop) appends the final velocity to the real cache.
    """
    if depth >= max_depth:
        x_out, v_curr, order = stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, rk_order)
        return x_out, v_curr, order, depth

    if len(v_cache) >= 1:
        v_probe   = model_fn(x, sigma_curr)
        v_prev, _ = v_cache[-1]
        cos_sim   = F.cosine_similarity(
            v_probe.flatten(1), v_prev.flatten(1), dim=1).mean().item()

        if cos_sim < threshold:
            sigma_mid = (sigma_curr + sigma_next) / 2.0
            # Clone cache for sub-step isolation — micro-steps must not
            # corrupt macro-step velocity history (RK coefficient math
            # assumes consistent macro-step intervals in the cache)
            local_cache = deque(v_cache, maxlen=v_cache.maxlen)
            x_mid, v_mid, o1, _ = sub_step_stork(
                local_cache, x, sigma_curr, sigma_mid, model_fn,
                rk_order, threshold, depth + 1, max_depth)
            local_cache.append((v_mid, sigma_curr))
            x_out, v_out, o2, _ = sub_step_stork(
                local_cache, x_mid, sigma_mid, sigma_next, model_fn,
                rk_order, threshold, depth + 1, max_depth)
            return x_out, v_out, max(o1, o2), depth + 1

    x_out, v_curr, order = stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, rk_order)
    return x_out, v_curr, order, depth


# ─────────────────────────────────────────────
# DPM++3M -- smooth schedule path
# ─────────────────────────────────────────────

def dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, model_fn):
    dt     = sigma_next - sigma_curr
    v_curr = model_fn(x, sigma_curr)

    if len(v_cache) >= 2:
        v1,s1=v_cache[-1]; v2,s2=v_cache[-2]
        h,h1=sigma_curr-s1, s1-s2
        if abs(h)<1e-8 or abs(h1)<1e-8:
            x_next=x+dt*v_curr
        else:
            cc=1.0+(dt/(2.0*h))+(dt**2/(3.0*h*h1))
            c1=-(dt/(2.0*h))*(1.0+dt/h1)
            c2=(dt**2)/(3.0*h*h1)
            x_next=x+dt*(cc*v_curr+c1*v1+c2*v2)
    elif len(v_cache) >= 1:
        v1,s1=v_cache[-1]
        h=sigma_curr-s1
        x_next=x+dt*v_curr if abs(h)<1e-8 else x+dt*(v_curr+(dt/(2.0*h))*(v_curr-v1))
    else:
        x_next=x+dt*v_curr

    return x_next, v_curr


# ─────────────────────────────────────────────
# STORM SAMPLER -- Full inference loop
# ─────────────────────────────────────────────

def storm_sampler(
    model_fn,
    x,
    sigmas,
    stiffness_threshold=0.15,
    hysteresis_margin=0.05,
    ema_alpha=0.3,
    cache_depth=5,
    rk_order="auto",
    calib_frac=0.12,
    adaptive_sub_step=True,
    sub_step_threshold=0.0,
    sub_step_max_depth=2,
    look_back_enabled=True,
    look_back_lambda=0.35,
    look_back_snr_power=1.5,
    enable_restarts=False,
    restart_steps=None,
    restart_noise_scale=0.5,
    restart_s_noise=1.0,
    restart_seed=42,
    restart_flush_cache=True,
    restart_aligned_noise=True,
    force_pure_euler=False,
    verbose=False,
    extra_args=None,
    callback=None,
    profiler=None,
):
    """
    Full STORM denoising loop.

    Flow-matching velocity: v = (x - denoised) / sigma
    RK order auto-escalates as cache fills: RK1→2→3→4→5
    Look-Back: SNR-adaptive EMA, heavy at high sigma, zero at zero
    Restarts: isotropic or velocity-aligned Langevin noise injection
    Sub-stepping: splits step on U-turn detection (cos_sim < threshold)
    """
    if extra_args is None:
        extra_args = {}
    if restart_steps is None:
        restart_steps = set()

    def _model(x_in, sigma_in):
        if not isinstance(sigma_in, torch.Tensor):
            sigma_in = torch.tensor([sigma_in], dtype=x_in.dtype, device=x_in.device)
        elif sigma_in.ndim == 0:
            sigma_in = sigma_in.unsqueeze(0)
        denoised     = model_fn(x_in, sigma_in, **extra_args)
        sigma_scalar = sigma_in.item() if sigma_in.numel() == 1 else sigma_in[0].item()
        return (x_in - denoised) / max(sigma_scalar, 1e-7)

    v_cache   = deque(maxlen=cache_depth)
    baseline  = {"sum": 0.0, "count": 0}
    n_steps   = len(sigmas) - 1
    sigma_max = sigmas[0].item()
    n_calib   = max(2, min(5, int(n_steps * calib_frac)))

    # x_prev_lb: initial look-back reference. Step-0 noise injection (10% of sigma_max)
    # gives mild diversity at the first blend vs a pure clone. Intentional.
    x_prev_lb = (x + torch.randn_like(x) * sigma_max * 0.1) if look_back_enabled else None

    import time as _time

    for i in range(n_steps):
        sigma_curr_t = sigmas[i]
        sigma_next_t = sigmas[i + 1]
        sigma_curr   = sigma_curr_t.item()
        sigma_next   = sigma_next_t.item()

        if sigma_next == 0.0:
            v_final = _model(x, sigma_curr_t)
            x       = x + (sigma_next - sigma_curr) * v_final
            if verbose:
                print(f"[STORM] Step {i:02d}: FINAL (Euler terminal)")
            if profiler is not None:
                profiler.record(i, sigma_curr, sigma_next, "FINAL", 1, 0.0, 0.0, None, 0.0)
            break

        x_prev_lb_before = x.clone() if look_back_enabled else None
        _t0 = _time.time()

        # ── STORM STEP ──
        if force_pure_euler:
            v_curr = _model(x, sigma_curr_t)
            x      = x + (sigma_next - sigma_curr) * v_curr
            v_cache.append((v_curr, sigma_curr))
            mode, actual_order, cos_sim = "EULER", 1, None
        else:
            # Stiffness detection
            if len(v_cache) >= 1:
                v_probe = _model(x, sigma_curr_t)
                stiff, baseline, cos_sim = compute_stiffness(
                    v_probe, v_cache, i, baseline,
                    stiffness_threshold, ema_alpha, n_calib)
                v_precomp = v_probe
            else:
                stiff, v_precomp, cos_sim = True, None, None

            prev_mode = baseline.get("prev_mode", "STORK")
            if prev_mode == "DPM++" and not stiff:
                if (baseline.get("last_ratio", 0) >
                        baseline.get("last_threshold", stiffness_threshold) + hysteresis_margin):
                    stiff = True

            def _model_cached(x_in, sigma_in):
                s = sigma_in.item() if isinstance(sigma_in, torch.Tensor) else sigma_in
                if v_precomp is not None and abs(s - sigma_curr) < 1e-7:
                    return v_precomp
                return _model(x_in, torch.tensor([s], dtype=x_in.dtype, device=x_in.device))

            if stiff:
                if adaptive_sub_step and len(v_cache) >= 1:
                    x, v_curr, actual_order, sub_depth = sub_step_stork(
                        v_cache, x, sigma_curr, sigma_next, _model_cached,
                        rk_order, sub_step_threshold, 0, sub_step_max_depth)
                else:
                    x, v_curr, actual_order = stork_step(
                        v_cache, x, sigma_curr, sigma_next, _model_cached, rk_order)
                mode = "STORK"
            else:
                x, v_curr = dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, _model_cached)
                mode, actual_order = "DPM++", 3

            if verbose:
                lr  = baseline.get("last_ratio", 0.0)
                lt  = baseline.get("last_threshold", stiffness_threshold)
                cs  = f"{cos_sim:.4f}" if cos_sim is not None else "N/A"
                spk = " -> CURVATURE SPIKE" if stiff and prev_mode == "DPM++" else ""
                print(f"[STORM] Step {i:02d}: {mode:5s} RK{actual_order} | "
                      f"Ratio: {lr:.3f} | Threshold: {lt:.3f} | cos_sim: {cs}{spk}")

            # NaN guard
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[STORM] NaN/Inf at step {i}. Flushing cache.")
                v_curr = _model(x, sigma_curr_t)
                x      = x + (sigma_next - sigma_curr) * v_curr
                v_cache.clear()
                baseline["prev_mode"] = "STORK"
                actual_order = 1

            v_cache.append((v_curr, sigma_curr))
            baseline["prev_mode"] = mode

        # ── LOOK-BACK ──
        lam = 0.0
        if look_back_enabled and x_prev_lb is not None:
            # In-place lerp -- saves [B,C,T] allocation vs look_back_smooth return
            sigma_ratio = min(sigma_curr / max(sigma_max, 1e-8), 1.0)
            lam         = look_back_lambda * (sigma_ratio ** look_back_snr_power)
            x.lerp_(x_prev_lb, lam)
            if verbose:
                print(f"[STORM] LookBack λ={lam:.4f} @ σ={sigma_curr:.3f}")
        x_prev_lb = x_prev_lb_before

        # ── RESTARTS ──
        if enable_restarts and i in restart_steps and sigma_next > 0:
            s_res = sigma_next + (sigma_curr - sigma_next) * restart_noise_scale
            n_amt = (max(0.0, s_res**2 - sigma_next**2) + 1e-8) ** 0.5
            gen   = torch.Generator(device=x.device).manual_seed(
                (restart_seed + i * 1000) % (2**63 - 1))

            if restart_aligned_noise and len(v_cache) >= 2:
                v_stack = torch.stack([v for v, _ in list(v_cache)], dim=0)
                # EMA-weighted velocity: recent vectors get highest weight
                _depth   = v_stack.shape[0]
                _weights = torch.tensor(
                    [0.5 ** (_depth - 1 - i) for i in range(_depth)],
                    dtype=v_stack.dtype, device=v_stack.device
                ).view(-1, *([1] * (v_stack.ndim - 1)))
                v_dir    = (v_stack * _weights).sum(dim=0) / _weights.sum()
                v_dir    = v_dir / (torch.norm(v_dir) + 1e-8)
                raw     = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
                rdims   = list(range(1, raw.ndim))
                proj    = (raw * v_dir).sum(dim=rdims, keepdim=True)
                noise   = raw - proj * v_dir
                noise   = noise * (torch.norm(raw) / (torch.norm(noise) + 1e-8))
                if verbose:
                    print(f"[STORM] ♻️  ALIGNED RESTART @ step {i} (noise ⊥ v_principal)")
            else:
                noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
                if verbose:
                    print(f"[STORM] ♻️  RESTART @ step {i}")

            x_renoise   = x + noise * n_amt * restart_s_noise
            sigma_res_t = torch.tensor([s_res], dtype=x.dtype, device=x.device)
            v_res       = _model(x_renoise, sigma_res_t)
            x           = x_renoise + (sigma_next - s_res) * v_res

            if restart_flush_cache:
                v_cache.clear()
                baseline["prev_mode"] = "STORK"

        # ── PROFILER ──
        if profiler is not None:
            # Batch-invariant norms -- prevents attractor scale blowout at batch>1
            _x_mag = torch.norm(x.flatten(1), dim=1).mean().item()
            _v_mag = torch.norm(v_cache[-1][0].flatten(1), dim=1).mean().item() if len(v_cache) > 0 else 0.0
            profiler.record(
                i, sigma_curr, sigma_next, mode, actual_order, lam,
                baseline.get("last_ratio", 0.0), cos_sim,
                _time.time() - _t0,
                x_mag=_x_mag, v_mag=_v_mag,
                sub_depth=sub_depth if 'sub_depth' in locals() else 0,
            )

        if callback is not None:
            try:
                callback({"x": x, "i": i, "sigma": sigmas[i + 1],
                          "sigma_hat": sigmas[i + 1], "denoised": x})
            except TypeError:
                callback(i, x, sigmas[i + 1])

    return x
