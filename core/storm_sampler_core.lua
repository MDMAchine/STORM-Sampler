--[[
storm_sampler_core.lua
STORM -- Stabilized Taylor Oscillation with Runge-Kutta Memory
Adaptive hybrid solver: STORK (stiff) + DPM++3M (stable), per-step dispatch

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3 -- Public version. Gradient norm stiffness detection only.

Version: 2.1.1
PARITY: Exact algorithmic parity with storm_sampler_core.py v2.1.1

Changelog:
    2.1.1 -- BUGFIX: sub_step_stork v_cache mutation during recursion.
             Micro-step velocities corrupted macro-step RK3/4/5 coefficients.
             Fix: sub-stepper operates on cloned cache.
    2.1.0 -- EMA-weighted velocity for aligned restarts
    2.0.0 -- RK2/3/4/5 adaptive multi-order (single NFE, cached derivatives)
             Adaptive sub-stepping (manifold fracture defense, HPC-style)
             SDE restarts: isotropic + velocity-aligned Langevin noise
             Adaptive calibration steps (CALIB_FRAC of total steps)
             Batch-safe reduce_dims projection in velocity-aligned restart
    1.2.0 -- Look-Back SNR smoother. Validated: lambda=0.35, snr_power=1.5
    1.0.0 -- Initial release

Validated configs:
    35-step simple:       look_back_lambda=0.35, look_back_snr_power=1.5
    25-step ddim_uniform: look_back_lambda=0.55, look_back_snr_power=1.3

TENSOR CONVENTION:
    Flat float arrays [1..n] with explicit size field n.
    model_fn(x, sigma, n) -> v  (velocity, not denoised)
--]]

local storm = {}

-- ─────────────────────────────────────────────────────────────────────────────
-- MATH HELPERS
-- ─────────────────────────────────────────────────────────────────────────────

local function vec_norm(v, n)
    local s = 0.0
    for i = 1, n do s = s + v[i]*v[i] end
    return math.sqrt(s)
end

local function vec_sub_norm(a, b, n)
    local s = 0.0
    for i = 1, n do local d=a[i]-b[i]; s=s+d*d end
    return math.sqrt(s)
end

local function vec_dot(a, b, n)
    local s = 0.0
    for i = 1, n do s = s + a[i]*b[i] end
    return s
end

local function vec_clone(v, n)
    local c = {}
    for i = 1, n do c[i] = v[i] end
    return c
end

local function vec_add_scaled(a, b, scale, n)
    local out = {}
    for i = 1, n do out[i] = a[i] + scale * b[i] end
    return out
end

local function has_nan_inf(v, n)
    for i = 1, n do
        if v[i] ~= v[i] or math.abs(v[i]) == math.huge then return true end
    end
    return false
end

local function clamp(x, lo, hi) return math.max(lo, math.min(hi, x)) end

local function randn_like(v, n, scale)
    local out = {}
    for i = 1, n, 2 do
        local u1 = math.max(1e-12, math.random())
        local u2 = math.random()
        local r   = scale * math.sqrt(-2.0 * math.log(u1))
        out[i]    = v[i] + r * math.cos(2 * math.pi * u2)
        if i+1 <= n then
            out[i+1] = v[i+1] + r * math.sin(2 * math.pi * u2)
        end
    end
    return out
end

local function randn_iso(n, scale)
    -- Standard isotropic noise (no base offset)
    local out = {}
    for i = 1, n, 2 do
        local u1 = math.max(1e-12, math.random())
        local u2 = math.random()
        local r   = scale * math.sqrt(-2.0 * math.log(u1))
        out[i]    = r * math.cos(2 * math.pi * u2)
        if i+1 <= n then
            out[i+1] = r * math.sin(2 * math.pi * u2)
        end
    end
    -- pad if n is odd
    if n % 2 == 1 then out[n] = out[n] or 0.0 end
    return out
end


-- ─────────────────────────────────────────────────────────────────────────────
-- LOOK-BACK SMOOTHER (arXiv:2602.09449)
-- Parity: storm_sampler_core.py::look_back_smooth
-- ─────────────────────────────────────────────────────────────────────────────

function storm.look_back_smooth(x_curr, x_prev, sigma_curr, sigma_max, lambda_base, snr_power, n)
    lambda_base = lambda_base or 0.35
    snr_power   = snr_power   or 1.5
    if x_prev == nil then return x_curr, 0.0 end
    local ratio  = math.min(sigma_curr / math.max(sigma_max, 1e-8), 1.0)
    local lam    = lambda_base * (ratio ^ snr_power)
    local out    = {}
    for i = 1, n do out[i] = (1.0-lam)*x_curr[i] + lam*x_prev[i] end
    return out, lam
end


-- ─────────────────────────────────────────────────────────────────────────────
-- STIFFNESS DETECTION (adaptive calib)
-- Parity: storm_sampler_core.py::compute_stiffness
-- ─────────────────────────────────────────────────────────────────────────────

function storm.compute_stiffness(v_curr, v_cache, step_idx, baseline, threshold, ema_alpha, n_calib, n)
    threshold = threshold or 0.15
    ema_alpha = ema_alpha or 0.3
    n_calib   = n_calib   or 4

    if #v_cache < 1 then return true, baseline, nil end

    local v_prev     = v_cache[#v_cache].v
    local norm_delta = vec_sub_norm(v_curr, v_prev, n)
    local norm_curr  = vec_norm(v_curr, n) + 1e-8
    local raw_ratio  = norm_delta / norm_curr

    local prev_ema   = baseline.ema or raw_ratio
    local smoothed   = ema_alpha * raw_ratio + (1.0 - ema_alpha) * prev_ema
    baseline.ema     = smoothed

    -- Cosine similarity -- curvature metric
    local dot      = vec_dot(v_curr, v_prev, n)
    local nc       = vec_norm(v_curr, n)
    local np_      = vec_norm(v_prev, n)
    local cos_sim  = dot / (nc * np_ + 1e-8)

    if step_idx < n_calib then
        baseline.sum        = (baseline.sum   or 0.0) + smoothed
        baseline.count      = (baseline.count or 0)   + 1
        baseline.last_ratio = smoothed
        return true, baseline, cos_sim
    end

    local bmean     = baseline.sum / math.max(baseline.count, 1)
    local adap_thr  = threshold * (bmean / 0.15)
    adap_thr        = clamp(adap_thr, 0.05, 0.50)

    local stiff = smoothed > adap_thr
    baseline.last_ratio     = smoothed
    baseline.last_threshold = adap_thr
    return stiff, baseline, cos_sim
end


-- ─────────────────────────────────────────────────────────────────────────────
-- STORK MULTI-ORDER (RK2/3/4/5 via cached derivatives -- single NFE)
-- Parity: storm_sampler_core.py::stork_step
-- ─────────────────────────────────────────────────────────────────────────────

function storm.stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, order, n)
    local dt      = sigma_next - sigma_curr
    local v_curr  = model_fn(x, sigma_curr, n)
    local n_cache = #v_cache

    -- Determine actual order
    local actual_order
    if order == "auto" then
        actual_order = (n_cache >= 1) and math.min(n_cache+1, 5) or 1
    else
        actual_order = (n_cache >= 1) and math.min(tonumber(order), n_cache+1) or 1
    end
    actual_order = math.max(actual_order, 1)

    if n_cache < 1 or actual_order <= 1 then
        local x_next = {}
        for i=1,n do x_next[i] = x[i] + dt*v_curr[i] end
        return x_next, v_curr, 1
    end

    local e0         = v_cache[#v_cache]
    local v_prev_0   = e0.v
    local sigma_prev = e0.sigma

    -- Curvature damping
    local dot     = vec_dot(v_curr, v_prev_0, n)
    local nc      = vec_norm(v_curr, n)
    local np_     = vec_norm(v_prev_0, n)
    local cos_sim = dot / (nc * np_ + 1e-8)
    local damping = clamp(cos_sim, 0.0, 1.0)  -- scalar, applied uniformly

    local denom = sigma_curr - sigma_prev
    if math.abs(denom) < 1e-8 then
        local x_next = {}
        for i=1,n do x_next[i] = x[i] + dt*v_curr[i] end
        return x_next, v_curr, 2
    end
    local alpha = (sigma_next - sigma_curr) / denom

    local x_next = {}

    if actual_order == 2 then
        -- AB2
        for i=1,n do
            local v_extrap = v_curr[i] + (alpha * damping) * (v_curr[i] - v_prev_0[i])
            x_next[i]     = x[i] + dt * (0.5*v_curr[i] + 0.5*v_extrap)
        end

    elseif actual_order == 3 and n_cache >= 2 then
        -- AB3 variable-step
        local v1,s1 = v_cache[#v_cache].v, v_cache[#v_cache].sigma
        local v2,s2 = v_cache[#v_cache-1].v, v_cache[#v_cache-1].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        if math.abs(h)<1e-8 or math.abs(h1)<1e-8 then
            for i=1,n do
                local ve = v_curr[i] + (alpha*damping)*(v_curr[i]-v1[i])
                x_next[i] = x[i] + dt*(0.5*v_curr[i]+0.5*ve)
            end
            actual_order = 2
        else
            local c0 = 1.0 + (dt/(2.0*h)) + (dt^2/(3.0*h*h1))
            local c1 = -(dt/(2.0*h)) * (1.0 + dt/h1)
            local c2 = (dt^2)/(3.0*h*h1)
            for i=1,n do
                local v_pred = c0*v_curr[i] + c1*v1[i] + c2*v2[i]
                x_next[i]   = x[i] + dt*(v_curr[i] + damping*(v_pred-v_curr[i]))
            end
        end

    elseif actual_order == 4 and n_cache >= 3 then
        -- AB4 variable-step
        local v1,s1 = v_cache[#v_cache].v,   v_cache[#v_cache].sigma
        local v2,s2 = v_cache[#v_cache-1].v, v_cache[#v_cache-1].sigma
        local v3,s3 = v_cache[#v_cache-2].v, v_cache[#v_cache-2].sigma
        local h  = sigma_curr-s1
        local h1 = s1-s2
        local h2 = s2-s3
        if math.abs(h)<1e-8 or math.abs(h1)<1e-8 or math.abs(h2)<1e-8 then
            local c0=1.0+(dt/(2.0*h))+(dt^2/(3.0*h*h1))
            local c1=-(dt/(2.0*h))*(1.0+dt/h1)
            local c2=(dt^2)/(3.0*h*h1)
            for i=1,n do
                local vp=c0*v_curr[i]+c1*v1[i]+c2*v2[i]
                x_next[i]=x[i]+dt*(v_curr[i]+damping*(vp-v_curr[i]))
            end
            actual_order=3
        else
            local c0=(1.0+(dt/(2.0*h))+(dt^2/(3.0*h*h1))+(dt^3/(4.0*h*h1*h2)))
            local c1=(-(dt/(2.0*h))*(1.0+dt/h1+dt^2/(2.0*h1*h2)))
            local c2=((dt^2)/(3.0*h*h1))*(1.0+dt/(2.0*h2))
            local c3=-(dt^3)/(4.0*h*h1*h2)
            for i=1,n do
                local vp=c0*v_curr[i]+c1*v1[i]+c2*v2[i]+c3*v3[i]
                x_next[i]=x[i]+dt*(v_curr[i]+damping*(vp-v_curr[i]))
            end
        end

    elseif actual_order >= 5 and n_cache >= 4 then
        -- AB5 variable-step
        local v1,s1=v_cache[#v_cache].v,   v_cache[#v_cache].sigma
        local v2,s2=v_cache[#v_cache-1].v, v_cache[#v_cache-1].sigma
        local v3,s3=v_cache[#v_cache-2].v, v_cache[#v_cache-2].sigma
        local v4,s4=v_cache[#v_cache-3].v, v_cache[#v_cache-3].sigma
        local h  = sigma_curr-s1
        local h1 = s1-s2
        local h2 = s2-s3
        local h3 = s3-s4
        if math.abs(h)<1e-8 or math.abs(h1)<1e-8 or math.abs(h2)<1e-8 or math.abs(h3)<1e-8 then
            local c0=(1.0+dt/(2.0*h)+dt^2/(3.0*h*h1)+dt^3/(4.0*h*h1*h2))
            local c1=-(dt/(2.0*h))*(1.0+dt/h1+dt^2/(2.0*h1*h2))
            local c2=(dt^2/(3.0*h*h1))*(1.0+dt/(2.0*h2))
            local c3=-(dt^3)/(4.0*h*h1*h2)
            for i=1,n do
                local vp=c0*v_curr[i]+c1*v1[i]+c2*v2[i]+c3*v3[i]
                x_next[i]=x[i]+dt*(v_curr[i]+damping*(vp-v_curr[i]))
            end
            actual_order=4
        else
            local c0=(1.0+dt/(2.0*h)+dt^2/(3.0*h*h1)+dt^3/(4.0*h*h1*h2)+dt^4/(5.0*h*h1*h2*h3))
            local c1=-(dt/(2.0*h))*(1.0+dt/h1+dt^2/(2.0*h1*h2)+dt^3/(3.0*h1*h2*h3))
            local c2=(dt^2/(3.0*h*h1))*(1.0+dt/(2.0*h2)+dt^2/(3.0*h2*h3))
            local c3=-(dt^3/(4.0*h*h1*h2))*(1.0+dt/(2.0*h3))
            local c4=dt^4/(5.0*h*h1*h2*h3)
            for i=1,n do
                local vp=c0*v_curr[i]+c1*v1[i]+c2*v2[i]+c3*v3[i]+c4*v4[i]
                x_next[i]=x[i]+dt*(v_curr[i]+damping*(vp-v_curr[i]))
            end
            actual_order=5
        end

    else
        -- Fallback AB2
        for i=1,n do
            local ve = v_curr[i] + (alpha*damping)*(v_curr[i]-v_prev_0[i])
            x_next[i] = x[i] + dt*(0.5*v_curr[i]+0.5*ve)
        end
        actual_order = 2
    end

    return x_next, v_curr, actual_order
end


-- ─────────────────────────────────────────────────────────────────────────────
-- ADAPTIVE SUB-STEPPING (Manifold Fracture Defense)
-- Parity: storm_sampler_core.py::sub_step_stork
-- ─────────────────────────────────────────────────────────────────────────────

function storm.sub_step_stork(v_cache, x, sigma_curr, sigma_next, model_fn, rk_order, threshold, depth, max_depth, n)
    rk_order  = rk_order  or "auto"
    threshold = threshold or 0.0
    depth     = depth     or 0
    max_depth = max_depth or 2

    if depth >= max_depth then
        local xo,vc,ord = storm.stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, rk_order, n)
        return xo, vc, ord, depth
    end

    if #v_cache >= 1 then
        local v_probe = model_fn(x, sigma_curr, n)
        local v_prev  = v_cache[#v_cache].v
        local dot     = vec_dot(v_probe, v_prev, n)
        local nc      = vec_norm(v_probe, n)
        local np_     = vec_norm(v_prev, n)
        local cos_sim = dot / (nc * np_ + 1e-8)

        if cos_sim < threshold then
            local sigma_mid = (sigma_curr + sigma_next) / 2.0
            -- Clone cache: micro-step velocities must not corrupt
            -- macro-step RK coefficient math (v2.1.1 bugfix)
            local local_cache = {}
            for i = 1, #v_cache do local_cache[i] = v_cache[i] end
            local xm, vm, o1, _ = storm.sub_step_stork(
                local_cache, x, sigma_curr, sigma_mid, model_fn, rk_order, threshold, depth+1, max_depth, n)
            table.insert(local_cache, {v=vm, sigma=sigma_curr})
            local xo, vo, o2, _ = storm.sub_step_stork(
                local_cache, xm, sigma_mid, sigma_next, model_fn, rk_order, threshold, depth+1, max_depth, n)
            return xo, vo, math.max(o1,o2), depth+1
        end
    end

    local xo, vc, ord = storm.stork_step(v_cache, x, sigma_curr, sigma_next, model_fn, rk_order, n)
    return xo, vc, ord, depth
end


-- ─────────────────────────────────────────────────────────────────────────────
-- DPM++3M -- smooth schedule path
-- Parity: storm_sampler_core.py::dpmpp3m_step
-- ─────────────────────────────────────────────────────────────────────────────

function storm.dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, model_fn, n)
    local dt     = sigma_next - sigma_curr
    local v_curr = model_fn(x, sigma_curr, n)
    local x_next = {}

    if #v_cache >= 2 then
        local v1,s1 = v_cache[#v_cache].v,   v_cache[#v_cache].sigma
        local v2,s2 = v_cache[#v_cache-1].v, v_cache[#v_cache-1].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        if math.abs(h)<1e-8 or math.abs(h1)<1e-8 then
            for i=1,n do x_next[i]=x[i]+dt*v_curr[i] end
        else
            local cc = 1.0+(dt/(2.0*h))+(dt^2/(3.0*h*h1))
            local c1 = -(dt/(2.0*h))*(1.0+dt/h1)
            local c2 = (dt^2)/(3.0*h*h1)
            for i=1,n do x_next[i]=x[i]+dt*(cc*v_curr[i]+c1*v1[i]+c2*v2[i]) end
        end
    elseif #v_cache >= 1 then
        local v1,s1 = v_cache[#v_cache].v, v_cache[#v_cache].sigma
        local h = sigma_curr - s1
        if math.abs(h)<1e-8 then
            for i=1,n do x_next[i]=x[i]+dt*v_curr[i] end
        else
            for i=1,n do x_next[i]=x[i]+dt*(v_curr[i]+(dt/(2.0*h))*(v_curr[i]-v1[i])) end
        end
    else
        for i=1,n do x_next[i]=x[i]+dt*v_curr[i] end
    end

    return x_next, v_curr
end


-- ─────────────────────────────────────────────────────────────────────────────
-- STORM STEP
-- Parity: storm_sampler_core.py (storm_sampler inner loop)
-- ─────────────────────────────────────────────────────────────────────────────

function storm.storm_step(v_cache, x, sigma_curr, sigma_next, model_fn, step_idx, baseline, opts, n)
    local thr       = opts.stiffness_threshold or 0.15
    local hyst      = opts.hysteresis_margin   or 0.05
    local ema_a     = opts.ema_alpha           or 0.3
    local depth_max = opts.cache_depth         or 5
    local verbose   = opts.verbose             or false
    local euler     = opts.force_pure_euler    or false
    local rk_order  = opts.rk_order           or "auto"
    local sub_step  = opts.adaptive_sub_step   ~= false
    local sub_thr   = opts.sub_step_threshold  or 0.0
    local sub_depth = opts.sub_step_max_depth  or 2
    local n_calib   = opts.n_calib             or 4

    if baseline == nil then baseline = {sum=0.0, count=0} end

    -- Euler bypass
    if euler then
        local v_curr = model_fn(x, sigma_curr, n)
        local x_next = {}
        for i=1,n do x_next[i]=x[i]+(sigma_next-sigma_curr)*v_curr[i] end
        if verbose then print(string.format("[STORM] Step %02d: EULER (bypass)", step_idx)) end
        table.insert(v_cache, {v=v_curr, sigma=sigma_curr})
        while #v_cache > depth_max do table.remove(v_cache, 1) end
        baseline.prev_mode = "STORK"
        return x_next, v_cache, "EULER", 1, nil, baseline
    end

    -- Stiffness detection
    local stiff, v_precomp, cos_sim_out
    if #v_cache >= 1 then
        v_precomp = model_fn(x, sigma_curr, n)
        stiff, baseline, cos_sim_out = storm.compute_stiffness(
            v_precomp, v_cache, step_idx, baseline, thr, ema_a, n_calib, n)
    else
        stiff, v_precomp, cos_sim_out = true, nil, nil
    end

    -- Hysteresis
    local prev_mode = baseline.prev_mode or "STORK"
    if prev_mode == "DPM++" and not stiff then
        if (baseline.last_ratio or 0) > (baseline.last_threshold or thr) + hyst then
            stiff = true
        end
    end

    -- Cached model wrapper
    local function model_cached(x_in, sigma_in, n_in)
        if v_precomp ~= nil and math.abs(sigma_in - sigma_curr) < 1e-7 then
            return v_precomp
        end
        return model_fn(x_in, sigma_in, n_in)
    end

    -- Dispatch
    local x_next, v_curr, actual_order, mode
    if stiff then
        if sub_step and #v_cache >= 1 then
            x_next, v_curr, actual_order, _ = storm.sub_step_stork(
                v_cache, x, sigma_curr, sigma_next, model_cached,
                rk_order, sub_thr, 0, sub_depth, n)
        else
            x_next, v_curr, actual_order = storm.stork_step(
                v_cache, x, sigma_curr, sigma_next, model_cached, rk_order, n)
        end
        mode = "STORK"
    else
        x_next, v_curr = storm.dpmpp3m_step(
            v_cache, x, sigma_curr, sigma_next, model_cached, n)
        mode         = "DPM++"
        actual_order = 3
    end

    -- Verbose
    if verbose then
        local lr  = baseline.last_ratio     or 0.0
        local lt  = baseline.last_threshold or thr
        local cs  = cos_sim_out and string.format("%.4f", cos_sim_out) or "N/A"
        local tag = (stiff and prev_mode=="DPM++") and " -> CURVATURE SPIKE" or ""
        print(string.format("[STORM] Step %02d: %-5s RK%d | Ratio: %.3f | Threshold: %.3f | cos_sim: %s%s",
            step_idx, mode, actual_order, lr, lt, cs, tag))
    end

    -- NaN guard
    if has_nan_inf(x_next, n) then
        print(string.format("[STORM] NaN/Inf at step %d. Flushing cache.", step_idx))
        local ve = model_fn(x, sigma_curr, n)
        local dt = sigma_next - sigma_curr
        x_next = {}
        for i=1,n do x_next[i]=x[i]+dt*ve[i] end
        v_curr = ve
        v_cache = {}
        baseline.prev_mode = "STORK"
        actual_order = 1
    end

    table.insert(v_cache, {v=v_curr, sigma=sigma_curr})
    while #v_cache > depth_max do table.remove(v_cache, 1) end
    baseline.prev_mode = mode

    return x_next, v_cache, mode, actual_order, cos_sim_out, baseline
end


-- ─────────────────────────────────────────────────────────────────────────────
-- STORM SAMPLER -- Full inference loop
-- Parity: storm_sampler_core.py::storm_sampler
-- ─────────────────────────────────────────────────────────────────────────────

function storm.storm_sampler(model_fn, x, sigmas, n, opts)
    opts = opts or {}
    local verbose            = opts.verbose             or false
    local callback           = opts.callback
    local look_back_enabled  = (opts.look_back_enabled ~= false)
    local look_back_lambda   = opts.look_back_lambda    or 0.35
    local look_back_snr_pow  = opts.look_back_snr_power or 1.5
    local calib_frac         = opts.calib_frac          or 0.12
    local enable_restarts    = opts.enable_restarts      or false
    local restart_steps      = opts.restart_steps        or {}  -- set/table of step indices
    local restart_noise_scale = opts.restart_noise_scale or 0.5
    local restart_s_noise    = opts.restart_s_noise      or 1.0
    local restart_seed       = opts.restart_seed         or 42
    local restart_flush_cache = (opts.restart_flush_cache ~= false)
    local restart_aligned    = opts.restart_aligned_noise or false

    local n_steps  = #sigmas - 1
    local v_cache  = {}
    local baseline = {sum=0.0, count=0}
    local sigma_max = sigmas[1]

    -- Adaptive calibration steps
    local n_calib = math.max(2, math.min(5, math.floor(n_steps * calib_frac)))
    opts.n_calib  = n_calib  -- pass through to storm_step

    if verbose then
        print(string.format("[STORM] Schedule: %d steps | Calib: %d | RK: %s | Cache: %d",
            n_steps, n_calib, tostring(opts.rk_order or "auto"), opts.cache_depth or 5))
    end

    -- Seed x_prev for step-0 look-back coverage
    local x_prev_lb = nil
    if look_back_enabled then
        x_prev_lb = randn_like(x, n, sigma_max * 0.1)
    end

    for i = 1, n_steps do
        local sigma_curr = sigmas[i]
        local sigma_next = sigmas[i+1]

        if sigma_next == 0.0 then
            local v_final = model_fn(x, sigma_curr, n)
            local dt      = sigma_next - sigma_curr
            for j=1,n do x[j] = x[j] + dt*v_final[j] end
            if verbose then
                print(string.format("[STORM] Step %02d: FINAL (Euler terminal)", i-1))
            end
            break
        end

        local x_prev_lb_before = nil
        if look_back_enabled then
            x_prev_lb_before = vec_clone(x, n)
        end

        local mode, actual_order, cos_sim_out
        x, v_cache, mode, actual_order, cos_sim_out, baseline = storm.storm_step(
            v_cache, x, sigma_curr, sigma_next, model_fn, i-1, baseline, opts, n)

        -- Look-Back smoothing
        local lam = 0.0
        if look_back_enabled and x_prev_lb ~= nil then
            x, lam = storm.look_back_smooth(
                x, x_prev_lb, sigma_curr, sigma_max,
                look_back_lambda, look_back_snr_pow, n)
            if verbose then
                print(string.format("[STORM] LookBack λ=%.4f @ σ=%.3f", lam, sigma_curr))
            end
        end
        x_prev_lb = x_prev_lb_before

        -- SDE Restarts
        local do_restart = enable_restarts and sigma_next > 0
        if do_restart then
            -- Check if current step index is in restart set
            local in_set = false
            for _, rs in ipairs(restart_steps) do
                if rs == (i-1) then in_set = true; break end
            end
            if in_set then
                local s_res = sigma_next + (sigma_curr - sigma_next) * restart_noise_scale
                local n_amt = math.sqrt(math.max(0.0, s_res^2 - sigma_next^2) + 1e-8)

                local noise
                if restart_aligned and #v_cache >= 2 then
                    -- Velocity-aligned Langevin noise
                    -- EMA-weighted mean: recent vectors get highest weight (0.5^(depth-1-i))
                    local _depth = #v_cache
                    local _wsum  = 0.0
                    local v_mean = {}
                    for j=1,n do v_mean[j]=0.0 end
                    for _i, entry in ipairs(v_cache) do
                        local _w = 0.5 ^ (_depth - _i)  -- recent = highest weight
                        _wsum = _wsum + _w
                        for j=1,n do v_mean[j]=v_mean[j]+entry.v[j]*_w end
                    end
                    for j=1,n do v_mean[j]=v_mean[j]/_wsum end
                    local v_norm = vec_norm(v_mean, n) + 1e-8
                    local v_dir  = {}
                    for j=1,n do v_dir[j]=v_mean[j]/v_norm end

                    -- Isotropic noise then project out principal direction
                    local raw   = randn_iso(n, n_amt * restart_s_noise)
                    local proj  = vec_dot(raw, v_dir, n)
                    local aligned = {}
                    for j=1,n do aligned[j] = raw[j] - proj*v_dir[j] end
                    -- Renormalize
                    local rn = vec_norm(raw, n)
                    local an = vec_norm(aligned, n) + 1e-8
                    noise = {}
                    for j=1,n do noise[j] = aligned[j]*(rn/an) end
                    if verbose then
                        print(string.format("[STORM] ♻️  ALIGNED RESTART @ step %d (noise ⊥ v_principal)", i-1))
                    end
                else
                    noise = randn_iso(n, n_amt * restart_s_noise)
                    if verbose then
                        print(string.format("[STORM] ♻️  RESTART @ step %d", i-1))
                    end
                end

                local x_renoise = {}
                for j=1,n do x_renoise[j] = x[j] + noise[j] end
                local v_res = model_fn(x_renoise, s_res, n)
                local dt_res = sigma_next - s_res
                for j=1,n do x[j] = x_renoise[j] + dt_res*v_res[j] end

                if restart_flush_cache then
                    v_cache = {}
                    baseline.prev_mode = "STORK"
                    if verbose then print("[STORM] Cache flushed after restart.") end
                end
            end
        end

        if callback then callback(i-1, x, sigma_next) end
    end

    return x
end

return storm
