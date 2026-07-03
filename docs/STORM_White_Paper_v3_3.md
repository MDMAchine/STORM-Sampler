# STORM: Stabilized Taylor Oscillation with Runge-Kutta Memory
## An Adaptive Stiffness-Switching ODE Sampler for Flow-Matching Diffusion Models

**Alexander Allan (MDMAchine)**  
A&E Concepts, New Bedford MA  

**Version:** v3.3, Pre-print, Patent Pending  
**Date:** July 2026  
**arXiv target:** cs.SD (Sound), cross-list cs.LG  
**Supersedes:** v3.2, v3.1, v3.0, v2.1, v2.0, v1.0

*IP Notice: This paper establishes prior art for the STORM adaptive stiffness-switching ODE
sampler, the Look-Back SNR trajectory smoother, and the ODE manifold shearing suppression
mechanism. A provisional patent application covers these methods.
GPL v3 clean math is shared publicly. © 2026 Alexander Allan / A&E Concepts. All rights
reserved.*

---

## Abstract

We introduce STORM (Stabilized Taylor Oscillation with Runge-Kutta Memory), an adaptive hybrid
ODE sampler for flow-matching diffusion models that dynamically switches between a high-order
stiffness-aware solver and a smooth-region solver on a per-step basis. STORM addresses a
fundamental limitation of uniform-step ODE solvers: the application of identical numerical methods
to regions of the denoising trajectory that differ dramatically in geometric stability. We
demonstrate that the ODE trajectory in flow-matching audio diffusion models contains sharp
curvature discontinuities concentrated in the high-sigma early denoising steps, and that these
discontinuities are the primary source of the characteristic harmonic artifact observed in
AI-generated audio. STORM detects these instability events in real time via an EMA-calibrated
stiffness ratio and deploys a multi-order Adams-Bashforth solver (RK2–5, single NFE per standard
step) with cosine-similarity curvature damping for stiff regions, falling back to DPM++3M for
smooth regions. An SNR-adaptive look-back trajectory smoother further suppresses high-sigma
shearing artifacts. Perceptual evaluation on ACE-Step XL Turbo 4B demonstrates 75–80% reduction
in harmonic hum, +6.18% transient punch, and +1.75% coherence vs. Euler baseline, with minimal
additional function evaluations (sub-stepping adds at most 1–2 NFE per generation on rare
manifold fracture events).

---

## 1. Introduction

Flow-matching diffusion models generate outputs by solving an ordinary differential equation that
transforms a noise distribution into a data distribution across a discrete sequence of denoising
steps. The numerical solver used to advance the ODE at each step has a significant impact on both
the quality and the character of the generated output.

Standard samplers (Euler, DPM++, DDIM) apply a fixed numerical method uniformly across all steps
of the denoising trajectory. This uniform application is computationally efficient but geometrically
naive: it treats steps near σ=1.0 (maximum noise, high geometric uncertainty) identically to steps
near σ=0 (final detail recovery, highly stable trajectory).

In practice, flow-matching audio diffusion models exhibit a characteristic failure mode at high
sigma: the velocity field, the direction the ODE is moving through latent space at each step, undergoes sharp directional reversals that standard solvers do not detect or compensate for. These
reversals create brief but significant trajectory instabilities that manifest acoustically as a
coherent harmonic resonance artifact. In ACE-Step XL Turbo, this artifact concentrates around
426 Hz (Ab4), a musical pitch, which explains why the artifact sounds tonally wrong rather than
simply noisy.

This is not a VAE decoder artifact. The artifact persists with alternative decoders and is
confirmed to originate in the ODE trajectory itself. The XL model's wider hidden dimension
(2560 vs 2048 in the base model) amplifies the interference via expanded cross-attention parameter
space during high-sigma steps. The result is phase-incoherent spectral peaks at 2–8kHz,
concentrated at 426Hz, that do not correspond to the expected harmonic content of the generated
audio.

STORM addresses this by treating the denoising trajectory as a physically heterogeneous system:
geometrically stiff in some regions, smooth in others. The appropriate solver is selected per step
based on real-time measurement of trajectory curvature, without additional model evaluations.

The contributions of this work are:

1. A formal characterization of ODE trajectory stiffness in flow-matching diffusion inference and
   its connection to harmonic artifact generation.
2. STORM: an adaptive stiffness-switching sampler with per-step dispatch between a multi-order RK
   solver (STORK, RK2–5) and DPM++3M.
3. An SNR-adaptive trajectory smoother (Look-Back) that suppresses residual high-sigma shearing
   artifacts.
4. Velocity-aligned SDE restarts that preserve low-frequency trajectory structure while
   perturbing high-frequency content.
5. Empirical validation on ACE-Step 2.6B and XL Turbo 4B with perceptual and community
   evaluation.

---

## 2. Related Work

**Flow Matching (Lipman et al., 2022)** introduced the continuous normalizing flow framework that
underlies ACE-Step and related audio diffusion models. The probability flow ODE formulation defines
a deterministic inference path that STORM's suppressor is designed to preserve.

**DDPM (Ho et al., 2020)** introduced ancestral sampling for discrete-time diffusion. The
mathematical grounding for stochastic noise injection in DDPM's reverse SDE does not transfer to
flow-matching models where no reverse SDE is defined. Applying ancestral noise to a flow-matching
model injects perturbations that move the latent off the learned ODE manifold.

**DPM-Solver (Lu et al., 2022)** and **DPM-Solver++ (Lu et al., 2023)** improve ODE solver
efficiency via high-order Taylor expansion. STORM's STORK solver shares the cached-derivative
multi-step motivation but applies it specifically to stiff trajectory regions detected at runtime,
rather than uniformly.

**ACE-Step (ByteDance, 2025)** is the flow-matching audio diffusion model on which STORM is
primarily validated. ACE-Step uses a DiT architecture with separate conditioning encoders for
lyrics, timbre, and genre.

**Adaptive Guidance (Sadat et al., 2025)** introduced momentum-based classifier-free guidance
correction. STORM's Look-Back smoother shares the trajectory memory motivation but applies it to
the denoising step itself rather than the guidance term.

**Adaptive ODE solvers in scientific computing** (LSODA, VODE, Dormand-Prince) establish the
broader class of methods that STORM adapts. Scientific adaptive solvers use embedded RK pairs to
estimate local truncation error and adjust step size. STORM cannot change step size (the sigma
schedule is fixed by the sampler interface) and instead changes solver order while maintaining
single-NFE per step.

**Look-Ahead/Look-Back flows (arXiv:2602.09449)** provide the mathematical basis for STORM's
Look-Back SNR smoother, applied here to the per-step denoising trajectory rather than a full-pass
post-process.

---

## 3. Method

### 3.1 Stiffness Detection

At each step i, STORM probes the current velocity v_curr = model_fn(x, σ_i) and computes a
stiffness ratio:

```
raw_ratio = ||v_curr - v_prev|| / ||v_curr||
```

This is EMA-smoothed (α=0.30) to reduce step-to-step noise:

```
ema_ratio = α × raw_ratio + (1-α) × ema_ratio_prev
```

During the first n_calib steps (12% of total, min 2, max 5), the smoothed ratio is accumulated
into a baseline mean. After calibration, an adaptive threshold is computed:

```
adaptive_threshold = stiffness_threshold × (baseline_mean / 0.15)
adaptive_threshold = clamp(adaptive_threshold, 0.05, 0.50)
```

If `ema_ratio > adaptive_threshold`, the step is classified as STIFF and dispatched to STORK.
Otherwise it is dispatched to DPM++3M. A hysteresis margin of 0.05 prevents rapid oscillation
between solvers when the ratio is near the threshold.

### 3.2 STORK: Stabilized Taylor Oscillation with Runge-Kutta Memory

STORK is a single-NFE multi-order Adams-Bashforth solver that uses cached velocity derivatives
from previous steps as virtual higher-order correction terms.

**Curvature damping** is applied across all orders:

```
damping = clamp(cosine_similarity(v_curr, v_prev), 0, 1)
```

When the trajectory turns sharply (low cosine similarity), the correction term is suppressed.
This prevents over-extrapolation at manifold boundaries.

**RK2 (1 cached derivative):**
```
v_extrap = v_curr + (α × damping) × (v_curr - v_prev)
x_next = x + dt × (0.5 × v_curr + 0.5 × v_extrap)
```

**RK3 (2 cached derivatives):** Variable-step Adams-Bashforth with Lagrange interpolation
coefficients derived from actual sigma spacing between cached steps. Curvature damping applied
to prediction delta only.

**RK4 / RK5:** Extensions of the variable-step Lagrange interpolation to 3 and 4 history points
respectively.

In `"auto"` mode (default), the highest order the cache supports is used at each step. The cache
fills progressively, early steps run RK2, mid steps RK3/4, late steps RK4/5. Cache depth is
capped at 5 velocity vectors.

### 3.3 DPM++3M (Smooth-Region Solver)

Standard 3rd-order Adams-Bashforth with variable step sizing derived from the sigma schedule.
Used for smooth trajectory regions where stiffness detection does not fire. Unchanged from the
reference implementation.

### 3.4 Adaptive Sub-Stepping (Manifold Fracture Defense)

When a U-turn is detected within a step (cosine similarity < 0.0), STORM recursively splits
the step into two half-steps:

```
sigma_mid = (sigma_curr + sigma_next) / 2
x_mid = stork_step(x, sigma_curr, sigma_mid)
x_out = stork_step(x_mid, sigma_mid, sigma_next)
```

Maximum recursion depth: 2. This prevents infinite recursion on pathological schedules while
providing meaningful correction for manifold fracture events. Sub-step events are tracked per
step in the profiler telemetry (red markers on the RK order panel).

### 3.5 Look-Back SNR Trajectory Smoother

The Look-Back component addresses residual trajectory instability in the high-sigma zone that
stiffness switching alone does not fully correct.

At each step, the current latent is blended with the previous step's pre-blend latent using
a lambda that decays with sigma:

```
lambda(sigma) = lambda_base × (sigma / sigma_max)^snr_power
x.lerp_(x_prev_lb, lambda)
```

At σ = σ_max (first step): lambda = lambda_base.  
At σ = 0 (final step): lambda = 0 (smoother fully disengaged, detail preserved).

`x_prev_lb` stores the raw solver output BEFORE the look-back blend is applied, not the
already-smoothed output. Storing the pre-lerp value prevents compounding non-linearity where
each step blends against an already-smoothed reference. This is the v2.1 critical fix.

**Validated parameters:**

| Schedule | `lambda_base` | `snr_power` |
|---|---|---|
| 25-step ddim_uniform | 0.55 | 1.3 |
| 35-step simple | 0.35 | 1.5 |

The perceptual effect, replacing the metallic artifact with musical saturation character, is
a byproduct of spectral coherence enforcement in the high-sigma zone. The smoother is not
targeted at musical character; it is targeted at manifold shearing suppression. The saturation
character emerges because redistributing spectral energy away from phase-incoherent artifact
frequencies produces the harmonic signature of analog-style distortion.

### 3.6 Velocity-Aligned SDE Restarts (Optional)

Optional stochastic restarts inject noise at specified steps to escape local optima. Two modes:

**Isotropic:** Standard Gaussian noise scaled by `restart_noise_scale`.

**Velocity-aligned Langevin:** Noise is projected perpendicular to the principal velocity
direction, computed as an EMA-weighted mean of cached velocity vectors (recent vectors weighted
by 0.5^(depth-1-i)):

```
v_principal = EMA_weighted_mean(v_cache) / ||EMA_weighted_mean(v_cache)||
proj = dot(noise, v_principal)
aligned_noise = noise - proj × v_principal
aligned_noise = aligned_noise × (||noise|| / ||aligned_noise||)  # renormalize
```

This preserves low-frequency groove and tempo trajectory (encoded in the principal velocity
direction) while injecting high-frequency stochastic perturbation. The EMA weighting ensures
the principal direction reflects the immediate trajectory rather than stale history.

**Usage constraint:** `enable_restarts: false` is the validated default. Restarts at
low-sigma steps (σ < 0.3) inject energy into crystallizing signal and produce audible artifacts.
When used, confine to early steps (step ≤ 7). Always use `ancestral_noise_type: "gaussian"`, Brownian noise accumulates energy drift via `cumsum / sqrt(T)`.

---

## 4. Experimental Validation

### 4.1 Setup

**Model:** ACE-Step XL Turbo 4B (acestep-v15-xl-turbo); ACE-Step 2.6B  
**Schedules:** 35-step simple, 25-step ddim_uniform  
**Hardware:** RTX 4070 Ti Super 16GB  
**Baseline:** Euler with equivalent step count  
**Evaluation:** SCT (30-clip perceptual scoring), spectral analysis (Observer VST),
community blind evaluation

### 4.2 Artifact Characterization

The metallic artifact produced by standard ancestral sampling on ACE-Step XL Turbo is
characterized by:

- Phase-incoherent spectral peaks concentrated at 426 Hz (Ab4), with spread to 2–8kHz
- Sustained tonal quality distinct from expected harmonic content
- Consistent appearance across genres, modulated by genre spectral profile
- Independence from lower-frequency DiT body resonance hum (74–654 Hz range)
- Persistence across VAE decoder variants, confirmed ODE-origin, not decoder artifact

The 426 Hz concentration in XL Turbo specifically is a consequence of the model's wider hidden
dimension (2560 vs 2048). The expanded cross-attention parameter space creates constructive
interference patterns at specific frequency components during high-sigma denoising. Artifact
frequency will differ across architectures and should be measured per deployment context.

### 4.3 Quantitative Results (vs Euler baseline, 35-step simple schedule)

| Metric | STORM vs Euler |
|---|---|
| Transient punch | +6.18% |
| Air | −28.32% (tunable via look_back_lambda) |
| Flatness | +4.11% |
| Coherence | +1.75% |
| Harmonic hum (426 Hz) | 80% → 15–25% |

Air reduction reflects the gold-standard `lambda=0.35` setting. Lower lambda values (0.15–0.20)
recover air with retained hum suppression.

### 4.4 NFE Count

STORM maintains a baseline of 1 NFE per step. Stiffness detection reuses the velocity probe
that STORK computes anyway, no additional model call. Adaptive sub-stepping dynamically
allocates additional evaluations only when manifold fractures are detected (cos_sim < 0.0),
adding at most 1–2 NFE per fracture event, bounded by `SUB_STEP_MAX_DEPTH=2`. In practice,
fracture events are rare (typically 0–2 per generation) and do not materially affect throughput.
All results in Section 4.3 were measured at equivalent total NFE to the Euler baseline.

### 4.5 Community Validation (STORM v2.1, May 2026)

STORM v2.1 was distributed to ACE-Step community developers for independent validation:

**serveurperso** (acestep.cpp maintainer): Received STORM v2.1 cores for C++ integration.
Confirmed functional in the GGUF inference backend.

**scragnog / Captain HOT-Step** (HOT-Step-CPP, ScragVAE maintainer): Implemented STORM same-day.
Confirmed perceptually superior results across tests. "Better than anything else tried."

**mvdirty**: Independently confirmed the perceptual finding without prior briefing on the artifact
mechanism:

> "replaces the typical metallic twinge with a form of distortion that sounds more musical, more
> explicitly produced, like adding saturation to a track or mix"

**Helikaon23**:

> "cleaner with minimal distortion, less of the metallic traits"
> "storm_v21 is the best with even less of the aforementioned traits"

**dernet / iRedsneth** (side-step training tool): Built a full STORM UI in 44 minutes. Porting
to C++.

mmoalem: Running automated sampler comparison tests (CLAP embedding batch
perceptual similarity scoring).

### 4.6 Perceptual Result

The consistent community finding is that STORM does not simply reduce or attenuate the metallic
artifact, it eliminates the phase-incoherent character entirely, replacing it with a perceptually
musical quality. This is consistent with the theoretical prediction: ODE manifold shearing
introduces non-musical phase incoherence; suppressing the shearing allows the learned velocity
field to produce the expected harmonic structure.

The saturation-like character noted by independent testers (mvdirty) is consistent with the
mechanism: spectral coherence enforcement in the high-sigma zone redistributes energy from
artifact frequency directions into adjacent harmonics, producing the perceptual signature of
analog-style harmonic distortion rather than digital artifact.

---

## 5. Discussion

### 5.1 Why 426 Hz

The concentration at 426 Hz (Ab4) in ACE-Step XL Turbo is architecture-specific. The XL model's
wider hidden dimension (2560 vs 2048) creates specific constructive interference patterns between
frequency components during high-sigma denoising. This frequency is a musical pitch rather than
an arbitrary noise frequency, which is why the artifact sounds tonally wrong, it is spectrally
coherent, just incorrectly placed. The artifact frequency will differ across model architectures
and should be empirically measured per deployment.

### 5.2 Relationship to Adaptive Scientific Solvers

STORM is philosophically related to LSODA and VODE, scientific ODE solvers that detect stiffness
at runtime and switch integration methods accordingly. The difference: scientific adaptive solvers
use embedded RK pairs to estimate local truncation error and adjust step size. STORM cannot adjust
step size (the sigma schedule is fixed by the sampler interface) and instead adjusts solver order
while maintaining single-NFE per step. The stiffness signal is geometric (velocity direction
change) rather than error-based, which is appropriate for the diffusion inference context where
error estimation is not available without additional model evaluations.

### 5.3 Modality Generalization

STORM's stiffness detection and STORK solver operate on velocity field geometry regardless of
what the field represents. The Look-Back smoother operates on the latent trajectory regardless of
modality. STORM is validated on audio (ACE-Step) but the C++ header-only and Lua implementations
are modality-agnostic. Pre-registered future validation targets: LTX-Video, Wan, Cosmos.

---

## 6. Distribution

STORM is distributed as:
- ComfyUI node (ComfyUI_MD_Nodes)
- C++ header-only (storm_sampler_core.hpp, C++17)
- Lua solver plugin (HOT-Step native)

All shared code uses clean GPL v3 mathematics.

---

## 7. Conclusion

STORM demonstrates that adaptive solver dispatch based on real-time trajectory stiffness
measurement is a viable and effective approach to improving diffusion inference quality without
increasing NFE count. The high-sigma zone of flow-matching audio diffusion models contains
geometrically distinct regions that benefit from different numerical treatment, this observation
generalizes beyond audio to any flow-matching architecture where trajectory curvature is
non-uniform across the denoising schedule.

The combination of stiffness-switching dispatch, Look-Back SNR smoothing, and velocity-aligned
SDE restarts addresses the harmonic artifact problem at its mathematical source rather than
through post-processing or model modification.

---

## References

[1] Lipman et al. "Flow Matching for Generative Modeling." arXiv:2210.02747, 2022.  
[2] Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.  
[3] Lu et al. "DPM-Solver." NeurIPS 2022.  
[4] Lu et al. "DPM-Solver++." arXiv:2211.01095, 2023.  
[5] ACE-Step: A Step Towards Music Generation Foundation Models. ByteDance Research, 2025.  
[6] Sadat et al. "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models." arXiv:2410.02416, 2025.  
[7] arXiv:2602.09449, Look-Ahead/Look-Back flows.

---

## Acknowledgments

Community validation: mvdirty, Helikaon23, scragnog (Captain HOT-Step), serveurperso,
dernet (iRedsneth), mmoalem.  
ACE-Step team at ByteDance/StepFun for the base model.

---

*© 2026 Alexander Allan (MDMAchine) · A&E Concepts*  
*Patent Pending · All Rights Reserved*  
*Version 3.3, July 2026*
