![STORM](assets/storm_gist_header_v3_animated.svg)

# STORM Sampler
### Stabilized Taylor Oscillation with Runge-Kutta Memory
**Adaptive Stiffness-Switching ODE Sampler for Flow-Matching Diffusion Models**

*Alexander Allan (MDMAchine) · A&E Concepts · GPL v3*

---

## What Is STORM?

STORM is an adaptive hybrid ODE sampler for ACE-Step and other flow-matching audio diffusion models. It solves a specific problem: standard DDPM ancestral samplers inject stochastic noise designed for Markovian reverse processes into flow-matching models that operate on probability flow ODEs. This architectural mismatch shears latents off the ODE manifold during early denoising steps, producing a characteristic metallic artifact in generated audio.

STORM fixes this via geometric trajectory stabilization: stiffness-detecting per-step solver dispatch, SNR-adaptive look-back smoothing, and velocity-aligned SDE restarts. The result is elimination of the metallic twinge and replacement with perceptually musical, harmonically coherent output.

**Validated on:** ACE-Step 2.6B, ACE-Step XL Turbo 4B  
**Deployment:** ComfyUI node · C++ header-only · Lua plugin (HOT-Step)  
**License:** GPL v3 · Commercial dual-license: contact A&E Concepts

---

## Architecture

STORM dispatches between two solvers per denoising step based on measured trajectory stiffness:

```
Input: x (latent), sigmas (schedule)
│
├── Calibration phase (first 12% of steps)
│   └── Build stiffness baseline via EMA of velocity delta ratio
│
├── Per-step dispatch loop
│   ├── Probe velocity: v_curr = model_fn(x, σ)
│   ├── Compute stiffness(v_curr, v_cache, baseline)
│   │   ├── STIFF  → STORK RK[2→5] with curvature damping
│   │   │   └── U-turn detected? → adaptive sub-step (recursive, max depth 2)
│   │   └── SMOOTH → DPM++3M (variable-step Adams-Bashforth)
│   ├── Look-Back SNR Smoother: x.lerp_(x_prev_lb, λ(σ))
│   ├── [Optional] SDE Restart: velocity-aligned Langevin noise
│   └── Update v_cache (depth=5), update baseline
│
└── Output: x (denoised latent)
```

### Stiffness Detection

Per-step EMA-smoothed velocity delta ratio, auto-calibrated over the first `calib_frac` of steps. When the trajectory velocity changes direction sharply, STORK handles the step. When the trajectory is geometrically smooth, DPM++3M handles it. A hysteresis margin prevents thrashing between modes.

### STORK (Stiff-Region Solver)

Adaptive-order Runge-Kutta using cached velocity derivatives, all orders are single-NFE (no extra model evaluations). Cache fills progressively: early steps run RK2, mid steps RK3/4, late steps RK4/5 in `"auto"` mode.

Curvature damping: `damping = clamp(cosine_similarity(v_curr, v_prev), 0, 1)`, when the trajectory curves sharply, the correction term is suppressed to prevent over-extrapolation on manifold boundaries.

**Adaptive sub-stepping:** When a U-turn is detected mid-step (`cos_sim < 0.0`), STORM recursively splits the step into two half-steps (max depth 2), walking the manifold corner carefully.

### Look-Back SNR Smoother

SNR-adaptive EMA blend of the current denoising output with a stored reference from the previous step:

```
λ(σ) = lambda_base × (σ / σ_max)^snr_power
x_smoothed = lerp(x_next, x_prev_lb, λ)
```

At high σ (early steps, where manifold shearing is worst): λ is large, strong reference pull, suppresses shearing.  
At low σ (late steps, detail recovery): λ approaches 0, full trust in current step.

`x_prev_lb` is stored **before** the lerp (not after), this prevents compounding non-linearity. This is the v2.1 critical fix.

### Velocity-Aligned SDE Restarts (optional)

Langevin noise injection at designated restart steps. Noise is aligned along the current velocity direction rather than injected isotropically, this maintains directional momentum while destabilizing stuck trajectories.

```
noise_aligned = noise + alignment_strength × velocity / (|velocity| + ε)
```

> **Note:** `enable_restarts: false` is gold standard. Late restarts (σ < 0.3) inject energy into crystallizing signal. If restarts needed, use early-only (step ≤ 7). Always use `ancestral_noise_type: "gaussian"`, Brownian noise produces cumulative energy drift.

---

## Validated Parameters

### ComfyUI / Python, 25-step ddim_uniform (recommended)

| Parameter | Value | Notes |
|---|---|---|
| `stiffness_threshold` | 0.15 | Auto-calibrates from this base |
| `hysteresis_margin` | 0.05 | Prevents mode thrashing |
| `ema_alpha` | 0.30 | Stiffness EMA smoothing |
| `cache_depth` | 5 | Max cached velocity vectors |
| `calib_frac` | 0.12 | 12% of steps for baseline |
| `rk_order` | `"auto"` | Highest order cache supports |
| `adaptive_sub_step` | `true` | Manifold fracture defense |
| `sub_step_threshold` | 0.0 | cos_sim below this triggers split |
| `sub_step_max_depth` | 2 | Max recursive split depth |
| `look_back_enabled` | `true` | |
| `look_back_lambda` | 0.55 | 25-step ddim_uniform validated |
| `look_back_snr_power` | 1.3 | 25-step ddim_uniform validated |
| `enable_restarts` | `false` | Gold standard off |
| `ancestral_noise_type` | `"gaussian"` | Never Brownian |

**35-step simple schedule:** `look_back_lambda=0.35`, `look_back_snr_power=1.5`

### HOT-Step Lua (scragnog_edit_storm_sampler_core.lua)

| Label | Key | Default |
|---|---|---|
| Detail Sensitivity | `stiffness_threshold` | 0.15 |
| Coherence Smoothing | `look_back_lambda` | 0.15 |
| Early-Step Focus | `look_back_snr_power` | 1.5 |
| Precision Level | `rk_order` | auto |

---

## Perceptual Results

Measured vs Euler baseline, 35-step simple schedule:

| Metric | Delta |
|---|---|
| Punch | +6.18% |
| Flatness | +4.11% |
| Coherence | +1.75% |
| Air | -28.32% (tunable via look_back_lambda) |
| Hum suppression | 80% to 15-25% |

Community finding (mvdirty, 2026-05-11): STORM replaces the metallic AI twinge with a form of saturation that sounds explicitly produced, musical rather than artifactual. This is a byproduct of spectral coherence enforcement in the high-sigma zone, not a targeted output.

---

## Deployment

### ComfyUI

```
custom_nodes/STORM_Sampler/
├── __init__.py                   ← node registration
├── core/
│   └── storm_sampler_core.py     ← pure math core
└── samplers/
    ├── __init__.py               ← required for Python import
    ├── MD_STORM_Sampler.py       ← ComfyUI node wrapper
    └── MD_LookBack_Smoother.py   ← standalone smoother (any sampler)
```

**`__init__.py` (root):**
```python
from .samplers.MD_STORM_Sampler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

**`samplers/__init__.py`:**
```python
from .MD_STORM_Sampler import *
```

Drop the `STORM_Sampler` folder into `ComfyUI/custom_nodes/`, restart ComfyUI, and look for **"MD: STORM Hybrid Solver"** in the sampler nodes.

Basic workflow: `BasicScheduler → MD STORM Sampler → VAE Decode`

YAML parameter override: paste any parameter from the YAML template into the `yaml_settings_str` input to override defaults without touching the sliders.

Telemetry: enable `plot_trajectory` to save a 5-panel diagnostic plot (stiffness, RK order, x_mag, v_mag, cos_sim) to ComfyUI `output/` on every run.

### C++ (acestep.cpp / HOT-Step-CPP / sidestep)

`storm_sampler_core.hpp` is a self-contained header-only implementation. Drop into your project and include:

```cpp
#include "storm_sampler_core.hpp"

storm::Config cfg;
cfg.threshold = 0.15f;
cfg.look_back_lambda = 0.55f;
cfg.look_back_snr_power = 1.3f;
// ... configure as needed

storm::run(x, sigmas, model_fn, userdata, cfg);
```

C++ port in production: [HOT-Step-CPP](https://github.com/scragnog/HOT-Step-CPP) (scragnog), [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (serveurperso).

### Lua ([HOT-Step-CPP](https://github.com/scragnog/HOT-Step-CPP))

`storm_sampler_core.lua`: standard HOT-Step solver plugin (owns_loop = true).  
`scragnog_edit_storm_sampler_core.lua`: HOT-Step optimized build with user-friendly labels. Recommended for HOT-Step users.

Place in your HOT-Step solvers directory. The solver will appear as **STORM** in the solver list.

### MD LookBack Smoother (standalone)

STORM has the full SNR-adaptive look-back smoother built in. For any other sampler (Euler, DPM++, etc.), this standalone node provides a simpler fixed-weight post-process version:

```
BasicScheduler → Sampler → MD: Look-Back Smoother → VAE Decode
```

`lambda_base` controls blend strength directly (default 0.05). No sigmas input needed, the post-process version uses the latent's own energy to scale the smoothing perturbation.

Validated: `lambda_base=0.03-0.08` for gentle post-process smoothing.

### Precision Note

STORM is validated at `float32` and `bfloat16`. Strict `float16` can produce NaN in the cosine similarity computation (dot-product overflow before division) used by stiffness detection and curvature damping. The solver's NaN guard catches this and falls back to cache flush + Euler, but `float32` or `bfloat16` is recommended for correct adaptive behavior.

---

## Files

| File | Description |
|---|---|
| `__init__.py` | ComfyUI node registration |
| `core/storm_sampler_core.py` | Core math, GPL v3, clean |
| `core/storm_sampler_core.lua` | Lua implementation |
| `core/storm_sampler_core.hpp` | C++ header-only implementation |
| `core/storm_sampler_core_hotstep.lua` | HOT-Step optimized Lua (credit: scragnog) |
| `samplers/MD_STORM_Sampler.py` | ComfyUI node wrapper |
| `samplers/MD_LookBack_Smoother.py` | Standalone look-back smoother node |
| `docs/STORM_White_Paper_v3_3.md` | Technical white paper |

---

## Acknowledgments

- **[scragnog / Captain HOT-Step](https://github.com/scragnog/HOT-Step-CPP)**, HOT-Step-CPP maintainer. Same-day STORM implementation, HOT-Step Lua port, perceptual validation. The `scragnog_edit` Lua file is his work. Shoutout to scragnog for featuring STORM in HOT-Step-CPP and the love shown on the [HOT-Step repo](https://github.com/scragnog/HOT-Step-CPP).
- **[serveurperso](https://github.com/ServeurpersoCom/acestep.cpp)**, acestep.cpp maintainer. C++ integration and review.
- **mvdirty**, perceptual validation. Coined "metallic twinge to musical saturation."
- **[dernet / iRedsneth](https://github.com/koda-dernet/Side-Step)**, STORM UI in 44 minutes. C++ port in progress.
- **Helikaon23**, early testing and feedback.
- **mmoalem**, automated sampler comparison testing.

---

## License

**GPL v3**, free for open-source use. See [`LICENSE`](LICENSE).

Commercial closed-source integration requires a dual-license commercial exemption. Contact: A&E Concepts on GitHub.

---

## White Paper

Full technical treatment of the stiffness-detecting solver dispatch, look-back SNR smoother, and velocity-aligned SDE restarts:

[`docs/STORM_White_Paper_v3_3.md`](docs/STORM_White_Paper_v3_3.md)

---

*© 2026 Alexander Allan (MDMAchine) · A&E Concepts*
