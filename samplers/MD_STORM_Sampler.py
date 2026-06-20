# MD_STORM_Sampler.py
# ComfyUI node wrapper -- STORM Adaptive Hybrid Solver
# Follows ComfyUI_MD_Nodes core/wrapper architecture
#
# © 2026 Alexander Allan (MDMAchine) | A&E Concepts
# GPL v3
#
# Version: 2.1.0
# Created: 2026-05-08 | Revised: 2026-05-10
# Pack: ComfyUI_MD_Nodes
# Location: samplers/MD_STORM_Sampler.py
#
# Changelog:
#   2.0.0 -- Full feature parity with dev node v1.5 (confirmed working 2026-05-10)
#             RK2/3/4/5 adaptive multi-order + YAML rk_order control
#             Adaptive sub-stepping (manifold fracture defense)
#             SDE restarts: isotropic + velocity-aligned Langevin
#             Adaptive calibration steps
#             Full 5-panel telemetry plot saved to ComfyUI output/ folder
#             YAML input/template/params outputs
#             PerformanceProfiler with x_mag/v_mag/cos_sim/rk_order per step
#   1.3.0 -- YAML, plot, profiler (basic)
#   1.2.0 -- Look-Back SNR smoother
#   1.1.0 -- Correct velocity, curvature damping, deque cache
#   1.0.0 -- Initial release

import io
import os
import time
import torch
import comfy.samplers

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

try:
    from ..core.storm_sampler_core import storm_sampler
    _STORM_SOURCE = "core"
except ImportError:
    try:
        from storm_sampler_core import storm_sampler
        _STORM_SOURCE = "local"
    except ImportError:
        storm_sampler = None
        _STORM_SOURCE = "unavailable"


# =============================================================================
# YAML TEMPLATE
# =============================================================================

YAML_TEMPLATE_STORM = """\
# ======================================================================
# ⚡ MD STORM Hybrid Solver v2.0 — Complete Parameter Reference
# Paste into yaml_settings_str to override any value.
# ======================================================================

# ── STIFFNESS DETECTION ──────────────────────────────────────────────
stiffness_threshold: 0.15    # Base threshold (0.05-0.50). Auto-calibrates.
hysteresis_margin: 0.05      # Dead-band preventing STORK/DPM++ thrashing.
ema_alpha: 0.30              # EMA weight on stiffness observation.
cache_depth: 5               # Velocity history depth. Must be >= rk_order.
calib_frac: 0.12             # Fraction of steps used for calibration (12%).

# ── RK MULTI-ORDER ───────────────────────────────────────────────────
rk_order: auto               # auto | 2 | 3 | 4 | 5
                              # auto = highest order cache supports each step.
                              # All orders are single NFE (cached derivatives).

# ── ADAPTIVE SUB-STEPPING ────────────────────────────────────────────
adaptive_sub_step: true      # Split step on U-turn detection (cos_sim < threshold).
sub_step_threshold: 0.0      # cos_sim below this triggers sub-step.
sub_step_max_depth: 2        # Max recursive depth (prevents infinite loops).

# ── LOOK-BACK SNR SMOOTHER ───────────────────────────────────────────
look_back_enabled: true
look_back_lambda: 0.35       # 0.35 @ 35-step simple. 0.55 @ 25-step ddim_uniform.
look_back_snr_power: 1.5     # 1.5 @ 35-step. 1.3 @ 25-step.

# ── SDE RESTARTS ─────────────────────────────────────────────────────
enable_restarts: false
restart_steps: "10,20"       # Comma-separated step indices.
restart_noise_scale: 0.5     # Jump distance (0=none, 1=full sigma gap).
restart_s_noise: 1.0         # Noise magnitude multiplier.
restart_seed: 42
restart_flush_cache: true    # Flush velocity cache after restart (recommended).
restart_aligned_noise: true  # Langevin: noise perpendicular to velocity direction.
                              # Preserves groove/tempo, glitches HF only.

# ── DEV / DEBUG ──────────────────────────────────────────────────────
force_pure_euler: false
debug_mode: "0 - Silent"     # 0-Silent | 1-Info | 2-Verbose
enable_profiling: false
plot_trajectory: true
"""


# =============================================================================
# YAML PARAMS BUILDER
# =============================================================================

def build_yaml_params(cfg):
    def fmt(v):
        if isinstance(v, bool):  return str(v).lower()
        if isinstance(v, str):   return f'"{v}"'
        if isinstance(v, float): return f"{v:.6g}"
        return str(v)
    W = 70
    lines = [
        "=" * W,
        "# ⚡ ACTIVE PARAMETERS — MD STORM Hybrid Solver v2.0",
        "# Copy any line into yaml_settings_str to override.",
        "=" * W, "",
        "# ── STIFFNESS ──────────────────────────────────────────────────────",
        f"stiffness_threshold: {fmt(cfg['stiffness_threshold'])}",
        f"hysteresis_margin:   {fmt(cfg['hysteresis_margin'])}",
        f"ema_alpha:           {fmt(cfg['ema_alpha'])}",
        f"cache_depth:         {fmt(cfg['cache_depth'])}",
        f"calib_frac:          {fmt(cfg['calib_frac'])}", "",
        "# ── RK ORDER ────────────────────────────────────────────────────────",
        f"rk_order:            {fmt(cfg['rk_order'])}", "",
        "# ── SUB-STEPPING ────────────────────────────────────────────────────",
        f"adaptive_sub_step:   {fmt(cfg['adaptive_sub_step'])}",
        f"sub_step_threshold:  {fmt(cfg['sub_step_threshold'])}",
        f"sub_step_max_depth:  {fmt(cfg['sub_step_max_depth'])}", "",
        "# ── LOOK-BACK ───────────────────────────────────────────────────────",
        f"look_back_enabled:   {fmt(cfg['look_back_enabled'])}",
        f"look_back_lambda:    {fmt(cfg['look_back_lambda'])}",
        f"look_back_snr_power: {fmt(cfg['look_back_snr_power'])}", "",
        "# ── RESTARTS ────────────────────────────────────────────────────────",
        f"enable_restarts:     {fmt(cfg['enable_restarts'])}",
        f"restart_steps:       {fmt(cfg['restart_steps'])}",
        f"restart_aligned_noise: {fmt(cfg['restart_aligned_noise'])}", "",
        "# ── DEBUG ───────────────────────────────────────────────────────────",
        f"force_pure_euler:    {fmt(cfg['force_pure_euler'])}",
        f"debug_mode:          {fmt(cfg['debug_mode'])}",
        f"enable_profiling:    {fmt(cfg['enable_profiling'])}",
        f"plot_trajectory:     {fmt(cfg['plot_trajectory'])}",
    ]
    return "\n".join(lines)


# =============================================================================
# PERFORMANCE PROFILER
# =============================================================================

class STORMProfiler:
    def __init__(self, enabled=True):
        self.enabled    = enabled
        self.start_time = None
        self.end_time   = None
        self.step_log   = []

    def start(self):
        if self.enabled:
            self.start_time = time.time()
            self.step_log   = []

    def stop(self):
        if self.enabled:
            self.end_time = time.time()

    def record(self, idx, sigma, s_next, mode, rk_order, lb_lam, ratio,
               cos_sim, elapsed_s, x_mag=0.0, v_mag=0.0, sub_depth=0):
        if self.enabled:
            self.step_log.append({
                "idx": idx, "sigma": sigma, "s_next": s_next,
                "mode": mode, "rk_order": rk_order,
                "lb_lam": lb_lam, "ratio": ratio,
                "cos_sim": cos_sim, "ms": elapsed_s * 1000.0,
                "x_mag": x_mag, "v_mag": v_mag,
                "sub_depth": sub_depth,
            })

    def print_report(self, verbose=False):
        if not self.enabled or not self.step_log:
            return
        total = (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0
        n     = len(self.step_log)
        stork = sum(1 for s in self.step_log if s["mode"] == "STORK")
        dpm   = sum(1 for s in self.step_log if s["mode"] == "DPM++")
        rk_d  = {}
        for s in self.step_log:
            rk_d[s["rk_order"]] = rk_d.get(s["rk_order"], 0) + 1
        W = 64
        lines = [
            "\n" + "=" * W,
            "📊 [MD_STORM v2.0] ANALYTICS REPORT",
            "=" * W,
            f"⏱️  Total: {total:.3f}s   Steps: {n}   Avg: {(total/n*1000):.1f}ms/step",
            f"⚡  Dispatch: STORK={stork}  DPM++={dpm}",
            f"🔢  RK dist: { {k: v for k, v in sorted(rk_d.items())} }",
        ]
        lb_vals  = [s["lb_lam"] for s in self.step_log if s["lb_lam"] > 0]
        cos_vals = [s["cos_sim"] for s in self.step_log if s["cos_sim"] is not None]
        if lb_vals:
            lines.append(f"🌊  LookBack λ: max={max(lb_vals):.4f} mean={sum(lb_vals)/len(lb_vals):.4f}")
        if cos_vals:
            lines.append(f"📐  cos_sim: max={max(cos_vals):.4f} mean={sum(cos_vals)/len(cos_vals):.4f}")
        if verbose:
            lines += ["",
                f"{'idx':>4}  {'σ':>7}  {'→σ':>7}  {'mode':>5}  {'RK':>2}  "
                f"{'ratio':>6}  {'cos':>6}  {'λLB':>6}  {'ms':>6}",
                "-" * 58]
            for s in self.step_log:
                cs = f"{s['cos_sim']:.4f}" if s["cos_sim"] is not None else "  N/A "
                lines.append(
                    f"{s['idx']:>4}  {s['sigma']:>7.4f}  {s['s_next']:>7.4f}"
                    f"  {s['mode']:>5}  {s['rk_order']:>2}  "
                    f"{s['ratio']:>6.3f}  {cs:>6}  {s['lb_lam']:>6.3f}  {s['ms']:>6.1f}")
        lines.append("=" * W)
        print("\n".join(lines))


# =============================================================================
# TRAJECTORY PLOT (5-panel + 3D attractor)
# =============================================================================

def build_trajectory_plot(step_log, cfg, calib_steps):
    if not _PLOT_AVAILABLE or not step_log:
        return None
    try:
        idxs    = [s["idx"]    for s in step_log]
        ratios  = [s["ratio"]  for s in step_log]
        lambdas = [s["lb_lam"] for s in step_log]
        modes     = [s["mode"]              for s in step_log]
        rk_ords   = [s["rk_order"]         for s in step_log]
        ms_vals   = [s["ms"]               for s in step_log]
        sub_depths = [s.get("sub_depth", 0) for s in step_log]
        cos_sims= [s["cos_sim"] if s["cos_sim"] is not None else 0.0 for s in step_log]
        x_mags  = [s.get("x_mag", 0.0) for s in step_log]
        v_mags  = [s.get("v_mag", 0.0) for s in step_log]

        mode_colors = {"STORK": "#00BFFF", "DPM++": "#00FF88",
                       "EULER": "#FFD700", "FINAL": "#888888"}
        rk_alpha    = {1: 0.30, 2: 0.45, 3: 0.60, 4: 0.80, 5: 0.95}
        bar_colors  = [mode_colors.get(m, "#AAAAAA") for m in modes]
        bar_alphas  = [rk_alpha.get(r, 0.60) for r in rk_ords]

        fig = plt.figure(figsize=(13, 12), facecolor="#0A0A0A",
                         constrained_layout=False)
        fig.suptitle("⚡ STORM v2.0 — Trajectory Telemetry & Phase-Space Attractor",
                     color="#00FFFF", fontsize=13, fontweight="bold")

        from matplotlib.gridspec import GridSpec
        gs   = GridSpec(4, 2, figure=fig, width_ratios=[1.6, 1.0], hspace=0.45, wspace=0.35)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
        ax3d = fig.add_subplot(gs[:, 1], projection="3d")

        def style_ax(ax):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#666666", labelsize=8)
            ax.grid(color="#1C1C1C", linewidth=0.5, axis="y")
            ax.spines[:].set_color("#2A2A2A")

        def style_3d(ax):
            ax.set_facecolor("#0A0A0A")
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor("#1A1A1A")
            ax.grid(False)
            ax.tick_params(colors="#555555", labelsize=6)

        transitions = [(idxs[i], modes[i-1], modes[i])
                       for i in range(1, len(modes))
                       if modes[i] != modes[i-1] and modes[i] not in ("FINAL",)]

        restart_marks = []
        if cfg.get("enable_restarts"):
            for rs in str(cfg.get("restart_steps", "")).split(","):
                try: restart_marks.append(int(rs.strip()))
                except: pass

        def draw_transitions(ax):
            for (xp, fm, tm) in transitions:
                ax.axvline(xp - 0.5, color="#FF4444", linewidth=0.8, linestyle="--", alpha=0.5)
            for rm in restart_marks:
                ax.axvline(rm, color="#FF9F43", linewidth=1.0, linestyle=":", alpha=0.7)
            ax.axvspan(-0.5, calib_steps - 0.5, color="#1A1A2E", alpha=0.4,
                       label=f"Calib ({calib_steps})")

        # ── Plot 1: Stiffness + cos_sim + mode bars ──
        ax1 = axes[0]; style_ax(ax1)
        for x, r, c, a in zip(idxs, ratios, bar_colors, bar_alphas):
            ax1.bar(x, r, color=c, alpha=a, width=0.8)
        ax1b = ax1.twinx()
        ax1b.plot(idxs, cos_sims, color="#FF9F43", linewidth=1.2, alpha=0.7, label="cos_sim")
        ax1b.set_ylabel("cos_sim", color="#FF9F43", fontsize=8)
        ax1b.tick_params(colors="#FF9F43", labelsize=7)
        ax1b.set_ylim(-0.1, 1.1)
        draw_transitions(ax1)
        ax1.set_ylabel("Stiffness Ratio", color="#AAAAAA", fontsize=9)
        patches = [mpatches.Patch(color=c, label=m) for m, c in mode_colors.items() if m != "FINAL"]
        rk_p    = [mpatches.Patch(color="#FFFFFF", alpha=rk_alpha.get(r, 0.6), label=f"RK{r}")
                   for r in sorted(set(rk_ords)) if isinstance(r, int)]
        ax1.legend(handles=patches + rk_p + [mlines.Line2D([], [], color="#FF9F43", label="cos_sim")],
                   loc="upper right", fontsize=7, facecolor="#1A1A1A",
                   labelcolor="#CCCCCC", edgecolor="#333333", ncol=3)

        # ── Plot 2: Look-back λ + ratio overlay ──
        ax2 = axes[1]; style_ax(ax2)
        if any(l > 0 for l in lambdas):
            ax2.plot(idxs, lambdas, color="#FF6B9D", linewidth=1.8, marker="o",
                     markersize=3, label="LookBack λ")
            ax2.fill_between(idxs, lambdas, alpha=0.15, color="#FF6B9D")
            rn = [r / max(max(ratios), 1e-8) * max(lambdas) for r in ratios]
            ax2.plot(idxs, rn, color="#00BFFF", linewidth=0.8, linestyle=":", alpha=0.5,
                     label="ratio (scaled)")
        else:
            ax2.text(0.5, 0.5, "Look-Back Disabled", transform=ax2.transAxes,
                     ha="center", va="center", color="#555555", fontsize=10)
        draw_transitions(ax2)
        ax2.set_ylabel("LookBack λ", color="#AAAAAA", fontsize=9)
        ax2.legend(fontsize=7, facecolor="#1A1A1A", labelcolor="#CCCCCC", edgecolor="#333333")

        # ── Plot 3: RK order ──
        ax3 = axes[2]; style_ax(ax3)
        rk_num = [r if isinstance(r, int) else 2 for r in rk_ords]
        ax3.bar(idxs, rk_num, color=[mode_colors.get(m, "#AAAAAA") for m in modes],
                alpha=0.7, width=0.8)
        ax3.set_ylabel("RK Order", color="#AAAAAA", fontsize=9)
        ax3.set_yticks([2, 3, 4, 5])
        ax3.set_yticklabels(["RK2", "RK3", "RK4", "RK5"], fontsize=7)
        ax3.set_ylim(1, 6)
        draw_transitions(ax3)
        # Sub-step fracture markers -- red downward triangle where manifold split
        for _si, _sd in zip(idxs, sub_depths):
            if _sd > 0:
                ax3.scatter(_si, min(_sd + 2, 5.5), color="#FF0000",
                            marker="v", s=40, zorder=5)
        ax3.legend(handles=[
            mpatches.Patch(color="#1A1A2E", alpha=0.6, label="Calibration"),
            mlines.Line2D([], [], color="#FF4444", linestyle="--", label="Mode switch"),
            mlines.Line2D([], [], color="#FF9F43", linestyle=":", label="Restart"),
            mlines.Line2D([], [], color="#FF0000", marker="v", linestyle="None",
                          markersize=6, label="Manifold fracture"),
        ], fontsize=7, facecolor="#1A1A1A", labelcolor="#CCCCCC", edgecolor="#333333")

        # ── Plot 4: Timing (log scale, step-0 annotated) ──
        ax4 = axes[3]; style_ax(ax4)
        ax4.bar(idxs, ms_vals, color="#9B59B6", alpha=0.75, width=0.8, label="ms/step")
        ms_body = ms_vals[1:] if len(ms_vals) > 1 else ms_vals
        avg_ms  = sum(ms_body) / len(ms_body) if ms_body else 0
        ax4.axhline(avg_ms, color="#E74C3C", linewidth=1.0, linestyle="--",
                    label=f"avg (excl. step 0): {avg_ms:.1f}ms")
        if ms_vals:
            ax4.annotate(f"{ms_vals[0]:.0f}ms\n(init)",
                         xy=(0, ms_vals[0]), xytext=(2, avg_ms * 1.3),
                         color="#FFAA00", fontsize=6,
                         arrowprops=dict(arrowstyle="->", color="#FFAA00", lw=0.8))
        if ms_vals and ms_vals[0] > avg_ms * 5:
            ax4.set_yscale("log")
            ax4.set_ylabel("ms / step (log)", color="#AAAAAA", fontsize=9)
        else:
            ax4.set_ylabel("ms / step", color="#AAAAAA", fontsize=9)
        ax4.set_xlabel("Step Index", color="#AAAAAA", fontsize=9)
        ax4.legend(fontsize=7, facecolor="#1A1A1A", labelcolor="#CCCCCC", edgecolor="#333333")

        # Config footer
        stork_n = sum(1 for m in modes if m == "STORK")
        dpm_n   = sum(1 for m in modes if m == "DPM++")
        fig.text(0.5, 0.005,
                 f"threshold={cfg['stiffness_threshold']}  λ={cfg['look_back_lambda']}  "
                 f"snr={cfg['look_back_snr_power']}  rk={cfg['rk_order']}  "
                 f"STORK={stork_n}  DPM++={dpm_n}  calib={calib_steps}",
                 ha="center", fontsize=7, color="#444444", style="italic")

        # ── 3D Phase-Space Attractor ──
        if len(x_mags) >= 3:
            style_3d(ax3d)
            xs = np.array(x_mags)
            ys = np.array(v_mags)
            zs = np.array(ratios)
            seg_colors = []
            for m in modes:
                if m == "STORK":    seg_colors.append([0.9, 0.2, 0.2, 0.9])
                elif m == "DPM++":  seg_colors.append([0.0, 0.75, 1.0, 0.9])
                else:               seg_colors.append([1.0, 0.85, 0.0, 0.6])
            for i in range(len(xs) - 1):
                ax3d.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                          color=seg_colors[i], linewidth=1.5, alpha=0.8)
            rk_sizes = [20 + (r if isinstance(r, int) else 2) * 8 for r in rk_ords]
            ax3d.scatter(xs, ys, zs, c=[m[:3] for m in seg_colors],
                         s=rk_sizes, alpha=0.7, depthshade=True)
            ax3d.scatter([xs[0]], [ys[0]], [zs[0]], color="#00FF00",
                         s=80, marker="^", label="Start")
            ax3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#FF00FF",
                         s=80, marker="*", label="End")
            ax3d.set_xlabel("||x||", color="#888888", fontsize=7, labelpad=4)
            ax3d.set_ylabel("||v||", color="#888888", fontsize=7, labelpad=4)
            ax3d.set_zlabel("Stiffness", color="#888888", fontsize=7, labelpad=4)
            ax3d.set_title("Phase-Space Attractor", color="#AAAAAA", fontsize=9, pad=6)
            ax3d.legend(fontsize=6, loc="upper right", facecolor="#1A1A1A",
                        labelcolor="#CCCCCC", edgecolor="#333333")
        else:
            ax3d.text2D(0.5, 0.5, "Need 3+ steps\nfor attractor",
                        transform=ax3d.transAxes, ha="center", va="center",
                        color="#555555", fontsize=10)
            ax3d.set_title("Phase-Space Attractor", color="#555555", fontsize=9)

        try:
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        except Exception:
            pass

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, facecolor="#0A0A0A", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        from PIL import Image
        img = Image.open(buf).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    except Exception as e:
        print(f"[MD_STORM] Plot failed: {e}")
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# SAMPLER BRIDGE
# =============================================================================

class STORMSamplerBridge:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.profiler = STORMProfiler(
            enabled=(cfg["enable_profiling"] or cfg["debug_mode"] != "0 - Silent"))
        self._calib_steps = max(2, min(5, int(35 * cfg["calib_frac"])))  # estimated

    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        if storm_sampler is None:
            raise RuntimeError("[MD_STORM] storm_sampler_core not found. Place in core/ folder.")
        if extra_args is None:
            extra_args = {}

        cfg = self.cfg
        n_steps = len(sigmas) - 1
        self._calib_steps = max(2, min(5, int(n_steps * cfg["calib_frac"])))

        # Parse restart steps
        restart_set = set()
        if cfg["enable_restarts"] and cfg["restart_steps"].strip():
            try:
                restart_set = {int(s.strip()) for s in cfg["restart_steps"].split(",") if s.strip()}
            except ValueError:
                pass

        self.profiler.start()

        def _model_fn(x_in, sigma_in, **kwargs):
            return model(x_in, sigma_in, **kwargs)

        verbose = (cfg["debug_mode"] != "0 - Silent")

        result = storm_sampler(
            model_fn=_model_fn,
            x=x,
            sigmas=sigmas,
            stiffness_threshold=cfg["stiffness_threshold"],
            hysteresis_margin=cfg["hysteresis_margin"],
            ema_alpha=cfg["ema_alpha"],
            cache_depth=cfg["cache_depth"],
            rk_order=cfg["rk_order"],
            calib_frac=cfg["calib_frac"],
            adaptive_sub_step=cfg["adaptive_sub_step"],
            sub_step_threshold=cfg["sub_step_threshold"],
            sub_step_max_depth=cfg["sub_step_max_depth"],
            look_back_enabled=cfg["look_back_enabled"],
            look_back_lambda=cfg["look_back_lambda"],
            look_back_snr_power=cfg["look_back_snr_power"],
            enable_restarts=cfg["enable_restarts"],
            restart_steps=restart_set,
            restart_noise_scale=cfg["restart_noise_scale"],
            restart_s_noise=cfg["restart_s_noise"],
            restart_seed=cfg["restart_seed"],
            restart_flush_cache=cfg["restart_flush_cache"],
            restart_aligned_noise=cfg["restart_aligned_noise"],
            force_pure_euler=cfg["force_pure_euler"],
            verbose=verbose,
            extra_args=extra_args,
            callback=callback,
            profiler=self.profiler,
        )

        self.profiler.stop()

        if cfg["enable_profiling"] or verbose:
            self.profiler.print_report(verbose=(cfg["debug_mode"] == "2 - Verbose"))

        # Save plot to disk (DAG-safe -- runs after sampling, inside closure)
        if cfg["plot_trajectory"]:
            plot = build_trajectory_plot(self.profiler.step_log, cfg, self._calib_steps)
            if plot is not None:
                try:
                    import folder_paths
                    from PIL import Image
                    arr      = (plot.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    img      = Image.fromarray(arr)
                    out_dir  = folder_paths.get_output_directory()
                    filename = f"STORM_telemetry_{int(time.time())}.png"
                    out_path = os.path.join(out_dir, filename)
                    img.save(out_path)
                    print(f"[MD_STORM] 📊 Plot saved → {out_path}")
                except Exception as e:
                    print(f"[MD_STORM] Plot save failed: {e}")

        return result


# =============================================================================
# MAIN NODE
# =============================================================================

class MD_STORM_Sampler:
    """
    MD: STORM Hybrid Solver v2.0 ⚡

    HPC-grade adaptive ODE sampler for flow-matching audio diffusion.

    STORK (Stabilized Taylor RK) handles stiff early steps.
    DPM++3M handles smooth late steps.
    Auto-switches per step based on velocity field curvature.

    Features:
    • RK2/3/4/5 adaptive multi-order (single NFE, cached derivatives)
    • Adaptive sub-stepping — splits step on U-turn detection
    • Look-Back SNR smoother — suppresses harmonic hum
    • SDE restarts — isotropic or velocity-aligned Langevin noise
    • Full telemetry plot saved to ComfyUI output/ folder
    • YAML input/template/params outputs

    Validated configs:
        35-step simple:       look_back_lambda=0.35, snr_power=1.5
        25-step ddim_uniform: look_back_lambda=0.55, snr_power=1.3
    """

    CATEGORY     = "MDMAchine/samplers"
    RETURN_TYPES = ("SAMPLER", "STRING",      "STRING")
    RETURN_NAMES = ("sampler", "yaml_params", "yaml_template")
    FUNCTION     = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Core ──
                "stiffness_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 0.50, "step": 0.01,
                    "tooltip": "Base stiffness threshold. Auto-calibrates to schedule.",
                }),
                "hysteresis_margin": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.20, "step": 0.01,
                    "tooltip": "Dead-band preventing STORK/DPM++ thrashing.",
                }),
                "ema_alpha": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA weight on stiffness observation.",
                }),
                "cache_depth": ("INT", {
                    "default": 5, "min": 3, "max": 6, "step": 1,
                    "tooltip": "Velocity history depth. Must be >= rk_order for full order.",
                }),
                "rk_order": (["auto", "2", "3", "4", "5"], {
                    "default": "auto",
                    "tooltip": (
                        "RK integration order. All orders are single NFE.\n"
                        "auto = highest order cache supports each step.\n"
                        "RK2→RK3→RK4→RK5 as cache fills during first steps."
                    ),
                }),
                # ── Sub-stepping ──
                "adaptive_sub_step": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Split step on U-turn detection. HPC manifold fracture defense.",
                }),
                "sub_step_threshold": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05,
                    "tooltip": "cos_sim below this triggers sub-step. 0.0=U-turns only.",
                }),
                # ── Look-Back ──
                "look_back_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "SNR-adaptive smoother. Suppresses harmonic hum.",
                }),
                "look_back_lambda": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 0.80, "step": 0.05,
                    "tooltip": "Smoothing weight. 0.35@35-step. 0.55@25-step ddim_uniform.",
                }),
                "look_back_snr_power": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "SNR falloff. 1.5@35-step. 1.3@25-step. Higher=faster fade.",
                }),
                # ── Restarts ──
                "enable_restarts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "SDE restart injection. Adds trajectory diversity.",
                }),
                "restart_steps": ("STRING", {
                    "default": "10,20",
                    "tooltip": "Comma-separated step indices to restart at.",
                }),
                "restart_aligned_noise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Velocity-aligned Langevin noise. Preserves groove/tempo, "
                        "glitches HF only. False=isotropic (standard)."
                    ),
                }),
                # ── Debug ──
                "force_pure_euler": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bypass STORM. Pure Euler for A/B testing.",
                }),
                "plot_trajectory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save 5-panel telemetry plot to ComfyUI output/ folder.",
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-step analytics report to console.",
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                }),
                "yaml_settings_str": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "YAML OVERRIDE — paste yaml_template output here.\n"
                        "UI values are defaults. YAML takes precedence.\n"
                        "Example: look_back_lambda: 0.40"
                    ),
                }),
            }
        }

    def get_sampler(
        self,
        stiffness_threshold, hysteresis_margin, ema_alpha, cache_depth, rk_order,
        adaptive_sub_step, sub_step_threshold,
        look_back_enabled, look_back_lambda, look_back_snr_power,
        enable_restarts, restart_steps, restart_aligned_noise,
        force_pure_euler, plot_trajectory, enable_profiling, debug_mode,
        yaml_settings_str,
    ):
        if storm_sampler is None:
            print("[MD_STORM] WARNING: storm_sampler_core not found. Place in core/.")

        cfg = {
            "stiffness_threshold":  stiffness_threshold,
            "hysteresis_margin":    hysteresis_margin,
            "ema_alpha":            ema_alpha,
            "cache_depth":          cache_depth,
            "rk_order":             rk_order,
            "calib_frac":           0.12,
            "adaptive_sub_step":    adaptive_sub_step,
            "sub_step_threshold":   sub_step_threshold,
            "sub_step_max_depth":   2,
            "look_back_enabled":    look_back_enabled,
            "look_back_lambda":     look_back_lambda,
            "look_back_snr_power":  look_back_snr_power,
            "enable_restarts":      enable_restarts,
            "restart_steps":        restart_steps,
            "restart_noise_scale":  0.5,
            "restart_s_noise":      1.0,
            "restart_seed":         42,
            "restart_flush_cache":  True,
            "restart_aligned_noise": restart_aligned_noise,
            "force_pure_euler":     force_pure_euler,
            "plot_trajectory":      plot_trajectory,
            "enable_profiling":     enable_profiling,
            "debug_mode":           debug_mode,
        }

        # YAML override
        if yaml_settings_str and yaml_settings_str.strip() and _YAML_AVAILABLE:
            try:
                overrides = yaml.safe_load(yaml_settings_str)
                if isinstance(overrides, dict):
                    _KEY_MAP = {
                        "stiffness_threshold":  float,
                        "hysteresis_margin":    float,
                        "ema_alpha":            float,
                        "cache_depth":          int,
                        "rk_order":             str,
                        "calib_frac":           float,
                        "adaptive_sub_step":    bool,
                        "sub_step_threshold":   float,
                        "sub_step_max_depth":   int,
                        "look_back_enabled":    bool,
                        "look_back_lambda":     float,
                        "look_back_snr_power":  float,
                        "enable_restarts":      bool,
                        "restart_steps":        str,
                        "restart_noise_scale":  float,
                        "restart_s_noise":      float,
                        "restart_seed":         int,
                        "restart_flush_cache":  bool,
                        "restart_aligned_noise": bool,
                        "force_pure_euler":     bool,
                        "plot_trajectory":      bool,
                        "enable_profiling":     bool,
                        "debug_mode":           str,
                    }
                    for k, cast in _KEY_MAP.items():
                        if k in overrides:
                            try:    cfg[k] = cast(overrides[k])
                            except: print(f"[MD_STORM] YAML: bad value for {k}")
            except Exception as e:
                print(f"[MD_STORM] YAML parse error: {e}")

        bridge  = STORMSamplerBridge(cfg)
        sampler = comfy.samplers.KSAMPLER(bridge.sample)

        return (sampler, build_yaml_params(cfg), YAML_TEMPLATE_STORM)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "MD_STORM_Sampler": MD_STORM_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_STORM_Sampler": "MD: STORM Hybrid Solver ⚡",
}
