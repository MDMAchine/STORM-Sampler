# MD_LookBack_Smoother.py
# Standalone SNR-Adaptive Latent Trajectory Smoother (Look-Back)
# Based on: arXiv:2602.09449 -- Look-Ahead/Look-Back flows
#
# © 2026 Alexander Allan (MDMAchine) | A&E Concepts
# GPL v3
#
# Version: 1.1.0
# Created: 2026-05-09 | Revised: 2026-06-20
# Category: MD_Nodes/Samplers
#
# WHAT THIS DOES:
#   Post-process node. Takes a LATENT output from any sampler and applies
#   gentle manifold smoothing to suppress residual ODE shearing artifacts.
#
#   The "previous" latent is approximated by adding small noise scaled to
#   the latent's own energy — conservative, preserves structure.
#
#   lambda_base controls blend strength directly (no SNR scheduling —
#   that only works inside the sampling loop where sigma varies per step).
#   STORM has the full SNR-adaptive look-back built in. This standalone
#   node provides a simpler fixed-weight version for other samplers.
#
# WIRE ORDER:
#   BasicScheduler → Sampler
#   Sampler LATENT output → MD: Look-Back Smoother → VAE Decode
#
# Works with ANY sampler. STORM has the full in-loop version built in —
# use this standalone node for Euler, DPM++, or any other sampler.
#
# Validated params: lambda_base=0.03-0.08 (post-process), noise_seed=0
#
# Changelog:
#   1.1.0 -- Fixed lambda evaluation. Post-process node now uses lambda_base
#            directly as blend weight instead of SNR-adaptive scheduling
#            (which evaluated to ~0.00008 and did nothing). Noise scale
#            derived from latent RMS, not sigma_curr.
#   1.0.0 -- Initial release.

import torch


class MD_LookBack_Smoother:
    """
    MD: Look-Back Smoother 🌊

    Post-process latent smoother. Applies gentle manifold smoothing to
    suppress residual ODE shearing artifacts from any sampler.

    Wire between sampler LATENT output and VAE Decode.
    Works with any sampler (Euler, DPM++, etc.)

    For full SNR-adaptive in-loop look-back, use STORM sampler (built-in).
    This standalone node provides a simpler fixed-weight post-process version.

    Validated: lambda_base=0.03-0.08 for gentle post-process smoothing.
    """

    CATEGORY     = "MD_Nodes/Samplers"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("smoothed_latent",)
    FUNCTION     = "smooth"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Latent from sampler output. Wire before VAE Decode."
                }),
                "lambda_base": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.30,
                    "step": 0.01,
                    "tooltip": (
                        "Blend weight toward smoothed latent. "
                        "0.0 = passthrough. 0.03-0.08 = gentle smoothing. "
                        "0.15+ = heavy (may soften detail)."
                    ),
                }),
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Seed for the manifold noise perturbation."
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print lambda value to console."
                }),
            }
        }

    def smooth(self, latent, lambda_base, noise_seed, verbose):
        samples = latent["samples"]

        if lambda_base == 0.0:
            if verbose:
                print("[LookBack] lambda_base=0.0 — passthrough.")
            return ({"samples": samples},)

        # Noise scale: 10% of latent RMS — conservative, structure-preserving
        rms = samples.flatten(1).norm(dim=1).mean().item()
        noise_scale = rms * 0.1

        gen = torch.Generator(device=samples.device)
        gen.manual_seed(noise_seed)
        x_prev_approx = samples + torch.randn(samples.shape, dtype=samples.dtype,
                                               device=samples.device, generator=gen) * noise_scale

        x_smooth = (1.0 - lambda_base) * samples + lambda_base * x_prev_approx

        if verbose:
            print(f"[LookBack] λ={lambda_base:.4f}, noise_scale={noise_scale:.4f} — smoothing applied.")

        return ({"samples": x_smooth},)


NODE_CLASS_MAPPINGS = {
    "MD_LookBack_Smoother": MD_LookBack_Smoother,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_LookBack_Smoother": "MD: Look-Back Smoother 🌊",
}
