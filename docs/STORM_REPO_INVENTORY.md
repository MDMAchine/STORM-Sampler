# STORM Repo — File Inventory
*June 20, 2026 — Updated to match D:\Projects\STORM disk layout*
*VAE in separate repo. Trajectory Anchor in separate repo.*

---

## LOCAL FOLDER STRUCTURE (D:\Projects\STORM)

```
STORM/
├── core/
│   ├── storm_sampler_core.py            ← v2.1.0 — CHANGELOG FIXED
│   ├── storm_sampler_core.lua           ← v2.1.0 — clean
│   ├── storm_sampler_core.hpp           ← v2.1.0 — clean
│   └── storm_sampler_core - HotStep deployed.lua  ← v3.0.0 scragnog port
├── sampler/
│   ├── MD_STORM_Sampler.py              ← ComfyUI wrapper — clean
│   └── MD_LookBack_Smoother.py          ← standalone node — clean
├── docs/
│   ├── README_STORM_Sampler.md          ← rename to README.md on push
│   ├── STORM_White_Paper_v3.md          ← arXiv paper — clean
│   ├── STORM_REPO_INVENTORY.md          ← this file
│   ├── STORM_Technical_History_v2.1.md  ← ships in repo
│   ├── storm_file_header.txt            ← ships in repo
│   ├── STORM_Internal_Reference.md      ← PRIVATE — never repo
│   ├── PR_SCRAGNOG_FEATURES_BUGS.md     ← internal — don't ship
│   └── md_nodes_implementation_tracker.html  ← internal — don't ship
├── baks/                                ← old archives — don't ship
└── omni_scan_root.py                    ← utility — don't ship
```

---

## SHIPS IN REPO (MDMAchine/STORM-Sampler)

### Repo layout on push:

```
STORM-Sampler/
├── README.md                            ← from README_STORM_Sampler.md
├── LICENSE                              ← GPL v3 (create)
├── __init__.py                          ← ComfyUI node registration
├── storm_gist_header_v3_animated.svg    ← animated header
├── storm_file_header.txt
├── core/
│   ├── storm_sampler_core.py
│   ├── storm_sampler_core.lua
│   └── storm_sampler_core.hpp
├── samplers/
│   ├── __init__.py                      ← Python import init
│   ├── MD_STORM_Sampler.py
│   └── MD_LookBack_Smoother.py
├── hotstep/
│   └── storm_sampler_core_hotstep.lua   ← renamed from "- HotStep deployed"
├── docs/
│   ├── STORM_White_Paper_v3.md
│   └── STORM_Technical_History_v2_1.md
```

| File | Source | Status |
|---|---|---|
| `__init__.py` (root) | create | ⬜ ComfyUI node registration |
| `samplers/__init__.py` | create | ⬜ Python import init |
| `core/storm_sampler_core.py` | core/ | ✅ **CHANGELOG FIXED** — v2.1.1 entries added |
| `core/storm_sampler_core.lua` | core/ | ✅ clean — changelog was already correct |
| `core/storm_sampler_core.hpp` | core/ | ✅ clean — changelog was already correct |
| `hotstep/storm_sampler_core_hotstep.lua` | core/ | ✅ clean — rename on push (spaces in filename) |
| `samplers/MD_STORM_Sampler.py` | sampler/ | ✅ clean — BioKit stripped |
| `samplers/MD_LookBack_Smoother.py` | sampler/ | ✅ clean |
| `docs/STORM_White_Paper_v3.md` | docs/ | ✅ clean — all IP leaks fixed |
| `docs/STORM_Technical_History_v2_1.md` | docs/ | ✅ ships |
| `storm_file_header.txt` | docs/ | ✅ ships |
| `storm_gist_header_v3_animated.svg` | new | ✅ animated header — XML validated |
| `README.md` | docs/ | ✅ rename README_STORM_Sampler.md |
| `LICENSE` | create | ⬜ GPL v3 text — create on push |

---

## DO NOT SHIP (stays local)

| File | Reason |
|---|---|
| `docs/STORM_Internal_Reference.md` | PRIVATE. Trade secret. Thermodynamic stack references. |
| `nag_core.py` | PRIVATE. Patent pending. Not in this folder — lives in ComfyUI. |
| `docs/PR_SCRAGNOG_FEATURES_BUGS.md` | Internal coordination doc. |
| `docs/md_nodes_implementation_tracker.html` | Internal tracker. |
| `baks/` | Old .rar archives. Don't push. |
| `omni_scan_root.py` | Omni-Trainer utility. Wrong repo. |

---

## SEPARATE REPOS

### MDMAchine/MD-Trajectory-Anchor (future — after STORM launch)
- `md_trajectory_anchor_V2.lua` — step() solver, 13-stage latent stabilization pipeline
- Separate paper, separate repo, separate arXiv submission
- Shared component: look-back smoother formula (properly attributed arXiv:2602.09449)
- Verify V2 inertia EMA fix before launch

### MDMAchine/MD-Audio-VAE-Tiled (separate — waiting on scragnog HS-CPP confirmation)
- Everything `md_audio_tiled_*` lives there
- Not tracked in this inventory

---

## README HEADER

Use animated SVG v3 in README.md:

```markdown
![STORM](storm_gist_header_v3_animated.svg)
```

---

## PRE-PUSH CHECKLIST

- [x] `storm_sampler_core.py` changelog — v2.1.1 entries added
- [x] White paper v3.1 — all IP leaks fixed, NAG attribution, NFE qualifier
- [x] Animated SVG header — XML validated, CSS-only animation
- [x] MD_LookBack_Smoother — v1.1.0, lambda-near-zero fix
- [ ] Create `__init__.py` (root) — ComfyUI node registration
- [ ] Create `samplers/__init__.py` — Python import init
- [ ] Create GPL v3 LICENSE file
- [ ] Rename `storm_sampler_core - HotStep deployed.lua` → `storm_sampler_core_hotstep.lua`
- [ ] Rename `README_STORM_Sampler.md` → `README.md`
- [ ] Add `![STORM](storm_gist_header_v3_animated.svg)` to top of README
- [ ] Convert `STORM_White_Paper_v3.md` → PDF for arXiv
- [ ] Create repo (private)
- [ ] Push all files
- [ ] arXiv submit → get link
- [ ] Add arXiv URL to README
- [ ] Go public
- [ ] Ping scragnog + serveurperso with repo link
- [ ] Drop in ACE-Step Discord

---

## COMMUNITY ANNOUNCEMENT DRAFT

> STORM Sampler is now public.
> GPL v3 — ComfyUI node, C++ header-only, Lua (HOT-Step native).
> Fixes the metallic artifact in ACE-Step XL Turbo at the source.
> arXiv pre-print in the repo.
> scragnog has it running in HS-CPP. serveurperso has the hpp.
> → github.com/MDMAchine/STORM-Sampler

---

*Last updated: 2026-06-20*
