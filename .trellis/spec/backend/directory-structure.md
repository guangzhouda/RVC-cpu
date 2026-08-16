# Directory Structure

> How the RVC project is organized.

---

## Top-Level Layout

```
Retrieval-based-Voice-Conversion-WebUI/
├── infer/                 # Core inference engine
│   ├── lib/               # Shared utilities
│   │   ├── audio.py       # Audio loading (ffmpeg + av)
│   │   ├── rmvpe.py       # RMVPE pitch extraction (24KB)
│   │   ├── rtrvc.py       # Real-time RVC pipeline (17KB)
│   │   ├── slicer2.py     # Audio slicer
│   │   ├── infer_pack/    # Model architecture definitions
│   │   ├── train/         # Training utilities (process_ckpt, mel_processing)
│   │   ├── uvr5_pack/     # UVR5 model weights & layers
│   │   └── jit/           # JIT-compiled modules
│   └── modules/           # High-level feature modules
│       ├── vc/            # Voice conversion (VC class, pipeline, utils)
│       └── uvr5/          # Ultimate Vocal Remover 5 (MDXNet, VR)
│
├── tools/                 # Utility scripts (model conversion, similarity, etc.)
├── configs/               # Model configurations
│   ├── config.json        # Main config
│   ├── config.py          # Config class
│   ├── v1/                # V1 model configs
│   └── v2/                # V2 model configs
├── assets/                # Static assets (UVR5 weights, etc.)
├── i18n/                  # Internationalization (i18n.py)
├── logs/                  # Application logs
├── docs/                  # Project documentation
│
├── infer-web.py           # Main Gradio web UI (65KB, current version)
├── gui_v1.py              # Legacy GUI (48KB)
├── api_231006.py          # REST API v1 (19KB)
├── api_240604.py          # REST API v2 (22KB)
│
├── pyproject.toml         # Poetry dependencies (Python ^3.9, torch 2.4.0)
├── poetry.lock            # Poetry lock file
├── requirements-*.txt     # Pip requirements per platform (amd, dml, ipex, win)
│
├── go-web.bat             # Windows launcher: web UI
├── go-realtime-gui.bat    # Windows launcher: realtime GUI
├── run.sh                 # Linux launcher
├── venv.sh                # Virtual env setup
└── Dockerfile             # Container build
```

---

## Module Conventions

### `infer/lib/` — Low-Level Utilities
- `audio.py` — ffmpeg-based audio loading, format conversion with `av`
- `rmvpe.py` — RMVPE pitch extraction algorithm
- `rtrvc.py` — Real-time RVC pipeline with streaming support
- `slicer2.py` — Audio slicing for processing long files
- `infer_pack/` — Model definitions (not organized by version)
- `train/` — Checkpoint processing (`process_ckpt.py`), mel spectrograms
- `uvr5_pack/` — UVR5 weights and network layers (multiple size variants)
- `jit/` — JIT-optimized kernels

### `infer/modules/` — High-Level Modules
- `vc/` — **Voice Conversion** core: `VC` class, `Pipeline`, utils
- `uvr5/` — **Ultimate Vocal Remover**: MDXNet, VR models

### `configs/` — Configuration
- Central `Config` class in `configs/config.py`
- Version-specific sub-configs in `v1/`, `v2/`

### Entry Points
- `infer-web.py` — Main entry (65KB, Gradio Blocks with tabs)
- `gui_v1.py` — Legacy version (keep for backward compat)
- `api_*.py` — REST API servers (FastAPI + uvicorn)
- `.bat` / `.sh` files — Quick launchers for different modes

## Naming Conventions

- Files: `snake_case.py` for utilities, `modules.py` for module entry points
- Classes: `PascalCase` (`VC`, `Config`, `Pipeline`)
- Functions: `snake_case` (`load_audio`, `change_choices`, `export_onnx`)
- Constants: Mixed — `UPPER_SNAKE` or just `snake_case`

## Where to Add New Code

| What | Where |
|------|-------|
| New voice conversion model | `infer/modules/vc/` |
| New audio preprocessing | `infer/lib/` |
| New Web UI feature | `infer-web.py` |
| New REST API endpoint | `api_240604.py` |
| New utility script | `tools/` |
| Translation | `i18n/` |
