# Quality Guidelines

> Code quality standards for the RVC project — documenting actual project norms.

---

## Tech Stack (from pyproject.toml)

| Component | Version | Purpose |
|-----------|---------|---------|
| torch + torchaudio | 2.4.0 | Deep learning |
| gradio | 3.34.0 | Web UI |
| fairseq | 0.12.2 | Hubert model |
| faiss-cpu | 1.7.3 | Feature index search |
| librosa | 0.9.1 | Audio analysis |
| soundfile | ≥0.12.1 | Audio file I/O |
| ffmpeg-python | ≥0.2.0 | Audio decoding |
| scikit-learn | ≥1.5.1 | Clustering (MiniBatchKMeans) |
| numpy | 1.23.5 | Numerical computing |
| numba | 0.56.4 | JIT compilation |
| pyworld | 0.3.2 | WORLD vocoder |
| praat-parselmouth | ≥0.4.2 | Pitch analysis |
| onnxruntime-gpu | ≥1.18.1 | ONNX model inference |
| fastapi + uvicorn | 0.88 / ≥0.21.1 | REST API |

## Dependency Management

**⚠️ Poetry lock + pip requirements duality:**
- `pyproject.toml` + `poetry.lock` — primary dependency management
- `requirements-*.txt` — pip-based platform-specific reqs (amd, dml, ipex, win)
- New deps go in `pyproject.toml` AND the relevant `requirements-*.txt`

## Module Organization

```
infer/
├── lib/                    # Shared utilities
│   ├── audio.py            # Audio loading (ffmpeg + av)
│   ├── rmvpe.py            # RMVPE pitch extraction (24KB)
│   ├── rtrvc.py            # Real-time RVC pipeline (17KB)
│   ├── slicer2.py          # Audio slicer
│   ├── infer_pack/         # Model architecture definitions
│   ├── train/              # Training utilities
│   └── uvr5_pack/          # UVR5 model weights & utils
└── modules/
    ├── vc/                 # Voice conversion (VC class)
    ├── uvr5/               # Ultimate Vocal Remover 5
    ├── train/              # Training UI modules
    ├── onnx/               # ONNX export utilities
    └── ipex/               # Intel IPEX optimizations
```

## Configuration Pattern

```python
# from configs/config.py
class Config:
    """Central config object loaded from configs/config.json"""
    ...
```

- Single `Config` class in `configs/config.py`
- Reads from `configs/config.json` and subdirectories
- Sub-configs: `configs/v1/`, `configs/v2/`

## Gradio Patterns

```python
# UI is defined in infer-web.py (65KB, primary) and gui_v1.py (48KB, legacy)
# Pattern: global logger, direct Gradio component binding

import gradio as gr
i18n = I18nAuto()  # Internationalization

with gr.Blocks() as app:
    with gr.Tab("音色转换"):
        ...
```

- **Two Gradio apps**: `infer-web.py` (current), `gui_v1.py` (legacy)
- i18n via `I18nAuto` from `i18n/i18n.py`
- Long monolithic Gradio block definitions

## Code Review Rules

From CONTRIBUTING.md:
1. No algorithm changes (unless fixing a code-level error/warning)
2. Focus PRs on: translations, WebUI improvements, infrastructure
3. Minimize changes
4. All PRs need `@RVC-Boss` approval

## Forbidden Patterns

| ❌ Pattern | Why |
|-----------|-----|
| New algorithm/model changes in PRs | CONTRIBUTING.md blocks them |
| `print()` for anything | Use `logger.info()` |
| Hardcoded paths with `\` | Use `os.path` or `pathlib` |
| Pinned GPU-only code | Check `torch.cuda.is_available()` |
