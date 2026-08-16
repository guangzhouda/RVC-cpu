# RVC Python Development Guidelines

> Python conventions for the Retrieval-based Voice Conversion project.

---

## Overview

This project implements RVC (Retrieval-based Voice Conversion) — a voice conversion system using:
- **PyTorch 2.4** for deep learning inference
- **Gradio 3.34** for the web interface
- **Fairseq** (Hubert) for speech feature extraction
- **Faiss** for feature index search
- **Various audio libraries** (librosa, soundfile, ffmpeg-python, pyworld, praat-parselmouth)

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Project layout: infer/, tools/, configs/ | ✅ Filled |
| [Python Conventions](./python-conventions.md) | Imports, naming, PyTorch patterns (no type hints!) | ✅ Filled |
| [Error Handling](./error-handling.md) | try/except + traceback.print_exc() pattern | ✅ Filled |
| [Quality Guidelines](./quality-guidelines.md) | Tech stack, gradio patterns, forbidden ops | ✅ Filled |
| [Logging Guidelines](./logging-guidelines.md) | logger.info() / logger.error() conventions | ✅ Filled |

---

## Quick Rules

1. **Python 3.9+** — no Python 3.10+ type hints in this project
2. **No type annotations** — follow existing convention (docstring, no `: type`)
3. **`try/except:` + `traceback.print_exc()`** — dominant error handling pattern
4. **`logger.info()`** — primary logging; no debug logging culture
5. **Gradio 3.34** — UI components in `infer-web.py` (primary) and `gui_v1.py` (legacy)
6. **Poetry + pip** — deps in both `pyproject.toml` and `requirements-*.txt`
7. **No algorithm changes** — CONTRIBUTING.md blocks them; focus on UI/translation/infra

---

## Pre-Development Checklist

Before writing code:
- [ ] Read [Directory Structure](./directory-structure.md) — know where files go
- [ ] Read [Python Conventions](./python-conventions.md) — coding style
- [ ] Read [Error Handling](./error-handling.md) — error pattern to use
- [ ] Read [Quality Guidelines](./quality-guidelines.md) — tech stack and forbidden patterns
- [ ] Check `configs/config.py` for relevant Config options
