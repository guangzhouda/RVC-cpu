# Python Conventions

> Actual Python coding patterns in the RVC project (documenting reality, not ideals).

---

## Python Version

- **Python 3.9+** — project uses `python = "^3.9"` in pyproject.toml
- Uses modern f-strings, but NOT Python 3.10+ type hint syntax

## Imports

**Actual pattern from `infer-web.py`:**

```python
import os
import sys
from dotenv import load_dotenv
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
import traceback
import logging
```

**Conventions observed:**
- `import os, sys` at the top
- `from .submodule import X` for project modules
- Third-party imports mixed (torch, numpy, gradio, fairseq, faiss)
- **No `from X import *`** — this is good!
- Multi-line imports use parentheses for continuation

## Naming

| Element | Observed Pattern | Example |
|---------|-----------------|---------|
| Modules | snake_case | `process_ckpt.py`, `mel_processing.py` |
| Classes | PascalCase | `VC`, `Config`, `Pipeline` |
| Functions | snake_case | `load_audio()`, `change_choices()`, `export_onnx()` |
| Variables | snake_case | `sample_rate`, `f0_curve` |
| Constants | UPPER_SNAKE or plain | `SOME_CONST` or `some_config` |

## No Type Hints

**⚠️ The codebase does NOT use type hints.** This is the current reality.
When writing new code, follow the existing style:
- No type annotations on function signatures
- No return types
- Use docstrings for documentation when needed

## PyTorch Patterns

**Actual patterns from the codebase:**

```python
# ✅ GPU count check (from infer-web.py)
ngpu = torch.cuda.device_count()

# ✅ Direct .cuda() rather than .to(device)
model = model.cuda()

# ✅ No torch.no_grad() in inference code
# (present in some internal modules but not uniformly)

# ✅ Audio is numpy arrays, not tensors
# Audio I/O returns numpy via soundfile, librosa
```

**Device management patterns:**
- `torch.cuda.device_count()` for GPU count
- Direct `.cuda()` calls, not `model.to(device)` pattern
- `torch.load(path, map_location="cpu")` for model loading
- Check `platform.system()` for OS detection

## Class Structure

```python
# Main pattern: single large class (VC, Config)
class VC:
    def __init__(self, **kwargs):
        ...

    def get_block_name(self):
        ...

    def forward_dml(self, ctx, x, scale):
        ...
```

One large class per module, passing `**kwargs` in constructors.

## Audio Processing

```python
# Audio I/O uses ffmpeg + av (from infer/lib/audio.py)
def load_audio(file, sr):
    try:
        file = clean_path(file)
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

# Audio conversion uses av
def wav2(i, o, format):
    inp = av.open(i, "rb")
    out = av.open(o, "wb", format=format)
    ...
```

- `ffmpeg-python` for audio loading
- `av` for format conversion
- `soundfile` for writing
- Audio represented as numpy arrays, channels first `[1, T]`
