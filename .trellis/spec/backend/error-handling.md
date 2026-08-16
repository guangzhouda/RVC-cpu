# Error Handling

> Actual error handling patterns observed in the RVC codebase.

---

## Dominant Pattern: `try/except:` + `traceback.print_exc()`

The most common pattern across the codebase is:

```python
try:
    # do something
    ...
except:
    traceback.print_exc()
```

**This is used for both expected and unexpected errors. No exception type filtering.**

## Examples from the Codebase

### Pattern 1: Bare except (most common — from infer-web.py and vc/modules.py)

```python
# Line ~642: File loading
try:
    big_npy = (
        ...
    )
except:
    traceback.print_exc()

# Line ~690: OS-specific operations
try:
    link = os.link if platform.system() == "Windows" else os.symlink
except:
    traceback.print_exc()

# Line ~785: Config loading
try:
    with open(...) as f:
        ...
except:
    traceback.print_exc()
```

### Pattern 2: Exception with re-raise (audio.py)

```python
# From infer/lib/audio.py - the better pattern (but rare)
try:
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
except Exception as e:
    traceback.print_exc()
    raise RuntimeError(f"Failed to load audio: {e}")
```

### Pattern 3: VC module import (from vc/modules.py)

```python
import traceback
import logging
logger = logging.getLogger(__name__)

# Error handling with logging + traceback
try:
    ...
except:
    logger.error("Operation failed")
    traceback.print_exc()
```

---

## Key Observations

| Observed Pattern | Prevalence |
|-----------------|------------|
| `except:` (bare except) | ⭐⭐⭐⭐⭐ Very common |
| `except Exception as e:` (qualified) | ⭐⭐ Rare |
| `traceback.print_exc()` | ⭐⭐⭐⭐⭐ Almost always paired |
| `logger.error(...)` | ⭐⭐⭐ Sometimes added |
| `raise RuntimeError(...)` | ⭐⭐ Rare (mostly in audio.py) |
| `gr.Error(...)` for Gradio UI | ⭐ Seen in UI handlers |

## Summary

When writing code for this project:
1. Use `try/except:` with `traceback.print_exc()` — this is the project norm
2. Add `logger.error(...)` for important error paths
3. Use `gr.Error()` / `gr.Warning()` when returning errors to the Gradio UI
4. Only use specific exceptions when you have a clear recovery path
5. For fatal errors, `raise RuntimeError(...)` from `except` block
