# Logging Guidelines

> Actual logging patterns observed in the RVC codebase.

---

## Setup

All modules use the standard library `logging` module:

```python
import logging
logger = logging.getLogger(__name__)
```

No structured logging frameworks (no loguru, no structlog).

## Log Levels

| Observed Usage | Level | Example |
|---------------|-------|---------|
| ✅ Most common | `logger.info(...)` | `logger.info("Execute: " + cmd)` |
| ✅ For errors | `logger.error(...)` | `logger.error("Model load failed")` |
| ⚠️ Rare | `logger.warning(...)` | Used sparingly |
| ❌ Not used | `logger.debug(...)` | Not found in codebase |

## Examples from the Codebase

```python
# Simple string messages (from infer-web.py)
logger.info(i18n("正在加载模型..."))           # i18n key
logger.info("Execute: " + cmd)              # f-string would be cleaner
logger.info(log)                            # pre-formatted string

# Error logging (from VC module)
logger.error(f"Failed to load {model_name}")
```

## Rules for New Code

1. Use `logger.info()` for progress messages
2. Use `logger.error()` + `traceback.print_exc()` for error paths
3. String concatenation with `+` is acceptable (matching existing style)
4. f-strings are fine too (also used in places)
5. Don't use `logger.debug()` — not part of the project's logging culture
6. Don't log in hot inference loops

## Unused Patterns

These are NOT used in this project (don't add them):
- ❌ Structured logging (JSON logs, key-value pairs)
- ❌ `logger.debug()` level logging
- ❌ `exc_info=True` parameter
- ❌ Custom log formatters
