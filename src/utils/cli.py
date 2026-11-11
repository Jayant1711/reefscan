"""Consistent, friendly error delivery for CLIs.

Real bugs still traceback; the common user errors (missing dataset/checkpoint,
bad flag values, CUDA OOM) become one clean line and exit code 1.

Usage:
    if __name__ == "__main__":
        raise SystemExit(run_cli(main))
"""

from __future__ import annotations

import logging


def run_cli(main_fn) -> int:
    """Run ``main_fn`` and translate expected user errors into clean messages."""
    logger = logging.getLogger("cli")
    try:
        return int(main_fn() or 0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except ModuleNotFoundError as exc:
        logger.error(
            "Missing dependency: %s — install requirements (pip install -r requirements.txt) "
            "or the optional extras named in the message.", exc,
        )
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            logger.error(
                "CUDA out of memory — lower the batch size, e.g. "
                "--set <task>.batch_size=16 (see docs/TRAINING.md)."
            )
            return 1
        raise  # a non-OOM RuntimeError is a real bug — let it traceback
