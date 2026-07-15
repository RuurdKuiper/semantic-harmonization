import logging
import sys

_CONFIGURED = False

# Silence noisy third-party libraries (sentence-transformers/httpx/huggingface_hub
# log INFO-level HTTP requests and an "unauthenticated requests" warning on every
# model load) — none of this is actionable for local, cached model usage.
_QUIET_LOGGERS = ("sentence_transformers", "httpx", "huggingface_hub")


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )
        for quiet_logger in _QUIET_LOGGERS:
            logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        try:
            # huggingface_hub emits its "unauthenticated requests" notice through
            # its own verbosity control rather than a plain logger level, so the
            # logger.setLevel above alone doesn't always suppress it.
            from huggingface_hub.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except ImportError:
            pass
        _CONFIGURED = True
    return logging.getLogger(name)
