import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


logger = logging.getLogger("uvicorn.error")


@contextmanager
def timed(operation: str, **fields: object) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        if details:
            logger.info("perf operation=%s elapsed_ms=%.1f %s", operation, elapsed_ms, details)
        else:
            logger.info("perf operation=%s elapsed_ms=%.1f", operation, elapsed_ms)
