import time
from contextlib import contextmanager
from app.observability.logger import logger

@contextmanager
def track(stage: str):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "latency",
        extra={"stage": stage, "latency_ms": round(elapsed, 2)},
    )


