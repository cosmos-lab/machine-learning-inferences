from app.config.settings import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    LANGFUSE_ENABLED,
)
from app.observability.logger import logger

# Singleton Langfuse client
# Safely disabled if keys are not configured
langfuse = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse
        logger.info(f"langfuse_connecting: host={LANGFUSE_HOST} key={LANGFUSE_PUBLIC_KEY[:10]}...")
        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        # test flush immediately on startup
        trace = langfuse.trace(name="startup-test", input={"status": "starting"})
        langfuse.flush()
        logger.info("langfuse_initialized_and_flushed", extra={"host": LANGFUSE_HOST})
    except Exception as e:
        logger.warning(f"langfuse_init_failed: {e}")
        langfuse = None
else:
    logger.warning("langfuse_disabled: no keys configured")