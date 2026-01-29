from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH
from app.observability.logger import logger

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH, force_rebuild=True)
    logger.info("build_artifacts_complete")


