import os
from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH
from app.observability.logger import logger

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH, force_rebuild=True)

    logger.info("build_artifacts_complete", extra={"data_path": DATA_PATH})
    print(f"Artifacts built and saved for {DATA_PATH}")



