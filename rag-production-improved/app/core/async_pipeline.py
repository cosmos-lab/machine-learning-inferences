# Import asyncio to run blocking CPU/GPU work in async FastAPI
import asyncio

# Import the synchronous RAG pipeline
from app.core.pipeline import RAGPipeline


# AsyncRAGPipeline inherits everything from RAGPipeline
# but adds async support for FastAPI
class AsyncRAGPipeline(RAGPipeline):

    # Async version of answer()
    async def a_answer(self, question: str, filters: dict = None) -> str:

        # Get current FastAPI event loop
        loop = asyncio.get_running_loop()

        # Retrieval step runs in separate thread pool
        # because FAISS + embedding model is blocking CPU/GPU work
        context = await loop.run_in_executor(
            None,   # None → default ThreadPoolExecutor
            lambda: self.retriever.retrieve(question, filters=filters),
        )

        # If nothing retrieved → return fallback response
        if not context:
            return "No relevant information found."

        # Generation step also runs in thread pool
        # because transformer.generate() is blocking
        return await loop.run_in_executor(
            None,
            self.generator.generate,
            question,
            context,
        )
