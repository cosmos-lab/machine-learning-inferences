import asyncio
from app.core.pipeline import RAGPipeline

class AsyncRAGPipeline(RAGPipeline):
    async def a_answer(self, question: str, filters: dict = None) -> str:
        loop = asyncio.get_running_loop()

        context = await loop.run_in_executor(
            None,
            lambda: self.retriever.retrieve(question, filters=filters),
        )

        if not context:
            return "No relevant information found."

        return await loop.run_in_executor(
            None,
            self.generator.generate,
            question,
            context,
        )


