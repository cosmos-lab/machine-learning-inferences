# Import asyncio to run blocking CPU/GPU work in async FastAPI
import asyncio

# Import the synchronous RAG pipeline
from app.core.pipeline import RAGPipeline

# Langfuse client for LLM tracing
from app.observability.langfuse_client import langfuse

# Structured logger for observability
from app.observability.logger import logger

# Load config values needed for Langfuse metadata
from app.config.settings import TOP_K, EMBED_MODEL, GEN_MODEL


# AsyncRAGPipeline inherits everything from RAGPipeline
# but adds async support for FastAPI
class AsyncRAGPipeline(RAGPipeline):

    # Async version of answer()
    async def a_answer(self, question: str, filters: dict = None) -> str:

        # Get current FastAPI event loop
        loop = asyncio.get_running_loop()

        # Compute semantic drift before retrieval
        # measures how far query is from knowledge base centroid
        drift_score = self.retriever.compute_drift(question)
        logger.info("semantic_drift", extra={"query": question, "drift_score": drift_score})

        # Warn if query is drifting outside knowledge base
        if drift_score > 0.5:
            logger.warning("high_semantic_drift", extra={"query": question, "drift_score": drift_score})

        # Start Langfuse trace for the full RAG query
        trace = None
        if langfuse:
            try:
                trace = langfuse.trace(
                    name="rag-query",
                    input={"question": question, "filters": filters},
                    metadata={
                        "top_k": TOP_K,
                        "embed_model": EMBED_MODEL,
                        "gen_model": GEN_MODEL,
                        "semantic_drift": drift_score,
                        "high_drift": drift_score > 0.5,
                    },
                )
                logger.info(f"langfuse_trace_created: {trace.id}")
            except Exception as e:
                logger.warning(f"langfuse_trace_failed: {e}")

        # Retrieval step runs in separate thread pool
        # because FAISS + embedding model is blocking CPU/GPU work
        retrieval_span = None
        if trace:
            try:
                retrieval_span = trace.span(name="retrieval", input={"question": question})
            except Exception as e:
                logger.warning(f"langfuse_retrieval_span_failed: {e}")

        context = await loop.run_in_executor(
            None,   # None → default ThreadPoolExecutor
            lambda: self.retriever.retrieve(question, filters=filters),
        )

        if retrieval_span:
            try:
                retrieval_span.end(output={"chunks_retrieved": len(context)})
            except Exception as e:
                logger.warning(f"langfuse_retrieval_span_end_failed: {e}")

        # If nothing retrieved → return fallback response
        if not context:
            if trace:
                try:
                    trace.update(output={"answer": "No relevant information found."})
                    langfuse.flush()
                except Exception as e:
                    logger.warning(f"langfuse_flush_failed: {e}")
            return "No relevant information found."

        # Generation step also runs in thread pool
        # because transformer.generate() is blocking
        generation_span = None
        if trace:
            try:
                generation_span = trace.generation(
                    name="generation",
                    model=GEN_MODEL,
                    input={"question": question, "context": context},
                )
            except Exception as e:
                logger.warning(f"langfuse_generation_span_failed: {e}")

        answer = await loop.run_in_executor(
            None,
            self.generator.generate,
            question,
            context,
        )

        if generation_span:
            try:
                generation_span.end(output={"answer": answer})
            except Exception as e:
                logger.warning(f"langfuse_generation_span_end_failed: {e}")

        # Update trace with final answer and flush immediately
        if trace:
            try:
                trace.update(output={"answer": answer})
                # Add drift as a Langfuse score — shows in Scores dashboard
                langfuse.score(
                    trace_id=trace.id,
                    name="semantic_drift",
                    value=drift_score,
                    comment="high drift" if drift_score > 0.5 else "normal",
                )
                langfuse.flush()
            except Exception as e:
                logger.warning(f"langfuse_flush_failed: {e}")

        return answer