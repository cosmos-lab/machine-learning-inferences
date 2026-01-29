import faiss
from app.observability.logger import logger

def build_faiss_index(embeddings, use_gpu: bool = True, nlist: int = 128):
    dim = embeddings.shape[1]

    if embeddings.shape[0] < nlist:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(8, nlist)

    if use_gpu and faiss.get_num_gpus() > 0:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("faiss_gpu_enabled")
        except Exception:
            logger.warning("faiss_gpu_failed_fallback_cpu")

    return index


