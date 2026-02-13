import faiss                      # Facebook AI Similarity Search library for fast vector similarity search
from app.observability.logger import logger   # Custom logger for observability / monitoring

# Function to build FAISS index from embedding matrix
# embeddings -> numpy array of shape (num_docs, embedding_dim)
# use_gpu -> whether to move FAISS index to GPU
# nlist -> number of IVF clusters (used in ANN search)
def build_faiss_index(embeddings, use_gpu: bool = True, nlist: int = 128):

    dim = embeddings.shape[1]     # Extract embedding dimension (vector size)

    # If dataset is very small then IVF is useless because clustering will be poor
    # In that case we fallback to Flat Index (Exact Search)
    if embeddings.shape[0] < nlist:
        index = faiss.IndexFlatIP(dim)   # Exact nearest neighbour search using Inner Product
        index.add(embeddings)            # Add all document embeddings into index
        
        # 👉 VECTOR DB INSERT POINT:
        # Instead of:
        #       index.add(embeddings)
        #
        # You can store embeddings in vector DB here like:
        #       vector_db.insert(embeddings)
        #
        # Example later:
        #       Pinecone / Milvus / Qdrant / Weaviate etc.
        
        return index

    # Quantizer is mandatory for IVF index
    # It acts as coarse cluster center finder
    quantizer = faiss.IndexFlatIP(dim)

    # IVF = Inverted File Index (Approximate Nearest Neighbour Search)
    # Faster than flat search for large dataset
    index = faiss.IndexIVFFlat(
        quantizer,                 # cluster builder
        dim,                       # vector dimension
        nlist,                     # number of clusters
        faiss.METRIC_INNER_PRODUCT # similarity metric (cosine if normalized)
    )

    # Train IVF clusters before inserting vectors
    index.train(embeddings)

    # Add embeddings into trained index
    index.add(embeddings)

    # 👉 VECTOR DB INSERT POINT:
    # THIS IS YOUR BEST PLACE TO MOVE TO VECTOR DB
    #
    # Instead of:
    #       index.add(embeddings)
    #
    # You should:
    #       vector_db.upsert(embeddings, metadata)
    #
    # Because:
    #       Retrieval pipeline logic stays same
    #       Only storage backend changes
    #
    # Retrieval later becomes:
    #       vector_db.search(query_embedding, top_k)

    # nprobe controls how many clusters are searched during retrieval
    # Higher = better accuracy but slower search
    index.nprobe = min(8, nlist)

    # Move FAISS index to GPU if available
    if use_gpu and faiss.get_num_gpus() > 0:
        try:
            res = faiss.StandardGpuResources()   # GPU memory manager
            index = faiss.index_cpu_to_gpu(res, 0, index)  # Move CPU index to GPU
            logger.info("faiss_gpu_enabled")
        except Exception:
            # If GPU fails fallback to CPU silently
            logger.warning("faiss_gpu_failed_fallback_cpu")

    # 👉 VECTOR DB NOTE:
    #
    # GPU transfer becomes unnecessary if using external Vector DB
    # because DB manages hardware scaling internally
    #
    # So this whole GPU block is removed when migrating to:
    #       Milvus / Pinecone / Weaviate / Qdrant etc.

    return index
