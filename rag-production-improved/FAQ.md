
---

## Feature Explanation

### **Chunking**
**What it does:**
- Splits large documents into smaller, semantically meaningful pieces
- Preserves context by keeping related content together
- Adds overlap between chunks to maintain continuity

**Why it's important:**
- **Better retrieval**: Smaller chunks = more precise matching
- **Token limits**: LLMs have input limits, chunks fit better
- **Semantic coherence**: Keeps paragraphs/sentences together instead of random splits

**3 Strategies available:**
1. **Semantic** (default): Splits on paragraphs → sentences → words
2. **Recursive**: Hierarchical splitting with multiple separators
3. **Sentence**: Simple sentence-based chunking

**Configuration:**
```bash
export CHUNKING_STRATEGY=semantic  # or "recursive" or "sentence"
export CHUNK_SIZE=512              # Max characters per chunk
export CHUNK_OVERLAP=128           # Overlap between chunks
```

---

### **Metadata Filtering**
**What it does:**
- Attaches metadata to each chunk (source file, chunk ID, size, etc.)
- Allows filtering search results based on metadata criteria
- Enables targeted retrieval from specific sources or chunk types

**Why it's important:**
- **Multi-document RAG**: Filter by specific documents when you have many
- **Source attribution**: Know which document answers came from
- **Quality control**: Filter by chunk size, date, or custom attributes
- **User permissions**: Filter by access rights or document categories

**Metadata stored per chunk:**
```json
{
  "chunk_id": 0,
  "source": "data/doc1.txt",
  "chunk_size": 487,
  "strategy": "semantic"
}
```

**Filter operators supported:**
- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$lt`: Less than
- `$lte`: Less than or equal
- `$in`: Value in list

**Example use cases:**

1. **Filter by source:**
```json
{
  "q": "What is machine learning?",
  "filters": {"source": "data/ml_book.txt"}
}
```

2. **Filter by chunk size (quality control):**
```json
{
  "q": "Explain neural networks",
  "filters": {"chunk_size": {"$gte": 200, "$lte": 600}}
}
```

3. **Multiple filters:**
```json
{
  "q": "What is deep learning?",
  "filters": {
    "source": "data/doc1.txt",
    "strategy": "semantic",
    "chunk_size": {"$gte": 300}
  }
}
```

