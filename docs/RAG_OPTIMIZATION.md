# RAG Optimization Guide

Complete guide to optimizing Retrieval-Augmented Generation (RAG) for better accuracy and performance.

## What is RAG?

RAG enhances AI responses by:
1. **Retrieving** relevant documents from your knowledge base
2. **Augmenting** the prompt with that context
3. **Generating** an answer based on your documents

This prevents hallucinations and grounds responses in your data.

---

## Quick Configuration

### Basic Settings (.env)

```bash
# Number of documents to retrieve
RAG_TOP_K=5

# Minimum similarity score (0.0 to 1.0)
RAG_SCORE_THRESHOLD=0.7

# Document chunking
CHUNK_SIZE=1000          # Characters per chunk
CHUNK_OVERLAP=200        # Overlap between chunks

# Embedding model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## Understanding Key Parameters

### 1. RAG_TOP_K (Default: 5)

**What it does**: Number of most relevant chunks to retrieve

**When to adjust**:

| Use Case | Recommended K | Reason |
|----------|---------------|--------|
| Simple Q&A | 3-5 | Less context needed |
| Complex analysis | 7-10 | More context helps |
| Summarization | 10-15 | Need broad coverage |
| Fact lookup | 1-3 | Precise answer needed |

**Example**:
```bash
# For simple questions
RAG_TOP_K=3

# For complex research
RAG_TOP_K=10
```

**Trade-offs**:
- ⬆️ Higher K = More context, better answers, higher cost, slower
- ⬇️ Lower K = Faster, cheaper, but may miss relevant info

### 2. RAG_SCORE_THRESHOLD (Default: 0.7)

**What it does**: Minimum similarity score (0.0-1.0) for a chunk to be included

**Similarity scores**:
- **0.9-1.0**: Nearly identical text
- **0.7-0.9**: Highly relevant
- **0.5-0.7**: Somewhat relevant
- **< 0.5**: Likely not relevant

**When to adjust**:

```bash
# Strict relevance (fewer but more accurate results)
RAG_SCORE_THRESHOLD=0.8

# More lenient (more results, some may be tangential)
RAG_SCORE_THRESHOLD=0.6

# Very strict (only near-exact matches)
RAG_SCORE_THRESHOLD=0.9
```

**Example scenarios**:
- **Legal/medical**: Use 0.8+ (precision critical)
- **General knowledge**: Use 0.7 (balanced)
- **Broad research**: Use 0.6 (recall matters more)

### 3. CHUNK_SIZE (Default: 1000)

**What it does**: Maximum characters per document chunk

**When to adjust**:

| Document Type | Recommended Size | Reason |
|---------------|------------------|--------|
| Short FAQs | 500 | Keep Q&A together |
| Articles | 1000 | Good balance |
| Technical docs | 1500 | Preserve context |
| Code files | 2000 | Keep functions together |

**Example**:
```bash
# For short-form content (tweets, FAQs)
CHUNK_SIZE=500

# For long-form content (articles, docs)
CHUNK_SIZE=1500
```

**Trade-offs**:
- ⬆️ Larger chunks = Better context, but less precise matching
- ⬇️ Smaller chunks = More precise, but may lose context

### 4. CHUNK_OVERLAP (Default: 200)

**What it does**: Characters that overlap between consecutive chunks

**Purpose**: Prevents important info from being split across chunks

**Example**:
```bash
# Minimal overlap (faster indexing, less storage)
CHUNK_OVERLAP=100

# More overlap (better context preservation)
CHUNK_OVERLAP=300
```

**Rule of thumb**: Set overlap to 20-30% of chunk size

```bash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200  # 20% overlap
```

---

## Advanced RAG Techniques

### 1. Hybrid Search (Semantic + Keyword)

Combine vector search with traditional keyword search:

**Update `app/rag.py`:**

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def hybrid_search(
    query: str,
    top_k: int = 5,
    keywords: list[str] = None
) -> list[dict]:
    """Combine semantic and keyword search."""

    # Get semantic results
    semantic_results = await search_documents(query, top_k=top_k*2)

    # Filter by keywords if provided
    if keywords:
        keyword_filtered = [
            doc for doc in semantic_results
            if any(kw.lower() in doc["content"].lower() for kw in keywords)
        ]
        return keyword_filtered[:top_k]

    return semantic_results[:top_k]
```

**Usage**:
```python
results = await hybrid_search(
    query="Python error handling",
    keywords=["try", "except", "error"]
)
```

### 2. Re-ranking Retrieved Documents

Improve relevance by re-ranking results:

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder model (one-time setup)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """Re-rank documents using cross-encoder."""

    # Score each doc against query
    pairs = [[query, doc["content"]] for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked]
```

### 3. Query Expansion

Expand user query to capture more relevant docs:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def expand_query(query: str) -> list[str]:
    """Generate alternative phrasings of the query."""

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Generate 3 alternative phrasings of the user's question."
        }, {
            "role": "user",
            "content": query
        }],
        temperature=0.7
    )

    alternatives = response.choices[0].message.content.split("\n")
    return [query] + alternatives

# Use it
queries = await expand_query("What is Python error handling?")
# ["What is Python error handling?",
#  "How does exception handling work in Python?",
#  "What are try-except blocks in Python?"]

# Search with all variations
all_results = []
for q in queries:
    results = await search_documents(q, top_k=3)
    all_results.extend(results)

# Deduplicate and take top K
unique_results = list({doc["id"]: doc for doc in all_results}.values())
```

### 4. Contextual Chunk Headers

Add metadata to chunks for better retrieval:

**Update `app/utils/chunking.py`:**

```python
def chunk_with_headers(
    text: str,
    chunk_size: int,
    metadata: dict
) -> list[dict]:
    """Add contextual headers to chunks."""

    chunks = recursive_character_chunking(text, chunk_size)

    # Add header to each chunk
    header = f"Document: {metadata.get('filename', 'unknown')}\n"
    header += f"Section: {metadata.get('section', 'N/A')}\n\n"

    enhanced_chunks = []
    for chunk in chunks:
        enhanced_chunks.append({
            "content": header + chunk,
            "metadata": metadata
        })

    return enhanced_chunks
```

**Benefits**: Agent knows which document each chunk came from

### 5. Metadata Filtering

Filter by document properties before searching:

```python
from qdrant_client.models import Filter, FieldCondition

async def search_with_filters(
    query: str,
    doc_type: str = None,
    date_range: tuple = None
) -> list[dict]:
    """Search with metadata filters."""

    # Build filter
    must_conditions = []

    if doc_type:
        must_conditions.append(
            FieldCondition(
                key="metadata.type",
                match=MatchValue(value=doc_type)
            )
        )

    if date_range:
        start, end = date_range
        must_conditions.append(
            FieldCondition(
                key="metadata.date",
                range={
                    "gte": start,
                    "lte": end
                }
            )
        )

    filter_obj = Filter(must=must_conditions) if must_conditions else None

    # Search with filter
    results = await qdrant_client.search(
        collection_name="documents",
        query_vector=await get_embedding(query),
        query_filter=filter_obj,
        limit=5
    )

    return results
```

**Usage**:
```python
# Only search PDFs from 2024
results = await search_with_filters(
    query="quarterly results",
    doc_type="pdf",
    date_range=("2024-01-01", "2024-12-31")
)
```

---

## Document Processing Best Practices

### 1. Clean Your Documents

**Before indexing**:

```python
import re

def clean_document(text: str) -> str:
    """Clean document before chunking."""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers
    text = re.sub(r'Page \d+', '', text)

    # Remove headers/footers (customize for your docs)
    text = re.sub(r'CONFIDENTIAL|DRAFT', '', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')

    return text.strip()
```

### 2. Extract Metadata

**Enrich chunks with metadata**:

```python
from datetime import datetime
import os

def extract_metadata(file_path: str) -> dict:
    """Extract useful metadata from file."""

    stat = os.stat(file_path)

    return {
        "filename": os.path.basename(file_path),
        "extension": os.path.splitext(file_path)[1],
        "size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }
```

### 3. Handle Different File Types

**Specialized parsers**:

```python
def parse_document(file_path: str) -> str:
    """Parse different document types."""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)

    elif ext == '.docx':
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    elif ext == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()

    elif ext in ['.txt', '.md']:
        with open(file_path, 'r') as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")
```

---

## Monitoring RAG Performance

### 1. Track Retrieval Quality

**Add logging in `app/rag.py`:**

```python
from loguru import logger

async def search_documents(query: str, top_k: int = 5):
    results = # ... your search logic

    # Log retrieval stats
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        logger.info(f"RAG retrieval: query='{query}', found={len(results)}, avg_score={avg_score:.3f}")
    else:
        logger.warning(f"RAG retrieval: query='{query}', found=0")

    return results
```

### 2. A/B Test Different Configurations

```python
import random

def get_rag_config(variant: str = None):
    """Try different RAG configs."""

    if variant is None:
        variant = random.choice(["baseline", "high_recall", "high_precision"])

    configs = {
        "baseline": {
            "top_k": 5,
            "threshold": 0.7
        },
        "high_recall": {
            "top_k": 10,
            "threshold": 0.6
        },
        "high_precision": {
            "top_k": 3,
            "threshold": 0.8
        }
    }

    return configs[variant]
```

### 3. Measure Retrieval Metrics

```python
def calculate_retrieval_metrics(
    retrieved_docs: list[str],
    relevant_docs: list[str]
) -> dict:
    """Calculate precision, recall, F1."""

    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)

    precision = true_positives / (true_positives + false_positives) if retrieved_set else 0
    recall = true_positives / (true_positives + false_negatives) if relevant_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

---

## Common RAG Problems & Solutions

### Problem 1: "No relevant documents found"

**Symptoms**: Agent says "I don't have information about that"

**Solutions**:
1. Lower `RAG_SCORE_THRESHOLD` (try 0.6)
2. Increase `RAG_TOP_K` (try 10)
3. Check if documents are actually indexed:
   ```bash
   curl http://localhost:6333/collections/documents
   ```
4. Try query expansion (see Advanced Techniques)

### Problem 2: "Retrieving irrelevant documents"

**Symptoms**: Agent includes unrelated info in answers

**Solutions**:
1. Raise `RAG_SCORE_THRESHOLD` (try 0.8)
2. Use smaller chunks (try `CHUNK_SIZE=500`)
3. Implement re-ranking
4. Clean your documents better before indexing

### Problem 3: "Context is cut off mid-sentence"

**Symptoms**: Chunks end abruptly

**Solutions**:
1. Increase `CHUNK_OVERLAP` (try 300)
2. Use semantic chunking:
   ```python
   def semantic_chunk(text: str, max_size: int = 1000):
       """Chunk on sentence boundaries."""
       sentences = text.split('. ')
       chunks = []
       current = ""

       for sentence in sentences:
           if len(current + sentence) < max_size:
               current += sentence + ". "
           else:
               chunks.append(current.strip())
               current = sentence + ". "

       if current:
           chunks.append(current.strip())

       return chunks
   ```

### Problem 4: "Slow retrieval"

**Symptoms**: RAG queries take > 2 seconds

**Solutions**:
1. Reduce `RAG_TOP_K` (try 3)
2. Use Qdrant's HNSW index optimization
3. Add caching:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   async def cached_search(query: str):
       return await search_documents(query)
   ```
4. Use batch processing for multiple queries

### Problem 5: "Expensive embedding costs"

**Symptoms**: High OpenAI bills for embeddings

**Solutions**:
1. Cache embeddings:
   ```python
   # Store query embeddings in Redis
   cache_key = f"emb:{hashlib.md5(query.encode()).hexdigest()}"
   cached = await redis.get(cache_key)

   if cached:
       return json.loads(cached)

   embedding = await get_embedding(query)
   await redis.setex(cache_key, 3600, json.dumps(embedding))
   return embedding
   ```
2. Use smaller embedding model (already using text-embedding-3-small)
3. Batch embed documents during indexing

---

## Recommended Configurations by Use Case

### Customer Support Chatbot
```bash
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.75
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```
**Why**: Need accurate answers, moderate context

### Technical Documentation
```bash
RAG_TOP_K=7
RAG_SCORE_THRESHOLD=0.7
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```
**Why**: Complex topics need more context

### Legal/Compliance
```bash
RAG_TOP_K=3
RAG_SCORE_THRESHOLD=0.85
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```
**Why**: Precision critical, exact matches preferred

### Research Assistant
```bash
RAG_TOP_K=10
RAG_SCORE_THRESHOLD=0.65
CHUNK_SIZE=1200
CHUNK_OVERLAP=250
```
**Why**: Broad exploration, recall matters

---

## Testing Your RAG System

### Create Test Suite

```python
# tests/test_rag_quality.py

import pytest
from app.rag import search_documents

test_cases = [
    {
        "query": "What is the capital of France?",
        "expected_content": "Paris",
        "min_score": 0.8
    },
    {
        "query": "Python error handling",
        "expected_content": "try-except",
        "min_score": 0.7
    }
]

@pytest.mark.asyncio
@pytest.mark.parametrize("case", test_cases)
async def test_rag_retrieval(case):
    results = await search_documents(case["query"], top_k=5)

    # Check we got results
    assert len(results) > 0, "No documents retrieved"

    # Check score threshold
    assert results[0]["score"] >= case["min_score"], "Score too low"

    # Check expected content appears
    contents = " ".join(r["content"] for r in results)
    assert case["expected_content"] in contents, "Expected content not found"
```

Run tests:
```bash
pytest tests/test_rag_quality.py -v
```

---

## Next Steps

1. ✅ Review current RAG settings in `.env`
2. 🎯 Choose configuration based on your use case
3. 📊 Add logging to monitor retrieval quality
4. 🧪 Create test suite for your domain
5. 🔄 Iterate based on user feedback
6. 📈 Track metrics in LangSmith

**Further Reading**:
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [RAG Best Practices](https://www.anthropic.com/index/retrieval-augmented-generation)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

Happy optimizing! 🚀
