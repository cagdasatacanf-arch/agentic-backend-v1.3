"""
Hybrid Search: BM25 (Sparse) + Dense Embeddings

Combines two retrieval methods:
1. BM25 (keyword/sparse matching) - Good for exact matches, names, acronyms
2. Dense embeddings (semantic) - Good for meaning, context, paraphrases

Uses Reciprocal Rank Fusion (RRF) to merge results.

Research basis:
- Hybrid retrieval consistently outperforms single-method retrieval
- RRF is simple yet effective fusion method
- 15-25% improvement in retrieval accuracy

Installation required:
    pip install rank-bm25
"""

from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import pickle
import os

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from app.rag import search_docs, embed
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    alpha: float = 0.7  # Weight for dense (0.7) vs sparse (0.3)
    top_k: int = 10  # Results to return
    bm25_k: int = 20  # BM25 candidates (retrieve more, then fuse)
    dense_k: int = 20  # Dense candidates
    rrf_k: int = 60  # RRF constant (usually 60 works well)


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense embeddings.

    Architecture:
    1. Dense retrieval: Query → Embedding → Qdrant search
    2. Sparse retrieval: Query → BM25 → Top matches
    3. Fusion: RRF (Reciprocal Rank Fusion) combines both

    RRF Formula:
        score(doc) = Σ 1 / (k + rank_in_method_i)

    Where k=60 is a constant, and we sum over both retrieval methods.

    Usage:
        retriever = HybridRetriever()
        await retriever.initialize()  # Build BM25 index
        results = await retriever.hybrid_search("query", top_k=10)
    """

    def __init__(self, config: Optional[HybridSearchConfig] = None):
        """
        Initialize hybrid retriever.

        Args:
            config: Search configuration (uses defaults if None)
        """
        self.config = config or HybridSearchConfig()
        self.qdrant_client = QdrantClient(url=settings.vector_db_url)
        self.collection_name = "docs"

        # BM25 index (initialized lazily)
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []  # Document cache for BM25
        self.doc_id_to_index: Dict[str, int] = {}  # Map doc_id → index in documents list

        # Index persistence
        self.index_path = "data/bm25_index.pkl"
        self.docs_path = "data/bm25_docs.pkl"

        logger.info(f"HybridRetriever initialized with alpha={self.config.alpha}")

    async def initialize(self, rebuild: bool = False) -> None:
        """
        Initialize BM25 index.

        Downloads all documents from Qdrant and builds BM25 index.
        Can load from disk if index already exists.

        Args:
            rebuild: Force rebuild even if index exists
        """
        # Check if index exists on disk
        if not rebuild and os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                logger.info("Loading BM25 index from disk...")
                with open(self.index_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                with open(self.docs_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.doc_id_to_index = data['doc_id_to_index']

                logger.info(f"BM25 index loaded: {len(self.documents)} documents")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 index from disk: {e}. Rebuilding...")

        # Build index from Qdrant
        await self._build_index()

    async def _build_index(self) -> None:
        """Build BM25 index from all documents in Qdrant"""
        logger.info("Building BM25 index from Qdrant...")

        try:
            # Scroll through all documents in collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on collection size
                with_payload=True,
                with_vectors=False  # Don't need vectors for BM25
            )

            points = scroll_result[0]
            logger.info(f"Retrieved {len(points)} documents from Qdrant")

            if not points:
                logger.warning("No documents found in Qdrant. BM25 will return empty results.")
                self.bm25 = None
                return

            # Build document list
            self.documents = []
            tokenized_corpus = []

            for i, point in enumerate(points):
                doc = {
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {})
                }
                self.documents.append(doc)
                self.doc_id_to_index[doc["id"]] = i

                # Tokenize for BM25 (simple whitespace tokenization)
                tokens = doc["text"].lower().split()
                tokenized_corpus.append(tokens)

            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_corpus)

            logger.info(f"BM25 index built with {len(self.documents)} documents")

            # Save to disk
            await self._save_index()

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
            self.bm25 = None

    async def _save_index(self) -> None:
        """Save BM25 index to disk"""
        try:
            os.makedirs("data", exist_ok=True)

            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25, f)

            with open(self.docs_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_id_to_index': self.doc_id_to_index
                }, f)

            logger.info("BM25 index saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")

    async def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25 (keyword) search.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of documents with BM25 scores
        """
        if self.bm25 is None or not self.documents:
            logger.warning("BM25 index not initialized. Call initialize() first.")
            return []

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                doc["retrieval_method"] = "bm25"
                results.append(doc)

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    async def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Dense (semantic) search using existing RAG system.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of documents with similarity scores
        """
        try:
            results = await search_docs(query, top_k=top_k)

            # Add retrieval method tag
            for doc in results:
                doc["retrieval_method"] = "dense"

            logger.debug(f"Dense search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        alpha: Optional[float] = None,
        k: Optional[int] = None
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(doc) = alpha * (1/(k + rank_dense)) + (1-alpha) * (1/(k + rank_sparse))

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from BM25
            alpha: Weight for dense (0.0-1.0), default uses config.alpha
            k: RRF constant, default uses config.rrf_k

        Returns:
            Fused and sorted results
        """
        alpha = alpha if alpha is not None else self.config.alpha
        k = k if k is not None else self.config.rrf_k

        # Build score dictionary
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        # Score from dense retrieval
        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("id", "")
            if doc_id:
                scores[doc_id] = alpha / (k + rank)
                doc_map[doc_id] = doc

        # Add score from sparse retrieval
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.get("id", "")
            if doc_id:
                sparse_score = (1 - alpha) / (k + rank)
                scores[doc_id] = scores.get(doc_id, 0.0) + sparse_score

                # Keep document with higher score's metadata
                if doc_id not in doc_map or doc.get("score", 0) > doc_map[doc_id].get("score", 0):
                    doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            doc["retrieval_method"] = "hybrid"  # Mark as hybrid
            results.append(doc)

        logger.debug(f"RRF fused {len(results)} unique documents")
        return results

    async def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Perform hybrid search (BM25 + Dense + RRF fusion).

        Args:
            query: Search query
            top_k: Number of final results (default: config.top_k)
            alpha: Weight for dense vs sparse (default: config.alpha)
            verbose: Log detailed retrieval info

        Returns:
            Fused and ranked results
        """
        top_k = top_k or self.config.top_k
        alpha = alpha if alpha is not None else self.config.alpha

        if verbose:
            logger.info(f"Hybrid search: query='{query}', top_k={top_k}, alpha={alpha}")

        # 1. Dense retrieval
        dense_results = await self.dense_search(query, top_k=self.config.dense_k)

        # 2. Sparse retrieval (BM25)
        sparse_results = await self.bm25_search(query, top_k=self.config.bm25_k)

        if verbose:
            logger.info(f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results")

        # 3. Fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=alpha
        )

        # 4. Return top-k
        final_results = fused_results[:top_k]

        if verbose:
            logger.info(f"Hybrid search returned {len(final_results)} results")
            for i, doc in enumerate(final_results[:3]):
                logger.info(
                    f"  {i+1}. RRF: {doc.get('rrf_score', 0):.4f}, "
                    f"Text: {doc.get('text', '')[:60]}..."
                )

        return final_results

    async def compare_methods(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare all three retrieval methods side by side.

        Args:
            query: Search query
            top_k: Results per method

        Returns:
            Dict with results from dense, sparse, and hybrid
        """
        dense_results = await self.dense_search(query, top_k=top_k)
        sparse_results = await self.bm25_search(query, top_k=top_k)
        hybrid_results = await self.hybrid_search(query, top_k=top_k)

        return {
            "query": query,
            "dense": {
                "count": len(dense_results),
                "results": dense_results
            },
            "sparse": {
                "count": len(sparse_results),
                "results": sparse_results
            },
            "hybrid": {
                "count": len(hybrid_results),
                "results": hybrid_results
            },
            "analysis": {
                "unique_to_dense": len([d for d in dense_results if d["id"] not in [s["id"] for s in sparse_results]]),
                "unique_to_sparse": len([s for s in sparse_results if s["id"] not in [d["id"] for d in dense_results]]),
                "overlap": len(set([d["id"] for d in dense_results]) & set([s["id"] for s in sparse_results]))
            }
        }


# ============================================================================
# Singleton Instance
# ============================================================================

_hybrid_retriever: Optional[HybridRetriever] = None


async def get_hybrid_retriever(rebuild: bool = False) -> HybridRetriever:
    """
    Get or create global hybrid retriever instance.

    Args:
        rebuild: Force rebuild BM25 index

    Returns:
        Initialized HybridRetriever
    """
    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
        await _hybrid_retriever.initialize(rebuild=rebuild)
    elif rebuild:
        await _hybrid_retriever.initialize(rebuild=True)

    return _hybrid_retriever


# ============================================================================
# Convenience Functions
# ============================================================================

async def hybrid_search(query: str, top_k: int = 10, alpha: float = 0.7) -> List[Dict]:
    """
    Quick hybrid search function.

    Args:
        query: Search query
        top_k: Number of results
        alpha: Weight for dense (0.7) vs sparse (0.3)

    Returns:
        Hybrid search results
    """
    retriever = await get_hybrid_retriever()
    return await retriever.hybrid_search(query, top_k=top_k, alpha=alpha)
