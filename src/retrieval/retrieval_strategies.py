"""
Advanced retrieval strategies for RAG system.
Demonstrates hybrid search, re-ranking, and query optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .vector_store import VectorStore
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Structure for retrieval results.
    
    Demonstrates proper data modeling with dataclasses.
    """
    document: str
    metadata: Dict[str, Any]
    score: float
    source: str  # "dense", "sparse", "hybrid"
    rank: int


class HybridRetriever:
    """
    Hybrid retrieval combining dense (semantic) and sparse (keyword) search.
    
    Demonstrates:
    - Multiple retrieval strategies
    - Reciprocal Rank Fusion (RRF)
    - Re-ranking
    - Query optimization
    
    Production pattern for advanced RAG systems.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
        """
        self.vector_store = vector_store
        self.top_k = settings.retrieval_top_k
        self.rerank_top_k = settings.rerank_top_k
        
        logger.info("HybridRetriever initialized")
    
    def dense_retrieval(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Dense retrieval using semantic embeddings.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filters
            
        Returns:
            List of retrieval results
        """
        top_k = top_k or self.top_k
        
        results = self.vector_store.search(query, top_k, filter_dict)
        
        return [
            RetrievalResult(
                document=r['document'],
                metadata=r['metadata'],
                score=1.0 - r['distance'] if r['distance'] else 0.5,  # Convert distance to similarity
                source="dense",
                rank=i
            )
            for i, r in enumerate(results)
        ]
    
    def keyword_retrieval(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Sparse retrieval using keyword matching (simple implementation).
        
        In production, would use BM25 or Elasticsearch.
        This is a simplified version for demonstration.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of retrieval results
        """
        # Simplified keyword matching
        # In production, use BM25 or Elasticsearch
        query_terms = set(query.lower().split())
        
        # Get all documents
        all_results = self.vector_store.search(query, top_k=top_k or self.top_k)
        
        # Score by keyword overlap
        scored_results = []
        for result in all_results:
            doc_terms = set(result['document'].lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            
            scored_results.append((score, result))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [
            RetrievalResult(
                document=r['document'],
                metadata=r['metadata'],
                score=score,
                source="sparse",
                rank=i
            )
            for i, (score, r) in enumerate(scored_results[:top_k or self.top_k])
        ]
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1/(k + rank(d))
        
        Demonstrates advanced retrieval fusion technique.
        
        Args:
            result_lists: List of result lists to fuse
            k: RRF constant (default 60)
            
        Returns:
            Fused and re-ranked results
        """
        # Calculate RRF scores
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, RetrievalResult] = {}
        
        for results in result_lists:
            for result in results:
                doc_id = result.metadata.get('drug_id', '')
                
                # RRF score
                rrf_score = 1.0 / (k + result.rank + 1)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_info[doc_id] = result
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            result = doc_info[doc_id]
            fused_results.append(
                RetrievalResult(
                    document=result.document,
                    metadata=result.metadata,
                    score=score,
                    source="hybrid",
                    rank=rank
                )
            )
        
        return fused_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Demonstrates production RAG retrieval pipeline.
        
        Args:
            query: Search query
            top_k: Number of final results
            use_rerank: Whether to use re-ranking
            
        Returns:
            Hybrid search results
        """
        top_k = top_k or self.rerank_top_k
        
        logger.info(f"Hybrid search for query: {query[:50]}...")
        
        # Get results from both methods
        dense_results = self.dense_retrieval(query, top_k=self.top_k)
        sparse_results = self.keyword_retrieval(query, top_k=self.top_k)
        
        logger.info(f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results")
        
        # Fuse with RRF
        fused_results = self.reciprocal_rank_fusion([dense_results, sparse_results])
        
        # Optionally re-rank (simplified - in production use cross-encoder)
        if use_rerank:
            fused_results = self.simple_rerank(query, fused_results, top_k)
        
        return fused_results[:top_k]
    
    def simple_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Simple re-ranking based on query term presence.
        
        In production, use cross-encoder models for re-ranking.
        
        Args:
            query: Search query
            results: Results to re-rank
            top_k: Number of results to keep
            
        Returns:
            Re-ranked results
        """
        query_terms = set(query.lower().split())
        
        # Re-score based on term frequency in document
        reranked = []
        for result in results:
            doc_lower = result.document.lower()
            term_count = sum(doc_lower.count(term) for term in query_terms)
            
            # Combine original score with term frequency
            new_score = result.score * 0.7 + (term_count / len(query_terms)) * 0.3
            
            reranked.append(
                RetrievalResult(
                    document=result.document,
                    metadata=result.metadata,
                    score=new_score,
                    source=f"{result.source}_reranked",
                    rank=result.rank
                )
            )
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i
        
        return reranked[:top_k]
    
    def query_expansion(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        Simplified version - in production use LLM or knowledge base.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        # Simple synonym expansion (in production, use proper NLP)
        expansions = [query]
        
        # Drug category synonyms
        synonym_map = {
            "pain": ["pain", "analgesia", "analgesic"],
            "fever": ["fever", "pyrexia", "antipyretic"],
            "antibiotic": ["antibiotic", "antibacterial", "antimicrobial"],
            "diabetes": ["diabetes", "diabetic", "hyperglycemia"],
        }
        
        query_lower = query.lower()
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn != term:
                        expansions.append(query.lower().replace(term, syn))
        
        return list(set(expansions))[:3]  # Limit to 3 variations
