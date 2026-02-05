"""
Vector store implementation with ChromaDB for pharmaceutical document retrieval.
Demonstrates production-ready RAG system with advanced retrieval strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

from sentence_transformers import SentenceTransformer

from ..config import settings
from ..utils import RetrievalException, retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for semantic search over pharmaceutical documents.
    
    Demonstrates:
    - Vector database integration (ChromaDB)
    - Embedding generation
    - Semantic search
    - Metadata filtering
    - Collection management
    
    Production patterns:
    - Lazy initialization
    - Error handling
    - Retry logic for resilience
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the vector collection
            persist_directory: Directory for persistent storage
            embedding_model: Name/path of embedding model
        """
        if chromadb is None:
            raise RetrievalException(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_model_name = embedding_model or settings.embedding_model
        
        # Lazy initialization
        self._client = None
        self._collection = None
        self._embedding_model = None
        
        logger.info(
            f"VectorStore initialized with collection '{self.collection_name}'"
        )
    
    @property
    def client(self) -> chromadb.Client:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.Client(
                ChromaSettings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        
        return self._client
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        
        return self._embedding_model
    
    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            # Create embedding function
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
        
        return self._collection
    
    @retry_with_exponential_backoff(max_attempts=3)
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents to vector store.
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
            
        Raises:
            RetrievalException: If indexing fails
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            raise RetrievalException(f"Failed to add documents: {str(e)}")
    
    @retry_with_exponential_backoff(max_attempts=3)
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using query text.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"category": "NSAID"})
            
        Returns:
            List of search results with documents, metadata, and scores
            
        Raises:
            RetrievalException: If search fails
        """
        top_k = top_k or settings.retrieval_top_k
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            logger.info(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            raise RetrievalException(f"Search failed: {str(e)}")
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            raise RetrievalException(f"Failed to delete collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory
        }


def load_pharmaceutical_data(data_path: Union[Path, str] = "data/drug_notices.json") -> VectorStore:
    """
    Load pharmaceutical data into vector store.
    
    Demonstrates:
    - Data ingestion pipeline
    - Document preparation
    - Metadata extraction
    - Batch indexing
    
    Args:
        data_path: Path to pharmaceutical data JSON file
        
    Returns:
        Initialized VectorStore with indexed data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise RetrievalException(f"Data file not found: {data_path}")
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    drugs = data.get('drugs', {})
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Prepare documents for indexing
    documents = []
    metadatas = []
    ids = []
    
    for drug_id, drug_info in drugs.items():
        # Create comprehensive document text
        doc_text = f"""
        Drug Name: {drug_info['name']}
        Active Substance: {drug_info['active_substance']}
        Category: {drug_info['category']}
        
        Indications: {drug_info['indications']}
        
        Dosage: {drug_info['dosage']}
        
        Contraindications: {drug_info['contraindications']}
        
        Side Effects: {drug_info['side_effects']}
        
        Warnings: {drug_info['warnings']}
        
        Drug Interactions: {drug_info.get('drug_interactions', 'None documented')}
        
        Mechanism of Action: {drug_info.get('mechanism_of_action', 'Not specified')}
        """
        
        documents.append(doc_text.strip())
        
        # Extract metadata
        metadatas.append({
            "drug_id": drug_id,
            "name": drug_info['name'],
            "category": drug_info['category'],
            "active_substance": drug_info['active_substance'],
            "regulatory_status": drug_info.get('regulatory_status', 'Unknown')
        })
        
        ids.append(f"drug_{drug_id}")
    
    # Index documents
    vector_store.add_documents(documents, metadatas, ids)
    
    stats = vector_store.get_collection_stats()
    logger.info(f"Pharmaceutical data indexed: {stats}")
    
    return vector_store
