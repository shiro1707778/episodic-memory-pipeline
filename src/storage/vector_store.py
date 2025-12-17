"""
FAISS vector store for semantic similarity search.

Manages vector embeddings for episodes, facts, and summaries.
"""
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu package required. Install with: pip install faiss-cpu")


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: int              # FAISS index ID
    score: float         # Similarity score (higher = more similar)
    distance: float      # L2 distance (lower = more similar)


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Uses a flat L2 index for simplicity. For larger datasets,
    consider IVF or HNSW indices.
    
    Design notes:
    - Maintains separate indices for episodes, facts, summaries
    - Index IDs map to database record IDs via metadata
    - Supports incremental additions and persistence
    """
    
    def __init__(self, base_path: Path, dimension: int = 1536):
        """
        Initialize vector store.
        
        Args:
            base_path: Directory for storing index files
            dimension: Embedding dimension
        """
        self.base_path = base_path
        self.dimension = dimension
        
        # Ensure directory exists
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load indices
        self._indices: dict[str, faiss.Index] = {}
        self._id_maps: dict[str, list[str]] = {}  # Maps FAISS IDs to record IDs
        
        self._load_or_create_indices()
    
    def _index_path(self, name: str) -> Path:
        """Get path for an index file."""
        return self.base_path.parent / f"{self.base_path.stem}_{name}.faiss"
    
    def _id_map_path(self, name: str) -> Path:
        """Get path for ID mapping file."""
        return self.base_path.parent / f"{self.base_path.stem}_{name}_ids.npy"
    
    def _load_or_create_indices(self):
        """Load existing indices or create new ones."""
        for name in ["episodes", "facts", "summaries"]:
            index_path = self._index_path(name)
            id_map_path = self._id_map_path(name)
            
            if index_path.exists():
                self._indices[name] = faiss.read_index(str(index_path))
                if id_map_path.exists():
                    self._id_maps[name] = list(np.load(str(id_map_path), allow_pickle=True))
                else:
                    self._id_maps[name] = []
            else:
                # Create flat L2 index (exact search)
                self._indices[name] = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
                self._id_maps[name] = []
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add(
        self,
        index_name: str,
        record_id: str,
        embedding: np.ndarray
    ) -> int:
        """
        Add a single vector to an index.
        
        Args:
            index_name: "episodes", "facts", or "summaries"
            record_id: Database record ID
            embedding: Vector embedding
            
        Returns:
            FAISS index ID
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        # Ensure correct shape and normalize
        embedding = embedding.reshape(1, -1).astype(np.float32)
        embedding = self._normalize(embedding)
        
        # Add to index
        faiss_id = self._indices[index_name].ntotal
        self._indices[index_name].add(embedding)
        self._id_maps[index_name].append(record_id)
        
        return faiss_id
    
    def add_batch(
        self,
        index_name: str,
        record_ids: list[str],
        embeddings: np.ndarray
    ) -> list[int]:
        """
        Add multiple vectors to an index.
        
        Args:
            index_name: "episodes", "facts", or "summaries"
            record_ids: List of database record IDs
            embeddings: 2D array of embeddings
            
        Returns:
            List of FAISS index IDs
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        if len(record_ids) != embeddings.shape[0]:
            raise ValueError("record_ids and embeddings must have same length")
        
        # Normalize
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalize(embeddings)
        
        # Record starting ID
        start_id = self._indices[index_name].ntotal
        
        # Add to index
        self._indices[index_name].add(embeddings)
        self._id_maps[index_name].extend(record_ids)
        
        return list(range(start_id, start_id + len(record_ids)))
    
    def search(
        self,
        index_name: str,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            index_name: "episodes", "facts", or "summaries"
            query_embedding: Query vector
            k: Number of results
            threshold: Minimum similarity score (0-1 for cosine)
            
        Returns:
            List of (record_id, similarity_score) tuples
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        index = self._indices[index_name]
        if index.ntotal == 0:
            return []
        
        # Prepare query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = self._normalize(query)
        
        # Search
        k = min(k, index.ntotal)
        scores, indices = index.search(query, k)
        
        # Map to record IDs and filter by threshold
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            if threshold is not None and score < threshold:
                continue
            
            record_id = self._id_maps[index_name][idx]
            results.append((record_id, float(score)))
        
        return results
    
    def search_with_filter(
        self,
        index_name: str,
        query_embedding: np.ndarray,
        valid_ids: set[str],
        k: int = 10,
        threshold: Optional[float] = None
    ) -> list[tuple[str, float]]:
        """
        Search with ID filtering (post-filter approach).
        
        Args:
            index_name: Index to search
            query_embedding: Query vector
            valid_ids: Set of record IDs to include
            k: Number of results
            threshold: Minimum similarity score
            
        Returns:
            Filtered list of (record_id, similarity_score) tuples
        """
        # Search more than k to account for filtering
        raw_results = self.search(
            index_name, 
            query_embedding, 
            k=min(k * 3, self._indices[index_name].ntotal),
            threshold=threshold
        )
        
        # Filter
        filtered = [(rid, score) for rid, score in raw_results if rid in valid_ids]
        
        return filtered[:k]
    
    def save(self):
        """Persist all indices to disk."""
        for name in self._indices:
            faiss.write_index(
                self._indices[name],
                str(self._index_path(name))
            )
            np.save(
                str(self._id_map_path(name)),
                np.array(self._id_maps[name], dtype=object)
            )
    
    def get_record_id(self, index_name: str, faiss_id: int) -> Optional[str]:
        """Get record ID from FAISS ID."""
        if index_name not in self._id_maps:
            return None
        if faiss_id < 0 or faiss_id >= len(self._id_maps[index_name]):
            return None
        return self._id_maps[index_name][faiss_id]
    
    def get_statistics(self) -> dict:
        """Get vector store statistics."""
        return {
            name: {
                "count": self._indices[name].ntotal,
                "dimension": self.dimension
            }
            for name in self._indices
        }

