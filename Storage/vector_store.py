"""
Vector Store for Event Embeddings and Similarity Search
Supports both Pinecone and OpenSearch (AWS)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, cast
from uuid import UUID
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import hashlib

from Data.models import GeopoliticalEvent
from Data.settings import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages event embeddings and similarity search"""
    
    def __init__(self, use_pinecone: bool = True):
        self.settings = Settings()
        self.use_pinecone = use_pinecone

        # Ensure Hugging Face caches write to a repo-local path to avoid permission issues
        cache_dir = Path("cache/hf")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        self.embedding_dim = 384
        
        # Initialize vector database
        if use_pinecone and self.settings.PINECONE_API_KEY:
            self._init_pinecone()
        else:
            logger.info("Using local in-memory vector store (not production-ready)")
            self.local_vectors = {}  # event_id -> embedding
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
            
            # Check if index exists
            index_name = self.settings.PINECONE_INDEX_NAME
            
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.settings.PINECONE_ENVIRONMENT
                    )
                )
            
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            logger.info("Falling back to local vector store")
            self.use_pinecone = False
            self.local_vectors = {}
    
    def _create_embedding_text(self, event: GeopoliticalEvent) -> str:
        """Create text for embedding from event"""
        parts = [
            event.title,
            event.description or "",
            " ".join(event.locations),
            " ".join(event.countries),
            " ".join(event.actors)
        ]
        return " ".join([p for p in parts if p])
    
    def embed_event(self, event: GeopoliticalEvent) -> np.ndarray:
        """Generate embedding for an event"""
        try:
            text = self._create_embedding_text(event)
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for raw text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def store_event(self, event: GeopoliticalEvent, embedding: Optional[np.ndarray] = None) -> bool:
        """Store event embedding in vector database"""
        
        if embedding is None:
            embedding = self.embed_event(event)
        
        event_id = str(event.event_id)
        
        try:
            if self.use_pinecone:
                # Store in Pinecone
                metadata = {
                    "title": event.title[:500],  # Pinecone has metadata limits
                    "source": event.source.value,
                    "timestamp": event.timestamp.isoformat(),
                    "threat_category": event.threat_category.value if event.threat_category else "unknown",
                    "locations": ",".join(event.locations[:5]),  # First 5 locations
                    "countries": ",".join(event.countries[:5]),
                    "importance_score": float(event.importance_score)
                }
                
                self.index.upsert(
                    vectors=[(event_id, embedding.tolist(), metadata)]
                )
            else:
                # Store locally
                self.local_vectors[event_id] = {
                    "embedding": embedding,
                    "event": event
                }
            
            logger.debug(f"Stored embedding for event: {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing event embedding: {e}")
            return False
    
    def find_similar(
        self,
        event: GeopoliticalEvent,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find similar events using vector similarity"""
        
        embedding = self.embed_event(event)
        
        try:
            if self.use_pinecone:
                # Query Pinecone
                results: Any = cast(
                    Any,
                    self.index.query(
                        vector=embedding.tolist(),
                        top_k=top_k,
                        include_metadata=True
                    ),
                )
                
                similar_events = []
                for match in getattr(results, "matches", []):
                    if getattr(match, "score", 0) >= similarity_threshold:
                        similar_events.append((getattr(match, "id", ""), getattr(match, "score", 0)))
                
                return similar_events
            
            else:
                # Local similarity search
                similarities = []
                for event_id, data in self.local_vectors.items():
                    stored_embedding = data["embedding"]
                    similarity = self._cosine_similarity(embedding, stored_embedding)
                    if similarity >= similarity_threshold:
                        similarities.append((event_id, similarity))
                
                # Sort by similarity descending
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
        
        except Exception as e:
            logger.error(f"Error finding similar events: {e}")
            return []
    
    def find_similar_by_text(
        self,
        text: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Find similar events by text query"""
        
        embedding = self.embed_text(text)
        
        try:
            if self.use_pinecone:
                # Query with optional filters
                results: Any = cast(
                    Any,
                    self.index.query(
                        vector=embedding.tolist(),
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter_dict
                    ),
                )
                
                similar_events = []
                for match in getattr(results, "matches", []):
                    similar_events.append((
                        getattr(match, "id", ""),
                        getattr(match, "score", 0.0),
                        getattr(match, "metadata", {})
                    ))
                
                return similar_events
            
            else:
                # Local search
                similarities = []
                for event_id, data in self.local_vectors.items():
                    stored_embedding = data["embedding"]
                    similarity = self._cosine_similarity(embedding, stored_embedding)
                    
                    # Apply filters if provided
                    if filter_dict:
                        event = data["event"]
                        match = True
                        for key, value in filter_dict.items():
                            if not self._matches_filter(event, key, value):
                                match = False
                                break
                        if not match:
                            continue
                    
                    similarities.append((
                        event_id,
                        similarity,
                        self._get_metadata(data["event"])
                    ))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
        
        except Exception as e:
            logger.error(f"Error in text similarity search: {e}")
            return []
    
    def deduplicate_events(
        self,
        events: List[GeopoliticalEvent],
        similarity_threshold: float = 0.85
    ) -> List[GeopoliticalEvent]:
        """Remove duplicate events based on embedding similarity"""
        
        unique_events = []
        seen_hashes = set()
        
        for event in events:
            # Quick hash-based deduplication first
            event_hash = self._compute_hash(event)
            if event_hash in seen_hashes:
                event.is_duplicate = True
                continue
            
            # Vector similarity deduplication
                similar = self.find_similar(event, top_k=5, similarity_threshold=similarity_threshold)
                
                if similar and len(similar) > 0:
                    # Mark as duplicate
                    event.is_duplicate = True
                    try:
                        event.duplicate_of = UUID(similar[0][0])
                    except Exception:
                        event.duplicate_of = None
                    logger.debug(f"Duplicate detected: {event.title[:50]}...")
            else:
                # Unique event
                unique_events.append(event)
                seen_hashes.add(event_hash)
                
                # Store embedding for future comparisons
                self.store_event(event)
        
        logger.info(f"Deduplicated: {len(events)} -> {len(unique_events)} unique events")
        return unique_events
    
    def _compute_hash(self, event: GeopoliticalEvent) -> str:
        """Compute hash of event for quick deduplication"""
        text = f"{event.title}{event.source_url}"
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _matches_filter(self, event: GeopoliticalEvent, key: str, value: Any) -> bool:
        """Check if event matches filter criteria"""
        if key == "threat_category":
            return bool(event.threat_category and event.threat_category.value == value)
        elif key == "source":
            return event.source.value == value
        elif key == "locations":
            return value in event.locations
        elif key == "countries":
            return value in event.countries
        return True
    
    def _get_metadata(self, event: GeopoliticalEvent) -> Dict[str, Any]:
        """Extract metadata from event"""
        return {
            "title": event.title,
            "source": event.source.value,
            "timestamp": event.timestamp.isoformat(),
            "threat_category": event.threat_category.value if event.threat_category else "unknown",
            "locations": ",".join(event.locations),
            "importance_score": event.importance_score
        }
    
    def cluster_events(
        self,
        events: List[GeopoliticalEvent],
        eps: float = 0.3,
        min_samples: int = 3
    ) -> Dict[int, List[str]]:
        """Cluster events using DBSCAN on embeddings"""
        
        if len(events) < min_samples:
            logger.warning("Not enough events for clustering")
            return {0: [str(e.event_id) for e in events]}
        
        try:
            from sklearn.cluster import DBSCAN
            
            # Generate embeddings
            embeddings = []
            event_ids = []
            
            for event in events:
                embedding = self.embed_event(event)
                embeddings.append(embedding)
                event_ids.append(str(event.event_id))
            
            embeddings_array = np.array(embeddings)
            
            # Cluster with DBSCAN
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'
            ).fit(embeddings_array)
            
            # Organize clusters
            clusters = {}
            for event_id, cluster_id in zip(event_ids, clustering.labels_):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(event_id)
            
            logger.info(f"Clustered {len(events)} events into {len(clusters)} clusters")
            return clusters
        
        except Exception as e:
            logger.error(f"Error clustering events: {e}")
            return {0: [str(e.event_id) for e in events]}
    
    def delete_event(self, event_id: str) -> bool:
        """Delete event from vector store"""
        try:
            if self.use_pinecone:
                self.index.delete(ids=[event_id])
            else:
                if event_id in self.local_vectors:
                    del self.local_vectors[event_id]
            
            logger.debug(f"Deleted event: {event_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting event: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if self.use_pinecone:
                stats = self.index.describe_index_stats()
                return {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "backend": "pinecone"
                }
            else:
                return {
                    "total_vectors": len(self.local_vectors),
                    "dimension": self.embedding_dim,
                    "backend": "local"
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from datetime import datetime
    from Data.models import EventSource
    
    # Initialize vector store
    vector_store = VectorStore(use_pinecone=False)  # Use local for testing
    
    # Create test events
    event1 = GeopoliticalEvent(
        timestamp=datetime.utcnow(),
        source=EventSource.GDELT,
        source_url="https://example.com/1",
        title="Chinese warships near Taiwan",
        description="PLA Navy conducts exercises in Taiwan Strait",
        locations=["Taiwan Strait", "Taiwan"]
    )
    
    event2 = GeopoliticalEvent(
        timestamp=datetime.utcnow(),
        source=EventSource.GDELT,
        source_url="https://example.com/2",
        title="Chinese military drills near Taiwan",
        description="China conducts naval exercises near Taiwan",
        locations=["Taiwan", "China"]
    )
    
    # Store events
    vector_store.store_event(event1)
    vector_store.store_event(event2)
    
    # Find similar
    similar = vector_store.find_similar(event1, top_k=5)
    print(f"\nSimilar events to '{event1.title[:30]}...':")
    for event_id, score in similar:
        print(f"  {event_id}: {score:.3f}")
    
    # Test deduplication
    events = [event1, event2]
    unique = vector_store.deduplicate_events(events, similarity_threshold=0.8)
    print(f"\nDeduplication: {len(events)} -> {len(unique)} unique events")
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Total Vectors: {stats['total_vectors']}")
    print(f"  Backend: {stats['backend']}")
