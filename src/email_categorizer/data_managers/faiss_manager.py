import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer

from ..types import Message, Category
from ..utils.utils import get_compact_message_representation


class FaissManager:
    """Manages all FAISS index operations.
    
    Responsibilities:
    - Embedding generation using sentence-transformers (all-mpnet-base-v2)
    - FAISS index building and management with IndexIDMap
    - Semantic search (vector similarity)
    """
    
    DEFAULT_INDEX_PATH = "data/embeddings"
    MODEL_NAME = "all-mpnet-base-v2"
    
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, force_recreate: bool = False):
        """
        Initialize FAISS manager.
        
        Args:
            index_path: Path to directory containing FAISS index files
            force_recreate: If True, recreate index from scratch
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Set single-threaded mode for stability
        import torch
        torch.set_num_threads(1)
        
        print(f"Loading embedding model: {self.MODEL_NAME}")
        self.embedding_model = SentenceTransformer(self.MODEL_NAME, device="cpu")
        print(f"Embedding model {self.MODEL_NAME} loaded")
        
        # Load or create FAISS index
        self.id_mapping: Dict[int, str] = None # Map FAISS index position to message ID
        self.faiss_index = None
        self._load_or_create_index(force_recreate)
    
    def _load_or_create_index(self, force_recreate: bool = False) -> None:
        """Load existing FAISS index or create a new one."""
        index_file = self.index_path / "faiss.index"
        mapping_file = self.index_path / "id_mapping.json"
        
        if force_recreate:
            if index_file.exists():
                index_file.unlink()
            if mapping_file.exists():
                mapping_file.unlink()
        
        if index_file.exists() and mapping_file.exists():
            self.faiss_index = faiss.read_index(str(index_file))
            with open(mapping_file, 'r') as f:
                self.id_mapping = {int(k): v for k, v in json.load(f).items()}
            
            print(f"Loaded preexisting FAISS index with {self.faiss_index.ntotal} vectors from {index_file}")
        else:
            self._create_new_index()
            print(f"Created new empty FAISS index.")
    
    def _create_new_index(self) -> None:
        """
        Create a new, empty FAISS index.
        
        Returns:
            Paths to the saved index file and mapping file
        """
        embedding_dim = self._get_embedding_model().get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.id_mapping = {}
        self._save_index()
    
    def _save_index(self) -> None:
        """Save the FAISS index and ID mapping to disk."""
        index_file = self.index_path / "faiss.index"
        mapping_file = self.index_path / "id_mapping.json"
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_file))
        
        # Save ID mapping
        with open(mapping_file, 'w') as f:
            json.dump(self.id_mapping, f, indent=2)

    def add_messages_to_index(self, messages: List[Message]) -> None:
        """
        Add multiple messages to the FAISS index in batch.
        
        Args:
            messages: List of Message objects to add
        """
        if not messages:
            return
                
        texts = []
        msg_ids = []
        for msg in messages:
            text_representation = get_compact_message_representation(msg)
            texts.append(text_representation)
            msg_ids.append(msg.msg_id)
        
        # Batch embed
        print(f"Encoding {len(texts)} messages...")
        embeddings_array = self._get_embedding_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,  # Disable to avoid multiprocessing issues
            batch_size=32
        ).astype(np.float32)
        print(f"Generated {len(embeddings_array)} embeddings")
        
        # Add embeddings to index
        start_idx = self.faiss_index.ntotal
        self.faiss_index.add(embeddings_array)
        
        # Update ID mapping
        for i, msg_id in enumerate(msg_ids):
            self.id_mapping[start_idx + i] = msg_id
        
        print(f"Added {len(messages)} messages to FAISS index")
    
    def search_similar(
        self,
        category: Category,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        Search for semantically similar messages.
        
        Args:
            category: Category object to search for
            limit: Maximum results
            
        Returns:
            List of (msg_id, similarity_score) tuples
        """
        # Return empty results for empty index
        if self.faiss_index.ntotal == 0:
            return []

        query_text = f"Category name: {category.name}\nCategory description: {category.description}"

        query_embedding = self._get_embedding_model().encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        query_2d = query_embedding.reshape(1, -1)
        
        # Search for similar vectors (cosine similarity)
        scores, indices = self.faiss_index.search(query_2d, min(limit, self.faiss_index.ntotal))
        
        # Convert results to (msg_id, score) tuples
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots (when fewer results than requested)
                break
                
            if idx in self.id_mapping:
                msg_id = self.id_mapping[idx]
                similarity = float(score)
                results.append((msg_id, similarity))
        
        return results
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Get the embedding model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        return self.embedding_model