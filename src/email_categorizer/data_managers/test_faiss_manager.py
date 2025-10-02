import pytest
import tempfile
import json
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from .faiss_manager import FaissManager
from ..types import Message


class TestFaissManager:
    """Test cases for FaissManager class."""
    
    @pytest.fixture
    def temp_index_path(self):
        """Create a temporary directory for FAISS index files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            Message(
                msg_id="msg1",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-01",
                subject="Machine Learning",
                preview_text="Test message about ML",
                body_text="This is a test message about machine learning and AI"
            ),
            Message(
                msg_id="msg2",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-02",
                subject="NLP Research",
                preview_text="Message about NLP",
                body_text="Another message discussing natural language processing"
            ),
            Message(
                msg_id="msg3",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-03",
                subject="Database Systems",
                preview_text="Database message",
                body_text="A message about database management and SQL queries"
            )
        ]
    
    def test_init_default_path(self):
        """Test FaissManager initialization with default path."""
        manager = FaissManager()
        assert manager.index_path == Path("data/embeddings")
        assert manager.embedding_model is not None
        assert manager.faiss_index is not None
    
    def test_init_custom_path(self, temp_index_path):
        """Test FaissManager initialization with custom path."""
        manager = FaissManager(index_path=temp_index_path)
        assert manager.index_path == Path(temp_index_path)
        assert manager.embedding_model is not None
        assert manager.faiss_index is not None
    
    def test_force_recreate_removes_existing_files(self, temp_index_path):
        """Test force_recreate flag removes existing index files."""
        # Create initial manager and save
        test_msg = Message(
            msg_id="test_msg",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="Test",
            preview_text="Test",
            body_text="test content"
        )
        manager1 = FaissManager(index_path=temp_index_path)
        manager1.add_messages_to_index([test_msg])
        manager1._save_index()
        
        # Verify files exist
        index_file = Path(temp_index_path) / "faiss.index"
        mapping_file = Path(temp_index_path) / "id_mapping.json"
        assert index_file.exists()
        assert mapping_file.exists()
        
        # Create new manager with force_recreate
        manager2 = FaissManager(index_path=temp_index_path, force_recreate=True)
        
        # Verify index is empty
        assert manager2.faiss_index.ntotal == 0
    
    def test_create_new_index(self, temp_index_path):
        """Test _create_new_index creates proper FAISS index."""
        manager = FaissManager(index_path=temp_index_path)
        
        # Verify index dimension matches model
        embedding_dim = manager.embedding_model.get_sentence_embedding_dimension()
        assert manager.faiss_index.d == embedding_dim
        assert manager.faiss_index.ntotal == 0
    
    def test_embedding_generation(self, temp_index_path):
        """Test embedding generation through add_messages_to_index."""
        manager = FaissManager(index_path=temp_index_path)
        
        test_msg = Message(
            msg_id="test_msg",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="Test Subject",
            preview_text="Test",
            body_text="This is a test message"
        )
        
        manager.add_messages_to_index([test_msg])
        
        # Check that embedding was added
        assert manager.faiss_index.ntotal == 1
        assert manager.id_mapping[0] == "test_msg"
    
    def test_add_messages_to_index_empty_list(self, temp_index_path):
        """Test add_messages_to_index with empty list."""
        manager = FaissManager(index_path=temp_index_path)
        
        initial_count = manager.faiss_index.ntotal
        manager.add_messages_to_index([])
        
        assert manager.faiss_index.ntotal == initial_count
    
    def test_add_messages_to_index_single(self, temp_index_path):
        """Test add_messages_to_index with single message."""
        manager = FaissManager(index_path=temp_index_path)
        
        test_msg = Message(
            msg_id="msg1",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="Test",
            preview_text="Test",
            body_text="This is a test message"
        )
        manager.add_messages_to_index([test_msg])
        
        assert manager.faiss_index.ntotal == 1
        assert 0 in manager.id_mapping
        assert manager.id_mapping[0] == "msg1"
    
    def test_add_messages_to_index_multiple(self, temp_index_path, sample_messages):
        """Test add_messages_to_index with multiple messages."""
        manager = FaissManager(index_path=temp_index_path)
        
        manager.add_messages_to_index(sample_messages)
        
        assert manager.faiss_index.ntotal == 3
        assert len(manager.id_mapping) == 3
        
        # Verify all message IDs are in mapping
        msg_ids = {manager.id_mapping[i] for i in range(3)}
        assert msg_ids == {"msg1", "msg2", "msg3"}
    
    def test_add_messages_to_index_incremental(self, temp_index_path):
        """Test adding messages incrementally maintains correct mapping."""
        manager = FaissManager(index_path=temp_index_path)
        
        # Add first batch
        msg1 = Message(
            msg_id="msg1",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="First",
            preview_text="First",
            body_text="First message"
        )
        manager.add_messages_to_index([msg1])
        assert manager.faiss_index.ntotal == 1
        assert manager.id_mapping[0] == "msg1"
        
        # Add second batch
        msg2 = Message(
            msg_id="msg2",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-02",
            subject="Second",
            preview_text="Second",
            body_text="Second message"
        )
        manager.add_messages_to_index([msg2])
        assert manager.faiss_index.ntotal == 2
        assert manager.id_mapping[0] == "msg1"
        assert manager.id_mapping[1] == "msg2"
    
    def test_search_similar_empty_index(self, temp_index_path):
        """Test search_similar with empty index."""
        manager = FaissManager(index_path=temp_index_path)
        
        results = manager.search_similar("test query", limit=10)
        assert results == []
    
    def test_search_similar_basic(self, temp_index_path, sample_messages):
        """Test search_similar with basic query."""
        manager = FaissManager(index_path=temp_index_path)
        manager.add_messages_to_index(sample_messages)
        
        results = manager.search_similar("machine learning", limit=10)
        
        # Should return results
        assert len(results) > 0
        
        # Each result should be (msg_id, score) tuple
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in results)
        
        # Top result should be msg1 (contains "machine learning")
        assert results[0][0] == "msg1"
    
    def test_search_similar_limit(self, temp_index_path, sample_messages):
        """Test search_similar with limit parameter."""
        manager = FaissManager(index_path=temp_index_path)
        manager.add_messages_to_index(sample_messages)
        
        results = manager.search_similar("test query", limit=2)
        assert len(results) <= 2
    
    def test_search_similar_ordering(self, temp_index_path, sample_messages):
        """Test search_similar returns results ordered by similarity."""
        manager = FaissManager(index_path=temp_index_path)
        manager.add_messages_to_index(sample_messages)
        
        results = manager.search_similar("natural language processing", limit=10)
        
        # Results should be ordered by score (higher is better for inner product)
        if len(results) > 1:
            scores = [item[1] for item in results]
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_search_similar_limit_exceeds_total(self, temp_index_path):
        """Test search_similar when limit exceeds total vectors."""
        manager = FaissManager(index_path=temp_index_path)
        test_msg = Message(
            msg_id="msg1",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="Test",
            preview_text="Test",
            body_text="test message"
        )
        manager.add_messages_to_index([test_msg])
        
        # Request more results than available
        results = manager.search_similar("test", limit=100)
        assert len(results) == 1
    
    def test_save_index(self, temp_index_path, sample_messages):
        """Test _save_index saves both FAISS index and ID mapping."""
        manager = FaissManager(index_path=temp_index_path)
        manager.add_messages_to_index(sample_messages)
        
        manager._save_index()
        
        # Verify files were created
        index_file = Path(temp_index_path) / "faiss.index"
        mapping_file = Path(temp_index_path) / "id_mapping.json"
        
        assert index_file.exists()
        assert mapping_file.exists()
        
        # Verify mapping file content
        with open(mapping_file, 'r') as f:
            saved_mapping = json.load(f)
        
        assert len(saved_mapping) == 3
        assert saved_mapping["0"] == "msg1"
        assert saved_mapping["1"] == "msg2"
        assert saved_mapping["2"] == "msg3"
    
    def test_load_existing_index(self, temp_index_path, sample_messages):
        """Test loading existing FAISS index from disk."""
        # Create and save index
        manager1 = FaissManager(index_path=temp_index_path)
        manager1.add_messages_to_index(sample_messages)
        manager1._save_index()
        
        # Create new manager that should load existing index
        manager2 = FaissManager(index_path=temp_index_path)
        
        # Verify index was loaded correctly
        assert manager2.faiss_index.ntotal == 3
        assert len(manager2.id_mapping) == 3
        assert manager2.id_mapping[0] == "msg1"
        assert manager2.id_mapping[1] == "msg2"
        assert manager2.id_mapping[2] == "msg3"
        
        # Verify search works with loaded index
        results = manager2.search_similar("machine learning", limit=10)
        assert len(results) > 0
        assert results[0][0] == "msg1"
    
    def test_search_semantic_similarity(self, temp_index_path):
        """Test that semantic search finds similar content even with different words."""
        manager = FaissManager(index_path=temp_index_path)
        
        messages = [
            Message(
                msg_id="msg1",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-01",
                subject="Happy",
                preview_text="Happy",
                body_text="I am very happy and excited"
            ),
            Message(
                msg_id="msg2",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-02",
                subject="Weather",
                preview_text="Weather",
                body_text="The weather is rainy today"
            ),
            Message(
                msg_id="msg3",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-03",
                subject="Joyful",
                preview_text="Joyful",
                body_text="I feel joyful and delighted"
            )
        ]
        manager.add_messages_to_index(messages)
        
        # Search for "happy" - should find msg1 and msg3 (semantic similarity)
        results = manager.search_similar("feeling happy and glad", limit=10)
        
        # Top results should be msg1 or msg3 (both about positive emotions)
        top_ids = {results[0][0], results[1][0]}
        assert "msg1" in top_ids or "msg3" in top_ids
        
        # msg2 (about weather) should have lower similarity
        msg2_score = next(score for msg_id, score in results if msg_id == "msg2")
        happy_scores = [score for msg_id, score in results if msg_id in ["msg1", "msg3"]]
        assert msg2_score < max(happy_scores)
    
    def test_id_mapping_persistence(self, temp_index_path):
        """Test that ID mapping persists correctly across saves/loads."""
        manager1 = FaissManager(index_path=temp_index_path)
        
        # Add messages in batches
        msg_a = Message(
            msg_id="msg_a",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="A",
            preview_text="A",
            body_text="First batch message A"
        )
        msg_b = Message(
            msg_id="msg_b",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-02",
            subject="B",
            preview_text="B",
            body_text="Second batch message B"
        )
        manager1.add_messages_to_index([msg_a])
        manager1.add_messages_to_index([msg_b])
        manager1._save_index()
        
        # Load in new manager
        manager2 = FaissManager(index_path=temp_index_path)
        
        # Verify mapping is correct
        assert manager2.id_mapping[0] == "msg_a"
        assert manager2.id_mapping[1] == "msg_b"
        
        # Add more messages to loaded index
        msg_c = Message(
            msg_id="msg_c",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-03",
            subject="C",
            preview_text="C",
            body_text="Third batch message C"
        )
        manager2.add_messages_to_index([msg_c])
        
        # Verify new mapping
        assert manager2.id_mapping[2] == "msg_c"
    
    def test_search_similar_empty_query(self, temp_index_path, sample_messages):
        """Test search_similar with empty or whitespace-only queries."""
        manager = FaissManager(index_path=temp_index_path)
        manager.add_messages_to_index(sample_messages)
        
        # Test empty string
        results = manager.search_similar("", limit=10)
        assert results == []
        
        # Test whitespace-only string
        results = manager.search_similar("   ", limit=10)
        assert results == []
        
        # Test None-like behavior (empty after strip)
        results = manager.search_similar("\t\n  ", limit=10)
        assert results == []
    
    def test_add_messages_with_empty_text(self, temp_index_path):
        """Test add_messages_to_index with messages containing empty text."""
        manager = FaissManager(index_path=temp_index_path)
        
        messages = [
            Message(
                msg_id="msg1",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-01",
                subject="",
                preview_text="",
                body_text=""
            ),
            Message(
                msg_id="msg2",
                sender="test@example.com",
                recipients=["user@example.com"],
                date="2024-01-02",
                subject="Valid Subject",
                preview_text="Valid",
                body_text="Valid content"
            )
        ]
        
        manager.add_messages_to_index(messages)
        
        # Both messages should be added
        assert manager.faiss_index.ntotal == 2
        assert manager.id_mapping[0] == "msg1"
        assert manager.id_mapping[1] == "msg2"
    
    def test_force_recreate_prints(self, temp_index_path, capsys):
        """Test that force_recreate prints appropriate messages."""
        # Create initial index
        test_msg = Message(
            msg_id="test_msg",
            sender="test@example.com",
            recipients=["user@example.com"],
            date="2024-01-01",
            subject="Test",
            preview_text="Test",
            body_text="test content"
        )
        manager1 = FaissManager(index_path=temp_index_path)
        manager1.add_messages_to_index([test_msg])
        manager1._save_index()
        
        # Clear captured output
        capsys.readouterr()
        
        # Create new manager with force_recreate
        manager2 = FaissManager(index_path=temp_index_path, force_recreate=True)
        
        # Check that appropriate messages were printed
        captured = capsys.readouterr()
        assert "Created new empty FAISS index" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

