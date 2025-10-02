import pytest
import tempfile
import os
import sqlite3
import yaml
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from .category_manager import CategoryManager
from ..types import Category, ClassificationResult
from .sqlite_utils import table_exists


class TestCategoryManager:
    """Test cases for CategoryManager class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture(autouse=True)
    def mock_llm_client(self):
        """Mock LLMClient for all tests."""
        with patch('email_categorizer.data_managers.category_manager.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture(autouse=True)
    def disable_foreign_keys(self, temp_db_path):
        """Disable foreign key constraints for testing."""
        with sqlite3.connect(temp_db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.commit()
    
    @pytest.fixture
    def mock_llm_keywords(self):
        """Mock LLM response for keyword generation."""
        return {
            "keywords": ["invoice", "billing", "payment", "receipt", "transaction"]
        }
    
    @pytest.fixture
    def mock_prompts_yaml(self):
        """Mock prompts.yaml content."""
        return """
keywords_prompt: |
  Generate keywords for category "{category_name}": {category_description}
  Return JSON: {{"keywords": ["keyword1", "keyword2"]}}
"""
    
    # ===== Slug Generation Tests =====
    
    def test_generate_slug_basic(self, temp_db_path):
        """Test basic slug generation from category name."""
        manager = CategoryManager(db_path=temp_db_path)
        
        assert manager.generate_slug("Work Emails") == "work-emails"
        assert manager.generate_slug("Personal") == "personal"
        assert manager.generate_slug("Important_Stuff") == "important-stuff"
    
    def test_generate_slug_special_characters(self, temp_db_path):
        """Test slug generation with special characters."""
        manager = CategoryManager(db_path=temp_db_path)
        
        assert manager.generate_slug("Bills & Invoices!") == "bills-invoices"
        assert manager.generate_slug("To-Do Items") == "to-do-items"
        assert manager.generate_slug("100% Complete") == "100-complete"
    
    def test_generate_slug_multiple_spaces(self, temp_db_path):
        """Test slug generation with multiple spaces and hyphens."""
        manager = CategoryManager(db_path=temp_db_path)
        
        assert manager.generate_slug("Work   Emails") == "work-emails"
        assert manager.generate_slug("Test---Category") == "test-category"
        assert manager.generate_slug("-Leading Trailing-") == "leading-trailing"
    
    def test_generate_slug_edge_cases(self, temp_db_path):
        """Test slug generation with edge cases."""
        manager = CategoryManager(db_path=temp_db_path)
        
        assert manager.generate_slug("") == ""
        assert manager.generate_slug("   ") == ""
        assert manager.generate_slug("!!!") == ""
        assert manager.generate_slug("123") == "123"
        assert manager.generate_slug("A-B-C") == "a-b-c"
    
    # ===== Initialization Tests =====
    
    def test_init_with_prompts_loading(self, temp_db_path, mock_prompts_yaml, mock_llm_client):
        """Test initialization with successful prompts loading."""
        with patch('builtins.open', mock_open(read_data=mock_prompts_yaml)):
            with patch('yaml.safe_load') as mock_yaml_load:
                mock_yaml_load.return_value = {'keywords_prompt': 'test prompt {category_name} {category_description}'}
                
                manager = CategoryManager(db_path=temp_db_path)
                assert manager.keywords_prompt == 'test prompt {category_name} {category_description}'
    
    def test_init_prompts_file_not_found(self, temp_db_path, mock_llm_client):
        """Test initialization when prompts file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(ValueError, match="Error loading prompt templates"):
                CategoryManager(db_path=temp_db_path)
    
    def test_init_prompts_yaml_error(self, temp_db_path, mock_prompts_yaml, mock_llm_client):
        """Test initialization when YAML parsing fails."""
        with patch('builtins.open', mock_open(read_data=mock_prompts_yaml)):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
                with pytest.raises(ValueError, match="Error loading prompt templates"):
                    CategoryManager(db_path=temp_db_path)
    
    def test_init_prompts_missing_key(self, temp_db_path, mock_llm_client):
        """Test initialization when required prompt key is missing."""
        with patch('builtins.open', mock_open(read_data="other_key: value")):
            with patch('yaml.safe_load', return_value={'other_key': 'value'}):
                with pytest.raises(ValueError, match="Error loading prompt templates"):
                    CategoryManager(db_path=temp_db_path)

    # ===== Database Initialization Tests =====
    
    def test_database_creation(self, temp_db_path):
        """Test database and table creation."""
        manager = CategoryManager(db_path=temp_db_path)
        
        # Check that tables were created
        assert table_exists(temp_db_path, "categories")
        assert table_exists(temp_db_path, "message_categories")
        
        # Check categories table structure
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(categories)")
            columns = [row[1] for row in cursor.fetchall()]
            
            expected_columns = ['slug', 'name', 'description', 'keywords', 'created_at']
            assert all(col in columns for col in expected_columns)
    
    def test_database_indices(self, temp_db_path):
        """Test that necessary indices are created."""
        manager = CategoryManager(db_path=temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='message_categories'
            """)
            indices = [row[0] for row in cursor.fetchall()]
            
            assert 'idx_message_categories_msg_id' in indices
            assert 'idx_message_categories_category_result' in indices
    
    def test_force_recreate(self, temp_db_path):
        """Test force_recreate functionality drops existing tables."""
        # Create initial manager and add a category
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager1 = CategoryManager(db_path=temp_db_path)
            manager1.create_category("Test Category", "Test description")
            
            # Verify category exists
            assert len(manager1.get_all_categories()) == 1
            
            # Create new manager with force_recreate
            manager2 = CategoryManager(db_path=temp_db_path, force_recreate=True)
            
            # Verify tables were recreated and data is gone
            assert len(manager2.get_all_categories()) == 0
    
    # ===== Keyword Generation Tests =====
    
    def test_generate_keywords_success(self, temp_db_path, mock_llm_keywords, mock_llm_client):
        """Test keyword generation with successful LLM response."""
        import json
        mock_llm_client.chat_completion.return_value = json.dumps(mock_llm_keywords)
        
        manager = CategoryManager(db_path=temp_db_path)
        # Mock the prompt template to avoid KeyError
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Billing", "Invoices and receipts")
        
        assert keywords == mock_llm_keywords["keywords"][:CategoryManager.MAX_KEYWORDS]
        assert len(keywords) <= CategoryManager.MAX_KEYWORDS
    
    def test_generate_keywords_max_limit(self, temp_db_path, mock_llm_client):
        """Test that keywords are limited to MAX_KEYWORDS."""
        # Mock response with more than MAX_KEYWORDS
        import json
        many_keywords = [f"keyword{i}" for i in range(20)]
        mock_llm_client.chat_completion.return_value = json.dumps({"keywords": many_keywords})
        
        manager = CategoryManager(db_path=temp_db_path)
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Test", "Test description")
        
        assert len(keywords) == CategoryManager.MAX_KEYWORDS
    
    def test_generate_keywords_llm_failure(self, temp_db_path, mock_llm_client):
        """Test keyword generation when LLM fails."""
        mock_llm_client.chat_completion.return_value = "Invalid JSON response"
        
        manager = CategoryManager(db_path=temp_db_path)
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Test", "Test description")
        
        assert keywords is None
    
    def test_generate_keywords_llm_exception(self, temp_db_path, mock_llm_client):
        """Test keyword generation when LLM throws exception."""
        mock_llm_client.chat_completion.side_effect = Exception("LLM error")
        
        manager = CategoryManager(db_path=temp_db_path)
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Test", "Test description")
        
        assert keywords is None
    
    def test_generate_keywords_malformed_json(self, temp_db_path, mock_llm_client):
        """Test keyword generation with malformed JSON response."""
        mock_llm_client.chat_completion.return_value = '{"keywords": [invalid json}'
        
        manager = CategoryManager(db_path=temp_db_path)
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Test", "Test description")
        
        assert keywords is None
    
    def test_generate_keywords_missing_keywords_key(self, temp_db_path, mock_llm_client):
        """Test keyword generation when JSON response missing keywords key."""
        mock_llm_client.chat_completion.return_value = '{"other_key": ["value"]}'
        
        manager = CategoryManager(db_path=temp_db_path)
        manager.keywords_prompt = "Test prompt for {category_name}: {category_description}"
        keywords = manager._generate_keywords("Test", "Test description")
        
        assert keywords is None
    
    # ===== Category CRUD Tests =====
    
    def test_create_category(self, temp_db_path, mock_llm_keywords):
        """Test creating a category."""
        with patch.object(CategoryManager, '_generate_keywords') as mock_keywords:
            mock_keywords.return_value = mock_llm_keywords["keywords"]
            manager = CategoryManager(db_path=temp_db_path)
            
            category = manager.create_category("Work Emails", "Emails related to work")
            
            assert category.name == "Work Emails"
            assert category.description == "Emails related to work"
            assert category.slug == "work-emails"
            assert category.keywords == mock_llm_keywords["keywords"]
    
    def test_create_category_duplicate_slug(self, temp_db_path):
        """Test that creating duplicate category raises ValueError."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "First description")
            
            # Should raise ValueError for duplicate
            with pytest.raises(ValueError, match="already exists"):
                manager.create_category("Test Category", "Different description")
    
    def test_create_category_without_keywords(self, temp_db_path):
        """Test creating category when keyword generation fails."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=None):
            manager = CategoryManager(db_path=temp_db_path)
            
            category = manager.create_category("Test", "Test description")
            
            assert category.keywords is None
    
    def test_create_category_empty_name(self, temp_db_path):
        """Test creating category with empty name."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            category = manager.create_category("", "Empty name category")
            
            assert category.slug == ""
            assert category.name == ""
    
    def test_create_category_long_name(self, temp_db_path):
        """Test creating category with very long name."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            long_name = "A" * 1000  # Very long name
            category = manager.create_category(long_name, "Long name category")
            
            assert category.name == long_name
            assert len(category.slug) > 0
    
    def test_get_category_existing(self, temp_db_path):
        """Test retrieving an existing category."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            created = manager.create_category("Test Category", "Test description")
            retrieved = manager.get_category("test-category")
            
            assert retrieved is not None
            assert retrieved.name == created.name
            assert retrieved.description == created.description
            assert retrieved.slug == created.slug
            assert retrieved.keywords == created.keywords
    
    def test_get_category_with_null_keywords(self, temp_db_path):
        """Test retrieving category with NULL keywords in database."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=None):
            manager = CategoryManager(db_path=temp_db_path)
            
            # Create category without keywords
            created = manager.create_category("Test Category", "Test description")
            retrieved = manager.get_category("test-category")
            
            assert retrieved is not None
            assert retrieved.keywords is None
    
    def test_get_category_nonexistent(self, temp_db_path):
        """Test retrieving a non-existent category."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=None):
            manager = CategoryManager(db_path=temp_db_path)
            
            result = manager.get_category("nonexistent-slug")
            
            assert result is None
    
    def test_get_all_categories_empty(self, temp_db_path):
        """Test get_all_categories with no categories."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=None):
            manager = CategoryManager(db_path=temp_db_path)
            
            categories = manager.get_all_categories()
            
            assert categories == []
    
    def test_get_all_categories_multiple(self, temp_db_path):
        """Test get_all_categories with multiple categories."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Work", "Work emails")
            manager.create_category("Personal", "Personal emails")
            manager.create_category("Bills", "Bills and invoices")
            
            categories = manager.get_all_categories()
            
            assert len(categories) == 3
            # Should be ordered by name
            assert categories[0].name == "Bills"
            assert categories[1].name == "Personal"
            assert categories[2].name == "Work"
    
    def test_delete_category_existing(self, temp_db_path):
        """Test deleting an existing category."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            result = manager.delete_category("test-category")
            
            assert result is True
            assert manager.get_category("test-category") is None
    
    def test_delete_category_nonexistent(self, temp_db_path):
        """Test deleting a non-existent category."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=None):
            manager = CategoryManager(db_path=temp_db_path)
            
            result = manager.delete_category("nonexistent-slug")
            
            assert result is False
    
    def test_delete_all_categories(self, temp_db_path):
        """Test deleting all categories."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Category 1", "Description 1")
            manager.create_category("Category 2", "Description 2")
            manager.create_category("Category 3", "Description 3")
            
            count = manager.delete_all_categories()
            
            assert count == 3
            assert len(manager.get_all_categories()) == 0
    
    # ===== Classification Results Tests =====
    
    def test_save_classification_results(self, temp_db_path):
        """Test saving classification results."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "Matches criteria"),
                "msg2": ClassificationResult("msg2", "test-category", False, "Doesn't match"),
                "msg3": ClassificationResult("msg3", "test-category", True, "Matches")
            }
            
            manager.save_classification_results("test-category", results)
            
            # Verify results were saved
            message_ids = manager.get_category_message_ids("test-category")
            assert len(message_ids) == 2
            assert "msg1" in message_ids
            assert "msg3" in message_ids
    
    def test_save_classification_results_empty(self, temp_db_path):
        """Test saving empty classification results."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            # Should not raise exception
            manager.save_classification_results("test-category", {})
            
            message_ids = manager.get_category_message_ids("test-category")
            assert len(message_ids) == 0
    
    def test_save_classification_results_with_none(self, temp_db_path):
        """Test saving classification results with None values."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "Matches"),
                "msg2": None,  # Should be filtered out
                "msg3": ClassificationResult("msg3", "test-category", False, "Doesn't match")
            }
            
            manager.save_classification_results("test-category", results)
            
            # Verify only non-None results were saved
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM message_categories")
                count = cursor.fetchone()[0]
                
                assert count == 2
    
    def test_save_classification_results_replace_existing(self, temp_db_path):
        """Test that saving classification results replaces existing ones."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            # Save initial results
            initial_results = {
                "msg1": ClassificationResult("msg1", "test-category", False, "Initial")
            }
            manager.save_classification_results("test-category", initial_results)
            
            # Update with new results
            updated_results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "Updated")
            }
            manager.save_classification_results("test-category", updated_results)
            
            # Verify message is now in category
            message_ids = manager.get_category_message_ids("test-category")
            assert "msg1" in message_ids
    
    def test_get_category_message_ids_empty(self, temp_db_path):
        """Test get_category_message_ids with no messages."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            message_ids = manager.get_category_message_ids("test-category")
            
            assert message_ids == []
    
    def test_get_category_message_ids_ordered(self, temp_db_path):
        """Test that get_category_message_ids returns results in consistent order."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            # Save multiple results at once
            results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "First"),
                "msg2": ClassificationResult("msg2", "test-category", True, "Second"),
                "msg3": ClassificationResult("msg3", "test-category", True, "Third")
            }
            manager.save_classification_results("test-category", results)
            
            message_ids = manager.get_category_message_ids("test-category")
            
            # Should return all messages in the category
            assert len(message_ids) == 3
            assert "msg1" in message_ids
            assert "msg2" in message_ids
            assert "msg3" in message_ids
            
            # Should return consistent ordering (multiple calls should return same order)
            message_ids2 = manager.get_category_message_ids("test-category")
            assert message_ids == message_ids2
    
    def test_get_category_message_ids_only_in_category(self, temp_db_path):
        """Test that get_category_message_ids only returns messages in category."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "In category"),
                "msg2": ClassificationResult("msg2", "test-category", False, "Not in category"),
                "msg3": ClassificationResult("msg3", "test-category", True, "In category"),
                "msg4": ClassificationResult("msg4", "test-category", False, "Not in category")
            }
            
            manager.save_classification_results("test-category", results)
            
            message_ids = manager.get_category_message_ids("test-category")
            
            assert len(message_ids) == 2
            assert "msg1" in message_ids
            assert "msg3" in message_ids
            assert "msg2" not in message_ids
            assert "msg4" not in message_ids
    
    # ===== Integration Tests =====
    
    def test_cascade_delete_category(self, temp_db_path):
        """Test that deleting a category cascades to message_categories."""
        with patch.object(CategoryManager, '_generate_keywords', return_value=["keyword1"]):
            manager = CategoryManager(db_path=temp_db_path)
            
            manager.create_category("Test Category", "Test description")
            
            results = {
                "msg1": ClassificationResult("msg1", "test-category", True, "Matches"),
                "msg2": ClassificationResult("msg2", "test-category", True, "Matches")
            }
            manager.save_classification_results("test-category", results)
            
            # Delete the category
            manager.delete_category("test-category")
            
            # Verify classification results were also deleted
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM message_categories WHERE category_slug = ?", 
                             ("test-category",))
                count = cursor.fetchone()[0]
                
                assert count == 0


if __name__ == "__main__":
    pytest.main([__file__])

