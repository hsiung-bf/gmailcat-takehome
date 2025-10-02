import pytest
import tempfile
import os
import sqlite3
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import List

from .message_manager import MessageManager
from ..types import Message
from .sqlite_utils import table_exists


class TestMessageManager:
    """Test cases for MessageManager class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            Message(
                msg_id="msg1",
                sender="sender1@example.com",
                recipients=["recipient1@example.com"],
                date="2023-01-01",
                subject="Test Subject 1",
                preview_text="Preview text for message 1",
                body_text="This is the full body text for message 1. It contains important information about testing."
            ),
            Message(
                msg_id="msg2",
                sender="sender2@example.com",
                recipients=["recipient2@example.com", "recipient3@example.com"],
                date="2023-01-02",
                subject="Test Subject 2",
                preview_text="Preview text for message 2",
                body_text="This is the full body text for message 2. It contains different information about testing and development."
            ),
            Message(
                msg_id="msg3",
                sender="sender3@example.com",
                recipients=["recipient4@example.com"],
                date="2023-01-03",
                subject="Different Subject",
                preview_text="Different preview",
                body_text="Different body content with unique keywords like python and database."
            )
        ]
    
    def test_init_default_path(self):
        """Test MessageManager initialization with default path."""
        with patch('email_categorizer.data_managers.message_manager.ensure_db_directory') as mock_ensure, \
             patch('email_categorizer.data_managers.message_manager.db_connection') as mock_conn:
            
            mock_conn.return_value.__enter__.return_value = MagicMock()
            
            manager = MessageManager()
            
            assert manager.db_path == "data/emails.db"
            mock_ensure.assert_called_once_with("data/emails.db")
    
    def test_init_custom_path(self, temp_db_path):
        """Test MessageManager initialization with custom path."""
        with patch('email_categorizer.data_managers.message_manager.ensure_db_directory') as mock_ensure, \
             patch('email_categorizer.data_managers.message_manager.db_connection') as mock_conn:
            
            mock_conn.return_value.__enter__.return_value = MagicMock()
            
            manager = MessageManager(db_path=temp_db_path)
            
            assert manager.db_path == temp_db_path
            mock_ensure.assert_called_once_with(temp_db_path)
    
    def test_database_creation(self, temp_db_path):
        """Test database and table creation."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Check that tables were created
        assert table_exists(temp_db_path, "messages")
        assert table_exists(temp_db_path, "messages_fts")
        
        # Check table structure
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(messages)")
            columns = [row[1] for row in cursor.fetchall()]
            
            expected_columns = ['msg_id', 'sender', 'recipients', 'date', 'subject', 'preview_text', 'body_text', 'created_at']
            assert all(col in columns for col in expected_columns)
            
            # Check FTS5 table
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts'")
            fts_sql = cursor.fetchone()[0]
            assert "USING fts5" in fts_sql
            assert "porter" in fts_sql
    
    def test_force_recreate(self, temp_db_path):
        """Test force_recreate functionality."""
        # Create initial manager and add some data
        manager1 = MessageManager(db_path=temp_db_path)
        
        # Add a message
        test_message = Message(
            msg_id="test_msg",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Test",
            preview_text="Preview",
            body_text="Body"
        )
        manager1.save_messages([test_message])
        
        # Verify message exists
        assert manager1.has_messages()
        
        # Create new manager with force_recreate
        manager2 = MessageManager(db_path=temp_db_path, force_recreate=True)
        
        # Verify tables were recreated and data is gone
        assert not manager2.has_messages()
    
    def test_has_messages_empty(self, temp_db_path):
        """Test has_messages() with empty database."""
        manager = MessageManager(db_path=temp_db_path)
        assert not manager.has_messages()
    
    def test_has_messages_with_data(self, temp_db_path, sample_messages):
        """Test has_messages() with data in database."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        assert manager.has_messages()
    
    def test_save_messages_empty_list(self, temp_db_path):
        """Test save_messages() with empty list."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Should not raise exception
        manager.save_messages([])
        assert not manager.has_messages()
    
    def test_save_messages_single(self, temp_db_path):
        """Test save_messages() with single message."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="single_msg",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Single Message",
            preview_text="Preview",
            body_text="Body content"
        )
        
        manager.save_messages([test_message])
        assert manager.has_messages()
        
        # Verify message was saved correctly
        retrieved = manager.get_messages_by_ids(["single_msg"])
        assert len(retrieved) == 1
        assert retrieved[0].msg_id == "single_msg"
        assert retrieved[0].sender == "sender@example.com"
    
    def test_save_messages_multiple(self, temp_db_path, sample_messages):
        """Test save_messages() with multiple messages."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        assert manager.has_messages()
        
        # Verify all messages were saved
        all_messages = manager.get_all_messages()
        assert len(all_messages) == 3
        
        # Check that messages are ordered by date DESC
        assert all_messages[0].date == "2023-01-03"  # Most recent first
        assert all_messages[2].date == "2023-01-01"  # Oldest last
    
    def test_save_messages_replace_existing(self, temp_db_path):
        """Test that save_messages() replaces existing messages."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Save initial message
        original_message = Message(
            msg_id="replace_test",
            sender="original@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Original Subject",
            preview_text="Original preview",
            body_text="Original body"
        )
        manager.save_messages([original_message])
        
        # Save updated message with same ID
        updated_message = Message(
            msg_id="replace_test",
            sender="updated@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-02",
            subject="Updated Subject",
            preview_text="Updated preview",
            body_text="Updated body"
        )
        manager.save_messages([updated_message])
        
        # Verify only one message exists and it's the updated one
        retrieved = manager.get_messages_by_ids(["replace_test"])
        assert len(retrieved) == 1
        assert retrieved[0].sender == "updated@example.com"
        assert retrieved[0].subject == "Updated Subject"
    
    def test_get_messages_by_ids_empty_list(self, temp_db_path):
        """Test get_messages_by_ids() with empty list."""
        manager = MessageManager(db_path=temp_db_path)
        result = manager.get_messages_by_ids([])
        assert result == []
    
    def test_get_messages_by_ids_nonexistent(self, temp_db_path):
        """Test get_messages_by_ids() with non-existent IDs."""
        manager = MessageManager(db_path=temp_db_path)
        result = manager.get_messages_by_ids(["nonexistent1", "nonexistent2"])
        assert result == []
    
    def test_get_messages_by_ids_existing(self, temp_db_path, sample_messages):
        """Test get_messages_by_ids() with existing IDs."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        result = manager.get_messages_by_ids(["msg1", "msg3"])
        assert len(result) == 2
        
        # Verify correct messages returned
        msg_ids = {msg.msg_id for msg in result}
        assert "msg1" in msg_ids
        assert "msg3" in msg_ids
    
    def test_get_all_messages_empty(self, temp_db_path):
        """Test get_all_messages() with empty database."""
        manager = MessageManager(db_path=temp_db_path)
        result = manager.get_all_messages()
        assert result == []
    
    def test_get_all_messages_ordered(self, temp_db_path, sample_messages):
        """Test get_all_messages() returns messages ordered by date DESC."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        result = manager.get_all_messages()
        assert len(result) == 3
        
        # Verify ordering (most recent first)
        assert result[0].date == "2023-01-03"
        assert result[1].date == "2023-01-02"
        assert result[2].date == "2023-01-01"
    
    def test_search_by_keywords_no_matches(self, temp_db_path, sample_messages):
        """Test search_by_keywords() with no matching results."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        result = manager.search_by_keywords("nonexistentkeyword")
        assert result == []
    
    def test_search_by_keywords_subject_match(self, temp_db_path, sample_messages):
        """Test search_by_keywords() matching subject."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        result = manager.search_by_keywords("Test Subject")
        assert len(result) >= 1
        
        # Verify results are (Message, score) tuples
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(item[0], Message) and isinstance(item[1], float) for item in result)
    
    def test_search_by_keywords_body_match(self, temp_db_path, sample_messages):
        """Test search_by_keywords() matching body content."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        result = manager.search_by_keywords("python database")
        assert len(result) >= 1
        
        # Should find msg3 which contains both "python" and "database"
        found_msg_ids = [msg[0].msg_id for msg in result]
        assert "msg3" in found_msg_ids
    
    def test_search_by_keywords_sender_match(self, temp_db_path, sample_messages):
        """Test search_by_keywords() matching sender."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Search for part of email address (FTS5 has issues with @ symbols)
        result = manager.search_by_keywords("sender1 example")
        assert len(result) >= 1
        
        # Should find msg1
        found_msg_ids = [msg[0].msg_id for msg in result]
        assert "msg1" in found_msg_ids
    
    def test_search_by_keywords_limit(self, temp_db_path, sample_messages):
        """Test search_by_keywords() with limit parameter."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Search for common term that should match multiple messages
        result = manager.search_by_keywords("testing", limit=2)
        assert len(result) <= 2
    
    def test_search_by_keywords_ordering(self, temp_db_path, sample_messages):
        """Test search_by_keywords() returns results ordered by relevance."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Search for term that appears in multiple messages
        result = manager.search_by_keywords("testing")
        assert len(result) >= 1
        
        # Results should be ordered by BM25 score (lower is better)
        if len(result) > 1:
            scores = [item[1] for item in result]
            assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    
    def test_row_to_message_conversion(self, temp_db_path):
        """Test _row_to_message() data conversion."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Create a test message
        test_message = Message(
            msg_id="conversion_test",
            sender="test@example.com",
            recipients=["rec1@example.com", "rec2@example.com"],
            date="2023-01-01",
            subject="Test Subject",
            preview_text="Test Preview",
            body_text="Test Body"
        )
        manager.save_messages([test_message])
        
        # Get the raw row from database
        with sqlite3.connect(temp_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE msg_id = ?", ("conversion_test",))
            row = cursor.fetchone()
        
        # Test conversion
        converted_message = manager._row_to_message(row)
        
        assert converted_message.msg_id == "conversion_test"
        assert converted_message.sender == "test@example.com"
        assert converted_message.recipients == ["rec1@example.com", "rec2@example.com"]
        assert converted_message.date == "2023-01-01"
        assert converted_message.subject == "Test Subject"
        assert converted_message.preview_text == "Test Preview"
        assert converted_message.body_text == "Test Body"
    
    def test_fts5_triggers_functionality(self, temp_db_path):
        """Test that FTS5 triggers work correctly."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Add a message
        test_message = Message(
            msg_id="trigger_test",
            sender="trigger@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Trigger Test Subject",
            preview_text="Trigger preview",
            body_text="Trigger body with special keywords"
        )
        manager.save_messages([test_message])
        
        # Search should find the message
        result = manager.search_by_keywords("Trigger Test")
        assert len(result) >= 1
        assert result[0][0].msg_id == "trigger_test"
        
        # Test that triggers work by using the manager's own methods
        # Update the message using the manager (which should trigger FTS5 update)
        updated_message = Message(
            msg_id="trigger_test",
            sender="trigger@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Updated Trigger Subject",
            preview_text="Trigger preview",
            body_text="Trigger body with special keywords"
        )
        manager.save_messages([updated_message])
        
        # Search should still work with updated content
        result = manager.search_by_keywords("Updated Trigger")
        assert len(result) >= 1
        assert result[0][0].subject == "Updated Trigger Subject"
        
        # Test deletion by removing the message entirely
        # We'll test this by creating a new manager instance to verify persistence
        manager2 = MessageManager(db_path=temp_db_path)
        result = manager2.search_by_keywords("Updated Trigger")
        assert len(result) >= 1
    
    def test_complex_search_queries(self, temp_db_path, sample_messages):
        """Test complex search queries with multiple terms."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Test AND search (should match messages containing both terms)
        result = manager.search_by_keywords("testing AND development")
        found_msg_ids = [msg[0].msg_id for msg in result]
        assert "msg2" in found_msg_ids  # msg2 contains both "testing" and "development"
        
        # Test OR search
        result = manager.search_by_keywords("python OR different")
        found_msg_ids = [msg[0].msg_id for msg in result]
        assert "msg3" in found_msg_ids  # msg3 contains both "python" and "different"
        
        # Test phrase search
        result = manager.search_by_keywords('"full body text"')
        found_msg_ids = [msg[0].msg_id for msg in result]
        assert len(found_msg_ids) >= 1  # Should find messages with exact phrase

    def test_drop_tables_functionality(self, temp_db_path):
        """Test _drop_tables() method functionality."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Add some data first
        test_message = Message(
            msg_id="drop_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Test",
            preview_text="Preview",
            body_text="Body"
        )
        manager.save_messages([test_message])
        
        # Verify data exists
        assert manager.has_messages()
        assert table_exists(temp_db_path, "messages")
        assert table_exists(temp_db_path, "messages_fts")
        
        # Test _drop_tables directly
        with sqlite3.connect(temp_db_path) as conn:
            manager._drop_tables(conn)
        
        # Verify tables are gone
        assert not table_exists(temp_db_path, "messages")
        assert not table_exists(temp_db_path, "messages_fts")
    
    def test_create_tables_functionality(self, temp_db_path):
        """Test _create_tables() method functionality."""
        # Start with empty database
        manager = MessageManager(db_path=temp_db_path)
        
        # Verify tables exist
        assert table_exists(temp_db_path, "messages")
        assert table_exists(temp_db_path, "messages_fts")
        
        # Test table structure
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check messages table structure
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_columns = {
                'msg_id': 'TEXT',
                'sender': 'TEXT', 
                'recipients': 'TEXT',
                'date': 'TEXT',
                'subject': 'TEXT',
                'preview_text': 'TEXT',
                'body_text': 'TEXT',
                'created_at': 'TIMESTAMP'
            }
            for col, col_type in expected_columns.items():
                assert col in columns
                assert columns[col] == col_type
            
            # Check FTS5 table
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts'")
            fts_sql = cursor.fetchone()[0]
            assert "USING fts5" in fts_sql
            assert "porter" in fts_sql
            assert "msg_id UNINDEXED" in fts_sql
            
            # Check triggers exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            triggers = [row[0] for row in cursor.fetchall()]
            expected_triggers = ['messages_fts_insert', 'messages_fts_delete', 'messages_fts_update']
            for trigger in expected_triggers:
                assert trigger in triggers
    
    def test_ensure_database_exists_with_recreate(self, temp_db_path):
        """Test _ensure_database_exists() with force_recreate=True."""
        # Create initial database with data
        manager1 = MessageManager(db_path=temp_db_path)
        test_message = Message(
            msg_id="recreate_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Test",
            preview_text="Preview",
            body_text="Body"
        )
        manager1.save_messages([test_message])
        assert manager1.has_messages()
        
        # Create new manager with force_recreate
        manager2 = MessageManager(db_path=temp_db_path, force_recreate=True)
        assert not manager2.has_messages()
    
    def test_save_messages_with_none_values(self, temp_db_path):
        """Test save_messages() with None values in message fields."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="none_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Test",
            preview_text=None,  # None value
            body_text=None      # None value
        )
        
        # Should not raise exception
        manager.save_messages([test_message])
        
        # Verify message was saved
        retrieved = manager.get_messages_by_ids(["none_test"])
        assert len(retrieved) == 1
        assert retrieved[0].preview_text is None
        assert retrieved[0].body_text is None
    
    def test_save_messages_with_empty_strings(self, temp_db_path):
        """Test save_messages() with empty string values."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="empty_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="",  # Empty string
            preview_text="",  # Empty string
            body_text=""      # Empty string
        )
        
        manager.save_messages([test_message])
        
        # Verify message was saved
        retrieved = manager.get_messages_by_ids(["empty_test"])
        assert len(retrieved) == 1
        assert retrieved[0].subject == ""
        assert retrieved[0].preview_text == ""
        assert retrieved[0].body_text == ""
    
    def test_save_messages_with_special_characters(self, temp_db_path):
        """Test save_messages() with special characters and unicode."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="special_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            preview_text="Unicode: 你好世界 🌍 émojis",
            body_text="SQL injection attempt: '; DROP TABLE messages; --"
        )
        
        manager.save_messages([test_message])
        
        # Verify message was saved correctly
        retrieved = manager.get_messages_by_ids(["special_test"])
        assert len(retrieved) == 1
        assert retrieved[0].subject == "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        assert retrieved[0].preview_text == "Unicode: 你好世界 🌍 émojis"
        assert retrieved[0].body_text == "SQL injection attempt: '; DROP TABLE messages; --"
    
    def test_search_by_keywords_empty_query(self, temp_db_path, sample_messages):
        """Test search_by_keywords() with empty query."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Empty query should raise an FTS5 syntax error
        with pytest.raises(sqlite3.OperationalError, match="fts5: syntax error"):
            manager.search_by_keywords("")
        
        # Whitespace-only query should also raise an error
        with pytest.raises(sqlite3.OperationalError, match="fts5: syntax error"):
            manager.search_by_keywords("   ")
    
    def test_search_by_keywords_special_characters(self, temp_db_path):
        """Test search_by_keywords() with special characters in query."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="special_search_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Test with special chars",
            preview_text="Preview",
            body_text="This message contains special characters like @#$% and unicode 你好"
        )
        manager.save_messages([test_message])
        
        # Test searching for special characters
        result = manager.search_by_keywords("special")
        assert len(result) >= 1
        assert result[0][0].msg_id == "special_search_test"
        
        # Test searching for unicode
        result = manager.search_by_keywords("你好")
        assert len(result) >= 1
        assert result[0][0].msg_id == "special_search_test"
    
    def test_search_by_keywords_very_long_query(self, temp_db_path, sample_messages):
        """Test search_by_keywords() with very long query."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Very long query
        long_query = "testing " * 1000
        result = manager.search_by_keywords(long_query)
        assert len(result) >= 1  # Should still find matches
    
    def test_row_to_message_with_malformed_json(self, temp_db_path):
        """Test _row_to_message() with malformed JSON in recipients."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Manually insert a row with malformed JSON
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (msg_id, sender, recipients, date, subject, preview_text, body_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "malformed_test",
                "test@example.com", 
                "malformed json",  # Not valid JSON
                "2023-01-01",
                "Test",
                "Preview",
                "Body"
            ))
            conn.commit()
        
        # This should raise a JSON decode error
        with pytest.raises(json.JSONDecodeError):
            manager.get_messages_by_ids(["malformed_test"])
    
    def test_database_connection_error_handling(self, temp_db_path):
        """Test error handling when database connection fails."""
        # Create a manager with invalid path
        invalid_path = "/invalid/path/that/does/not/exist/database.db"
        
        with pytest.raises((sqlite3.OperationalError, OSError)):
            MessageManager(db_path=invalid_path)
    
    def test_fts5_tokenizer_behavior(self, temp_db_path):
        """Test FTS5 porter tokenizer behavior (stemming)."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="stemming_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="Running and runners",
            preview_text="Preview",
            body_text="The runner runs quickly while running"
        )
        manager.save_messages([test_message])
        
        # Test that stemming works - "run" should match "running", "runs", "runner"
        result = manager.search_by_keywords("run")
        assert len(result) >= 1
        assert result[0][0].msg_id == "stemming_test"
        
        # Test plural/singular matching
        result = manager.search_by_keywords("runner")
        assert len(result) >= 1
        assert result[0][0].msg_id == "stemming_test"
    
    def test_search_by_keywords_case_insensitive(self, temp_db_path):
        """Test that FTS5 search is case insensitive."""
        manager = MessageManager(db_path=temp_db_path)
        
        test_message = Message(
            msg_id="case_test",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date="2023-01-01",
            subject="UPPERCASE and lowercase",
            preview_text="Preview",
            body_text="MiXeD cAsE TeXt"
        )
        manager.save_messages([test_message])
        
        # Test case insensitive matching
        result = manager.search_by_keywords("uppercase")
        assert len(result) >= 1
        assert result[0][0].msg_id == "case_test"
        
        result = manager.search_by_keywords("LOWERCASE")
        assert len(result) >= 1
        assert result[0][0].msg_id == "case_test"
        
        result = manager.search_by_keywords("mixed")
        assert len(result) >= 1
        assert result[0][0].msg_id == "case_test"
    
    def test_get_messages_by_ids_with_duplicate_ids(self, temp_db_path, sample_messages):
        """Test get_messages_by_ids() with duplicate IDs in input."""
        manager = MessageManager(db_path=temp_db_path)
        manager.save_messages(sample_messages)
        
        # Request same ID multiple times
        result = manager.get_messages_by_ids(["msg1", "msg1", "msg1"])
        
        # Should return only one instance
        assert len(result) == 1
        assert result[0].msg_id == "msg1"
    
    def test_save_messages_batch_large(self, temp_db_path):
        """Test save_messages() with a large batch of messages."""
        manager = MessageManager(db_path=temp_db_path)
        
        # Create 1000 messages
        large_batch = []
        for i in range(1000):
            message = Message(
                msg_id=f"batch_msg_{i}",
                sender=f"sender{i}@example.com",
                recipients=[f"recipient{i}@example.com"],
                date="2023-01-01",
                subject=f"Batch Message {i}",
                preview_text=f"Preview {i}",
                body_text=f"Body content for message {i}"
            )
            large_batch.append(message)
        
        # Save all at once
        manager.save_messages(large_batch)
        
        # Verify all were saved
        assert manager.has_messages()
        all_messages = manager.get_all_messages()
        assert len(all_messages) == 1000
        
        # Verify specific messages
        result = manager.get_messages_by_ids(["batch_msg_0", "batch_msg_999"])
        assert len(result) == 2
        msg_ids = {msg.msg_id for msg in result}
        assert "batch_msg_0" in msg_ids
        assert "batch_msg_999" in msg_ids


if __name__ == "__main__":
    pytest.main([__file__])
