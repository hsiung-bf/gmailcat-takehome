"""
Tests for email ingestion pipeline.

Tests the InboxIngestor class and email body processing functionality.
"""

import pytest
import tempfile
import os
import json
import base64
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from .ingestion import InboxIngestor
from .process_email_body import (
    process_email_body,
    decode_body_base64,
    html_to_text
)
from ..types import Message


class TestProcessEmailBody:
    """Test email body processing functions."""
    
    def test_decode_body_base64_success(self):
        """Test successful base64 decoding."""
        test_text = "Hello, this is a test email body!"
        encoded = base64.b64encode(test_text.encode('utf-8')).decode('utf-8')
        
        result = decode_body_base64(encoded)
        
        assert result == test_text
    
    def test_decode_body_base64_invalid_base64(self):
        """Test base64 decoding with invalid base64 string."""
        with pytest.raises(ValueError, match="Failed to decode base64 body"):
            decode_body_base64("invalid-base64-string!")
    
    def test_decode_body_base64_empty_string(self):
        """Test base64 decoding with empty string."""
        result = decode_body_base64("")
        assert result == ""
    
    def test_html_to_text_basic(self):
        """Test basic HTML to text conversion."""
        html = "<p>Hello <strong>world</strong>!</p>"
        result = html_to_text(html)
        
        assert "Hello world!" in result
        assert "<" not in result  # No HTML tags should remain
    
    def test_html_to_text_empty(self):
        """Test HTML to text conversion with empty HTML."""
        result = html_to_text("")
        assert result == ""
    
    def test_html_to_text_none(self):
        """Test HTML to text conversion with None input."""
        result = html_to_text(None)
        assert result == ""
    
    def test_html_to_text_with_links(self):
        """Test HTML to text conversion with links."""
        html = '<p>Check out <a href="https://example.com">this link</a> for more info.</p>'
        result = html_to_text(html)
        
        assert "Check out" in result
        assert "this link (https://example.com)" in result
        assert "<a" not in result
    
    def test_html_to_text_with_images(self):
        """Test HTML to text conversion with images."""
        html = '<p>Here is an image: <img src="test.jpg" alt="Test image"> and another <img src="test2.jpg"></p>'
        result = html_to_text(html)
        
        assert "[Image: Test image]" in result
        assert "[Image]" in result
        assert "<img" not in result
    
    def test_html_to_text_with_lists(self):
        """Test HTML to text conversion with lists."""
        html = """
        <ul>
            <li>First item</li>
            <li>Second item</li>
        </ul>
        """
        result = html_to_text(html)
        
        assert "• First item" in result
        assert "• Second item" in result
        assert "<li>" not in result
    
    def test_html_to_text_removes_scripts(self):
        """Test that script and style tags are removed."""
        html = """
        <p>Visible content</p>
        <script>alert('hidden');</script>
        <style>body { color: red; }</style>
        <p>More visible content</p>
        """
        result = html_to_text(html)
        
        assert "Visible content" in result
        assert "More visible content" in result
        assert "alert" not in result
        assert "color: red" not in result
    
    def test_html_to_text_whitespace_cleanup(self):
        """Test that excessive whitespace is cleaned up."""
        html = """
        <p>First paragraph</p>
        
        
        <p>Second paragraph</p>   <p>Third paragraph</p>
        """
        result = html_to_text(html)
        
        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in result
        # Should have single spaces instead of multiple
        assert "  " not in result
    
    def test_process_email_body_full_pipeline(self):
        """Test the complete email body processing pipeline."""
        test_text = "<p>Hello <strong>world</strong>!</p>"
        encoded = base64.b64encode(test_text.encode('utf-8')).decode('utf-8')
        
        full_text, truncated_text = process_email_body(encoded, max_chars=10)
        
        assert "Hello world!" in full_text
        assert len(truncated_text) <= 13  # 10 chars + "..."
        assert truncated_text.endswith("...")
        # The truncated text should be a prefix of the full text (minus the "...")
        assert truncated_text[:-3] in full_text
    
    def test_process_email_body_truncation(self):
        """Test email body processing with truncation."""
        test_text = "This is a very long email body that should be truncated when max_chars is small."
        encoded = base64.b64encode(test_text.encode('utf-8')).decode('utf-8')
        
        full_text, truncated_text = process_email_body(encoded, max_chars=20)
        
        assert len(truncated_text) <= 23  # 20 chars + "..."
        assert truncated_text.endswith("...")
        assert truncated_text.startswith("This is a very")
    
    def test_process_email_body_invalid_base64(self):
        """Test email body processing with invalid base64."""
        with pytest.raises(ValueError, match="Failed to decode base64 body"):
            process_email_body("invalid-base64")


class TestInboxIngestor:
    """Test InboxIngestor class."""
    
    @pytest.fixture
    def temp_file_path(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def sample_jsonl_content(self):
        """Sample JSONL content for testing."""
        return """{"id": "msg1", "subject": "Test Subject", "from": "test@example.com", "to": ["recipient@example.com"], "snippet": "Test snippet", "body": "SGVsbG8gd29ybGQ=", "date": "2025-01-01T00:00:00Z"}
{"id": "msg2", "subject": "Another Subject", "from": "test2@example.com", "to": ["recipient2@example.com"], "snippet": "Another snippet", "body": "SGVsbG8gYWdhaW4=", "date": "2025-01-02T00:00:00Z"}"""
    
    @pytest.fixture
    def sample_message_dict(self):
        """Sample message dictionary for testing."""
        return {
            "id": "test-msg",
            "subject": "Test Subject",
            "from": "sender@example.com",
            "to": ["recipient@example.com"],
            "snippet": "Test snippet",
            "body": "SGVsbG8gd29ybGQ=",  # "Hello world" in base64
            "date": "2025-01-01T00:00:00Z"
        }
    
    def test_load_jsonl_messages_success(self, temp_file_path, sample_jsonl_content):
        """Test successful loading of JSONL messages."""
        with open(temp_file_path, 'w') as f:
            f.write(sample_jsonl_content)
        
        ingestor = InboxIngestor()
        messages = ingestor.load_jsonl_messages(temp_file_path)
        
        assert len(messages) == 2
        assert messages[0]['id'] == 'msg1'
        assert messages[1]['id'] == 'msg2'
    
    def test_load_jsonl_messages_file_not_found(self):
        """Test loading JSONL messages from non-existent file."""
        ingestor = InboxIngestor()
        
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            ingestor.load_jsonl_messages("nonexistent.jsonl")
    
    def test_load_jsonl_messages_invalid_json(self, temp_file_path):
        """Test loading JSONL messages with invalid JSON."""
        with open(temp_file_path, 'w') as f:
            f.write('{"id": "msg1", "subject": "Test"}\n')
            f.write('invalid json line\n')
            f.write('{"id": "msg2", "subject": "Test2"}\n')
        
        ingestor = InboxIngestor()
        
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            ingestor.load_jsonl_messages(temp_file_path)
    
    def test_load_jsonl_messages_empty_lines(self, temp_file_path):
        """Test loading JSONL messages with empty lines."""
        with open(temp_file_path, 'w') as f:
            f.write('{"id": "msg1", "subject": "Test"}\n')
            f.write('\n')  # Empty line
            f.write('   \n')  # Whitespace only
            f.write('{"id": "msg2", "subject": "Test2"}\n')
        
        ingestor = InboxIngestor()
        messages = ingestor.load_jsonl_messages(temp_file_path)
        
        assert len(messages) == 2
        assert messages[0]['id'] == 'msg1'
        assert messages[1]['id'] == 'msg2'
    
    def test_process_raw_message_success(self, sample_message_dict):
        """Test successful processing of raw message."""
        ingestor = InboxIngestor()
        
        with patch('email_categorizer.ingestion.ingestion.process_email_body') as mock_process:
            mock_process.return_value = ("Hello world!", "Hello world!")
            
            message = ingestor.process_raw_message(sample_message_dict)
            
            assert isinstance(message, Message)
            assert message.msg_id == "test-msg"
            assert message.sender == "sender@example.com"
            assert message.recipients == ["recipient@example.com"]
            assert message.subject == "Test Subject"
            assert message.preview_text == "Test snippet"
            assert message.body_text == "Hello world!"
    
    def test_process_raw_message_missing_required_field(self, sample_message_dict):
        """Test processing raw message with missing required field."""
        del sample_message_dict['subject']
        
        ingestor = InboxIngestor()
        
        with pytest.raises(ValueError, match="Missing required field: subject"):
            ingestor.process_raw_message(sample_message_dict)
    
    def test_process_raw_message_body_processing_failure(self, sample_message_dict):
        """Test processing raw message when body processing fails."""
        ingestor = InboxIngestor()
        
        with patch('email_categorizer.ingestion.ingestion.process_email_body') as mock_process:
            mock_process.side_effect = ValueError("Body processing failed")
            
            with pytest.raises(ValueError, match="Failed to process email body for message test-msg"):
                ingestor.process_raw_message(sample_message_dict)
    
    def test_process_raw_messages_batch_success(self, sample_message_dict):
        """Test batch processing of raw messages."""
        raw_messages = [sample_message_dict, sample_message_dict.copy()]
        raw_messages[1]['id'] = 'test-msg-2'
        
        ingestor = InboxIngestor()
        
        with patch('email_categorizer.ingestion.ingestion.process_email_body') as mock_process:
            mock_process.return_value = ("Hello world!", "Hello world!")
            
            processed_messages, failed_count = ingestor._process_raw_messages(raw_messages)
            
            assert len(processed_messages) == 2
            assert failed_count == 0
            assert all(isinstance(msg, Message) for msg in processed_messages)
    
    def test_process_raw_messages_with_failures(self, sample_message_dict):
        """Test batch processing with some failures."""
        raw_messages = [sample_message_dict, sample_message_dict.copy()]
        raw_messages[1]['id'] = 'test-msg-2'
        del raw_messages[1]['subject']  # This will cause a failure
        
        ingestor = InboxIngestor()
        
        with patch('email_categorizer.ingestion.ingestion.process_email_body') as mock_process:
            mock_process.return_value = ("Hello world!", "Hello world!")
            
            processed_messages, failed_count = ingestor._process_raw_messages(raw_messages)
            
            assert len(processed_messages) == 1
            assert failed_count == 1
    
    @patch('email_categorizer.ingestion.ingestion.FaissManager')
    @patch('email_categorizer.ingestion.ingestion.CategoryManager')
    @patch('email_categorizer.ingestion.ingestion.MessageManager')
    def test_init(self, mock_message_manager, mock_category_manager, mock_faiss_manager):
        """Test InboxIngestor initialization."""
        ingestor = InboxIngestor()
        
        # Verify managers are initialized with force_recreate=True
        mock_message_manager.assert_called_once_with(force_recreate=True)
        mock_category_manager.assert_called_once_with(force_recreate=True)
        mock_faiss_manager.assert_called_once_with(force_recreate=True)
    
    @patch('email_categorizer.ingestion.ingestion.FaissManager')
    @patch('email_categorizer.ingestion.ingestion.CategoryManager')
    @patch('email_categorizer.ingestion.ingestion.MessageManager')
    def test_ingest_full_pipeline(self, mock_message_manager, mock_category_manager, mock_faiss_manager, temp_file_path, sample_jsonl_content):
        """Test the complete ingestion pipeline."""
        # Setup mocks
        mock_msg_manager = MagicMock()
        mock_faiss_manager_instance = MagicMock()
        mock_message_manager.return_value = mock_msg_manager
        mock_faiss_manager.return_value = mock_faiss_manager_instance
        
        # Create test file
        with open(temp_file_path, 'w') as f:
            f.write(sample_jsonl_content)
        
        ingestor = InboxIngestor()
        
        with patch.object(ingestor, '_process_raw_messages') as mock_process:
            mock_messages = [
                Message("msg1", "test@example.com", ["recipient@example.com"], "2025-01-01T00:00:00Z", "Test Subject", "Test snippet", "Hello world!"),
                Message("msg2", "test2@example.com", ["recipient2@example.com"], "2025-01-02T00:00:00Z", "Another Subject", "Another snippet", "Hello again")
            ]
            mock_process.return_value = (mock_messages, 0)
            
            ingestor.ingest(temp_file_path)
            
            # Verify the pipeline steps were called
            mock_process.assert_called_once()
            mock_msg_manager.save_messages.assert_called_once_with(mock_messages)
            mock_faiss_manager_instance.add_messages_to_index.assert_called_once_with(mock_messages)
            mock_faiss_manager_instance._save_index.assert_called_once()
    
    @patch('email_categorizer.ingestion.ingestion.FaissManager')
    @patch('email_categorizer.ingestion.ingestion.CategoryManager')
    @patch('email_categorizer.ingestion.ingestion.MessageManager')
    def test_ingest_with_failures(self, mock_message_manager, mock_category_manager, mock_faiss_manager, temp_file_path, sample_jsonl_content):
        """Test ingestion pipeline with processing failures."""
        # Setup mocks
        mock_msg_manager = MagicMock()
        mock_faiss_manager_instance = MagicMock()
        mock_message_manager.return_value = mock_msg_manager
        mock_faiss_manager.return_value = mock_faiss_manager_instance
        
        # Create test file
        with open(temp_file_path, 'w') as f:
            f.write(sample_jsonl_content)
        
        ingestor = InboxIngestor()
        
        with patch.object(ingestor, '_process_raw_messages') as mock_process:
            mock_messages = [Message("msg1", "test@example.com", ["recipient@example.com"], "2025-01-01T00:00:00Z", "Test Subject", "Test snippet", "Hello world!")]
            mock_process.return_value = (mock_messages, 1)  # 1 failure
            
            ingestor.ingest(temp_file_path)
            
            # Verify the pipeline still completed despite failures
            mock_msg_manager.save_messages.assert_called_once_with(mock_messages)
            mock_faiss_manager_instance.add_messages_to_index.assert_called_once_with(mock_messages)
    
    def test_ingest_file_not_found(self):
        """Test ingestion with non-existent file."""
        ingestor = InboxIngestor()
        
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            ingestor.ingest("nonexistent.jsonl")


class TestIngestionIntegration:
    """Integration tests for the ingestion pipeline."""
    
    @pytest.fixture
    def temp_file_path(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_real_base64_decoding(self, temp_file_path):
        """Test with real base64 encoded email body."""
        # Real base64 encoded HTML email body
        real_body_b64 = "SGksIHRoYW5rcyBmb3IgZmx5aW5nIHdpdGggRGVsdGEuIFJlY2VpcHQgZm9yIGNvbmZpcm1hdGlvbiA0MTYzMTE5Nzg1LiBGYXJlICQgMjk4LiBUaGlzIGlzIHlvdXIgb2ZmaWNpYWwgcmVjZWlwdC4="
        
        sample_message = {
            "id": "real-msg",
            "subject": "Test Subject",
            "from": "test@example.com",
            "to": ["recipient@example.com"],
            "snippet": "Test snippet",
            "body": real_body_b64,
            "date": "2025-01-01T00:00:00Z"
        }
        
        with open(temp_file_path, 'w') as f:
            json.dump(sample_message, f)
            f.write('\n')
        
        ingestor = InboxIngestor()
        messages = ingestor.load_jsonl_messages(temp_file_path)
        processed_message = ingestor.process_raw_message(messages[0])
        
        assert isinstance(processed_message, Message)
        assert "thanks for flying" in processed_message.body_text.lower()
        assert "delta" in processed_message.body_text.lower()
    
    def test_complex_html_processing(self):
        """Test processing of complex HTML email."""
        complex_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Email</title></head>
        <body>
            <h1>Welcome!</h1>
            <p>Check out our <a href="https://example.com">website</a> for more info.</p>
            <ul>
                <li>Feature 1</li>
                <li>Feature 2</li>
            </ul>
            <img src="logo.png" alt="Company Logo">
            <script>console.log('hidden');</script>
        </body>
        </html>
        """
        
        encoded = base64.b64encode(complex_html.encode('utf-8')).decode('utf-8')
        full_text, truncated = process_email_body(encoded)
        
        assert "Welcome!" in full_text
        assert "website (https://example.com)" in full_text
        assert "• Feature 1" in full_text
        assert "• Feature 2" in full_text
        assert "[Image: Company Logo]" in full_text
        assert "console.log" not in full_text  # Script should be removed
        assert "<" not in full_text  # No HTML tags should remain
    
    def test_edge_cases_robustness(self, temp_file_path):
        """Test robustness with various edge cases."""
        edge_cases = [
            # Empty message
            {"id": "empty", "subject": "", "from": "", "to": [], "snippet": "", "body": "", "date": "2025-01-01T00:00:00Z"},
            # Very long subject
            {"id": "long", "subject": "A" * 1000, "from": "test@example.com", "to": ["recipient@example.com"], "snippet": "Test", "body": "SGVsbG8=", "date": "2025-01-01T00:00:00Z"},
            # Special characters
            {"id": "special", "subject": "Test with émojis 🎉 and ñ", "from": "test@example.com", "to": ["recipient@example.com"], "snippet": "Test", "body": "SGVsbG8gd29ybGQ=", "date": "2025-01-01T00:00:00Z"}
        ]
        
        with open(temp_file_path, 'w') as f:
            for case in edge_cases:
                json.dump(case, f)
                f.write('\n')
        
        ingestor = InboxIngestor()
        messages = ingestor.load_jsonl_messages(temp_file_path)
        
        assert len(messages) == 3
        
        # Test processing each message
        for msg_dict in messages:
            processed = ingestor.process_raw_message(msg_dict)
            assert isinstance(processed, Message)
            assert processed.msg_id is not None


if __name__ == "__main__":
    pytest.main([__file__])
