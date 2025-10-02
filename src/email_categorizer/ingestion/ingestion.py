"""
Email ingestion pipeline.

Loads JSONL messages, processes text content, and saves to SQLite database.
"""

import json
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from ..types import Message
from .process_email_body import process_email_body
from ..data_managers import MessageManager, CategoryManager, FaissManager

# Load environment variables from .env file
load_dotenv()


class InboxIngestor:
    """Handles ingestion of email messages from JSONL files."""
    
    def __init__(self):
        """
        Initialize the ingestor.

        Initialize the managers with force_recreate=True to force recreation of the data structures for fresh ingestion.
        """
        self.message_manager = MessageManager(force_recreate=True)
        self.category_manager = CategoryManager(force_recreate=True)
        self.faiss_manager = FaissManager(force_recreate=True)
    
    def ingest(self, input_file: str) -> None:
        """
        Run the ingestion pipeline.

        1. Load messages from input file
        2. Process the messages (clean the text, conver the HTML body to text)
        3. Invoke the MessageManager to save the messages to the database
        4. Invoke the FaissManager to build an embedding index
        
        Args:
            input_file: Path to input JSONL file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            Exception: If processing fails
        """
        print(f"\nLoading messages from {input_file}...")
        raw_messages = self.load_jsonl_messages(input_file)
        print(f"Loaded {len(raw_messages)} messages from {input_file}")
        
        print("\nProcessing messages...")
        processed_messages, failed_count = self._process_raw_messages(raw_messages)
        print(f"Successfully processed {len(processed_messages)} messages")
        if failed_count > 0:
            print(f"Failed to process {failed_count} messages")
        
        print(f"\nSaving messages to database...")
        self.message_manager.save_messages(processed_messages)
        
        print(f"\nGenerating embeddings and adding to FAISS index...")
        self.faiss_manager.add_messages_to_index(processed_messages)
        
        # Save the FAISS index
        self.faiss_manager._save_index()
        
        print(f"\n✅ Ingestion complete!")

    def _process_raw_messages(self, raw_messages: List[Dict[str, Any]]) -> List[Message]:
        """
        Process a list of raw messages into a list of Message objects.
        
        Args:
            raw_messages: List of raw message dictionaries
        """
        processed_messages = []
        failed_count = 0
        
        for raw_msg in raw_messages:
            try:
                processed_msg = self.process_raw_message(raw_msg)
                processed_messages.append(processed_msg)
            except Exception as e:
                print(f"Warning: Failed to process message {raw_msg.get('id', 'unknown')}: {e}")
                failed_count += 1
                continue

        return processed_messages, failed_count
    
    def process_raw_message(self, raw_msg: Dict[str, Any]) -> Message:
        """
        Convert raw JSONL message to processed Message object.
        
        Args:
            raw_msg: Raw message dictionary from JSONL
            
        Returns:
            Processed Message object
            
        Raises:
            ValueError: If required fields are missing or processing fails
        """
        # Validate required fields
        required_fields = ['id', 'subject', 'from', 'to', 'snippet', 'body', 'date']
        for field in required_fields:
            if field not in raw_msg:
                raise ValueError(f"Missing required field: {field}")
        
        # Process the email body
        try:
            cleaned_text, truncated_text = process_email_body(raw_msg['body'])
        except Exception as e:
            raise ValueError(f"Failed to process email body for message {raw_msg['id']}: {e}")
        
        # Create Message object (truncation happens on-demand in classifier)
        return Message(
            msg_id=raw_msg['id'],
            sender=raw_msg['from'],
            recipients=raw_msg['to'],
            date=raw_msg['date'],
            subject=raw_msg['subject'],
            preview_text=raw_msg['snippet'],
            body_text=cleaned_text
        )
    
    
    def load_jsonl_messages(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load raw messages from a JSONL file.
        
        Args:
            input_file: Path to input JSONL file
            
        Returns:
            List of raw message dictionaries
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        messages = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        message = json.loads(line)
                        messages.append(message)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read input file {input_file}: {e}")
        
        return messages


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m email_categorizer.ingestion <input_jsonl_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        ingestor = InboxIngestor()
        ingestor.ingest(input_file)
    except Exception as e:
        print(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)