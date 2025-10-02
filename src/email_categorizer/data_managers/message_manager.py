"""
Message Manager - Owns all message operations.

Responsibilities:
- Database initialization (creates messages table + FTS5 index)
- Message CRUD (save, get, update, delete)
- Keyword search (FTS5 BM25)
"""

import sqlite3
import json
from typing import List, Optional, Tuple, Dict
from ..types import Message
from .sqlite_utils import (
    DEFAULT_DB_PATH,
    ensure_db_directory,
    get_connection,
    db_connection
)


class MessageManager:
    """Manages all message operations: storage and search."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, force_recreate: bool = False):
        """
        Initialize message manager.
        
        Args:
            db_path: Path to SQLite database file
            force_recreate: If True, drop and recreate all tables (for --force flag)
        """
        self.db_path = db_path
        self._ensure_database_exists(force_recreate=force_recreate)
    
    def _ensure_database_exists(self, force_recreate: bool = False):
        """
        Create database and tables if they don't exist.
        
        Args:
            force_recreate: If True, drop existing tables first
        """
        ensure_db_directory(self.db_path)
        with db_connection(self.db_path) as conn:
            if force_recreate:
                self._drop_tables(conn)
            
            self._create_tables(conn)
    
    def _drop_tables(self, conn: sqlite3.Connection):
        """
        Drop all message-related tables.
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Drop tables in reverse order of dependencies
        cursor.execute("DROP TABLE IF EXISTS messages_fts")
        cursor.execute("DROP TABLE IF EXISTS messages")
        
        conn.commit()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """
        Create messages table and FTS5 index.
        
        Creates:
        - messages table (stores email data)
        - messages_fts virtual table (FTS5 full-text search index)
        - Triggers to keep FTS5 in sync
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='messages'
        """)
        tables_exist = cursor.fetchone() is not None
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                recipients TEXT NOT NULL,
                date TEXT NOT NULL,
                subject TEXT NOT NULL,
                preview_text TEXT,
                body_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                msg_id UNINDEXED,
                subject,
                body_text,
                sender,
                content=messages,
                tokenize='porter'
            )
        """) # The porter tokenizer stems the words before indexing and querying
        
        # Create triggers to keep FTS5 in sync with messages table
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_insert 
            AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, msg_id, subject, body_text, sender)
                VALUES (new.rowid, new.msg_id, new.subject, new.body_text, new.sender);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_delete 
            AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.rowid;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_update 
            AFTER UPDATE ON messages BEGIN
                UPDATE messages_fts SET
                    subject = new.subject,
                    body_text = new.body_text,
                    sender = new.sender
                WHERE rowid = new.rowid;
            END
        """)
        
        conn.commit()
        
        if not tables_exist:
            print("Created empty messages tables and FTS5 index")
        else:
            print("Using preexisting messages tables and FTS5 index")
        
    def has_messages(self) -> bool:
        """
        Check if messages have been ingested.
        
        Returns:
            True if messages exist in database, False otherwise
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            return count > 0
    
    def save_messages(self, messages: List[Message]) -> None:
        """Batch save messages to database."""
        if not messages:
            return
        
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            data = [
                (
                    msg.msg_id,
                    msg.sender,
                    json.dumps(msg.recipients),
                    msg.date,
                    msg.subject,
                    msg.preview_text,
                    msg.body_text
                )
                for msg in messages
            ]

            cursor.executemany("""
                INSERT OR REPLACE INTO messages 
                (msg_id, sender, recipients, date, subject, preview_text, body_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            conn.commit()
            print(f"Saved {len(messages)} messages to database")

    def get_messages_by_ids(self, msg_ids: List[str]) -> Dict[str, Message]:
        """
        Retrieve multiple messages by IDs.
        
        Args:
            msg_ids: List of message identifiers
            
        Returns:
            Dictionary mapping message ID to Message object
        """
        if not msg_ids:
            return {}
        
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(msg_ids))
            query = f"SELECT * FROM messages WHERE msg_id IN ({placeholders})"
            cursor.execute(query, msg_ids)
            rows = cursor.fetchall()
            
            return {row[0]: self._row_to_message(row) for row in rows}
    
    def get_all_messages(self) -> List[Message]:
        """
        Load all messages from database.
        
        Returns:
            List of all Message objects, ordered by date DESC
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY date DESC")
            rows = cursor.fetchall()
            
            return [self._row_to_message(row) for row in rows]
        
    def search_by_keywords(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Tuple[Message, float]]:
        """
        Search messages using FTS5 keyword search.
        
        Args:
            query: Search query (e.g., category name + description)
            limit: Maximum number of results
            
        Returns:
            List of (Message, bm25_score) tuples, sorted by relevance
            Higher score = better match
            IMPORTANT NOTE: messages without any keyword matches will not be returned 
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query FTS5 with JOIN to get full message data in one query
            # Note: In FTS5, more negative BM25 = better match, so ORDER BY ASC
            cursor.execute("""
                SELECT 
                    m.*,
                    bm25(messages_fts) as score
                FROM messages_fts
                JOIN messages m ON messages_fts.msg_id = m.msg_id
                WHERE messages_fts MATCH ?
                ORDER BY bm25(messages_fts) ASC
                LIMIT ?
            """, (query, limit))
            
            results = cursor.fetchall()
            
            # Convert to (Message, score) tuples
            return [
                (self._row_to_message(row), row['score'])
                for row in results
            ]
        
    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """
        Convert database row to Message object.
        
        Args:
            row: SQLite Row object
            
        Returns:
            Message object
        """
        return Message(
            msg_id=row['msg_id'],
            sender=row['sender'],
            recipients=json.loads(row['recipients']),
            date=row['date'],
            subject=row['subject'],
            preview_text=row['preview_text'],
            body_text=row['body_text']
        )

