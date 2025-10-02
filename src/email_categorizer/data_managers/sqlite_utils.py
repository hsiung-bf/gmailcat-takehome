"""
SQLite utilities for database managers.

Provides common database operations and connection management.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator


# Default database path used by all managers
DEFAULT_DB_PATH = "data/emails.db"


def ensure_db_directory(db_path: str) -> None:
    """
    Ensure the database directory exists.
    
    Args:
        db_path: Path to database file
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a database connection with row_factory configured.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection with Row factory and foreign keys enabled
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    return conn


@contextmanager
def db_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Automatically handles connection cleanup.
    
    Args:
        db_path: Path to SQLite database file
        
    Yields:
        SQLite connection
        
    Example:
        with db_connection("data/emails.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages")
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def execute_script(db_path: str, sql_script: str) -> None:
    """
    Execute a SQL script (multiple statements).
    
    Args:
        db_path: Path to SQLite database file
        sql_script: SQL script to execute
    """
    with db_connection(db_path) as conn:
        conn.executescript(sql_script)
        conn.commit()


def table_exists(db_path: str, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of table to check
        
    Returns:
        True if table exists, False otherwise
    """
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None


def drop_table(db_path: str, table_name: str) -> None:
    """
    Drop a table if it exists.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of table to drop
    """
    with db_connection(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()

