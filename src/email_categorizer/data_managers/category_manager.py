"""
Category Manager - Owns all category operations.

Responsibilities:
- Database initialization (creates categories + message_categories tables)
- Category CRUD (create, get, list, delete)
- Classification results storage and retrieval
- Slug generation and validation
- Keyword generation for efficient retrieval
"""

import sqlite3
import re
import json
import yaml
from pathlib import Path
from typing import List, Optional, Dict
from ..types import Category, ClassificationResult
from ..llm_client.llm_client import LLMClient
from ..utils.utils import extract_json_from_response
from .sqlite_utils import (
    DEFAULT_DB_PATH,
    ensure_db_directory,
    get_connection,
    db_connection
)


class CategoryManager:
    """Manages all category operations: CRUD and classifications."""

    PROMPTS_PATH = Path(__file__).parent.parent / "llm_client" / "prompts.yaml"
    KEYWORD_PROMPT_KEY = "keywords_prompt"
    MAX_KEYWORDS = 10
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, force_recreate: bool = False):
        """
        Initialize category manager.
        
        Args:
            db_path: Path to SQLite database file
            force_recreate: If True, drop and recreate all tables (for --force flag)
        """
        self.db_path = db_path
        self._ensure_database_exists(force_recreate=force_recreate)
        self.llm_client = LLMClient()
        
        try:
            with open(self.PROMPTS_PATH, 'r') as f:
                templates = yaml.safe_load(f)
                self.keywords_prompt = templates[self.KEYWORD_PROMPT_KEY]
        except Exception as e:
            raise ValueError(f"Error loading prompt templates from {self.PROMPTS_PATH}: {e}")
    
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
        Drop all category-related tables.
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Drop tables in reverse order of dependencies
        cursor.execute("DROP TABLE IF EXISTS message_categories")
        cursor.execute("DROP TABLE IF EXISTS categories")
        
        conn.commit()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """
        Create categories and message_categories tables.
        
        Creates:
        - categories table (stores user-defined categories)
        - message_categories table (stores classification results)
        - Indices for fast queries
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='categories'
        """)
        tables_exist = cursor.fetchone() is not None
        
        # Create categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create message_categories junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_categories (
                msg_id TEXT NOT NULL,
                category_slug TEXT NOT NULL,
                is_in_category BOOLEAN NOT NULL,
                explanation TEXT,
                classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (msg_id, category_slug),
                FOREIGN KEY (category_slug) REFERENCES categories(slug) ON DELETE CASCADE
            )
        """)
        
        # Create indices for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_categories_msg_id
            ON message_categories(msg_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_categories_category_result
            ON message_categories(category_slug, is_in_category)
        """)
        
        conn.commit()
        
        if not tables_exist:
            print("Created empty category tables")
        else:
            print("Using preexisting category tables")
        
    def _generate_keywords(self, name: str, description: str) -> Optional[List[str]]:
        """
        Generate focused keywords for a category using an LLM.
        
        Args:
            name: Category name
            description: Category description
            
        Returns:
            List of specific keywords for retrieval
        """
        try:
            prompt = self.keywords_prompt.format(
                category_name=name,
                category_description=description
            )
            response = self.llm_client.chat_completion(prompt)
            
            json_response = extract_json_from_response(response)
            keywords = json_response["keywords"]
            
            return keywords[:self.MAX_KEYWORDS]
        except Exception as e:
            print(f"Warning: Failed to generate keywords: {e}")
            return None
    
    def create_category(self, name: str, description: str) -> Category:
        """
        Create a new category with auto-generated keywords.
        
        Args:
            name: Category name (user-provided)
            description: Natural language description
            
        Returns:
            Created Category object with keywords
            
        Raises:
            ValueError: If category with same slug already exists
        """
        slug = self.generate_slug(name)
        
        # Keyword generation commented out as it's not being used in retrieval
        # print(f"Generating keywords for category '{name}'...")
        # keywords = self._generate_keywords(name, description)
        # if keywords:
        #     print(f"   Keywords: {', '.join(keywords)}")
        # else:
        #     print("   Keywords: None")
        keywords = None
        
        keywords_json = json.dumps(keywords) if keywords else None
        
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if category already exists
            cursor.execute("SELECT slug FROM categories WHERE slug = ?", (slug,))
            if cursor.fetchone():
                raise ValueError(f"Category with slug '{slug}' already exists")
            
            # Insert new category
            cursor.execute("""
                INSERT INTO categories (slug, name, description, keywords)
                VALUES (?, ?, ?, ?)
            """, (slug, name, description, keywords_json))
            
            conn.commit()
            return Category(name=name, description=description, slug=slug, keywords=keywords)
    
    def get_category(self, slug: str) -> Optional[Category]:
        """
        Retrieve a category by slug.
        
        Args:
            slug: Category slug identifier
            
        Returns:
            Category object if found, None otherwise
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories WHERE slug = ?", (slug,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            keywords = json.loads(row['keywords']) if row['keywords'] else None
            return Category(
                name=row['name'],
                description=row['description'],
                slug=row['slug'],
                keywords=keywords
            )
    
    def get_all_categories(self) -> List[Category]:
        """
        Get all categories.
        
        Returns:
            List of all Category objects, ordered by name
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories ORDER BY name")
            rows = cursor.fetchall()
            
            return [
                Category(
                    name=row['name'],
                    description=row['description'],
                    slug=row['slug'],
                    keywords=json.loads(row['keywords']) if row['keywords'] else None
                )
                for row in rows
            ]
    
    def delete_category(self, slug: str) -> bool:
        """
        Delete a category.
        
        Args:
            slug: Category slug identifier
            
        Returns:
            True if deleted, False if not found
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM categories WHERE slug = ?", (slug,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_all_categories(self) -> int:
        """
        Delete all categories.
        
        Returns:
            Number of categories deleted
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM categories")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM categories")
            conn.commit()
            return count
    
    
    def save_classification_results(
        self,
        slug: str,
        results: Dict[str, Optional[ClassificationResult]]
    ) -> None:
        """
        Save classification results for a category.
        
        Args:
            slug: Category slug identifier
            results: Dict mapping msg_id to ClassificationResult
        """
        if not results:
            return
        
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Filter out None results
            data = [
                (msg_id, slug, result.is_in_category, result.explanation)
                for msg_id, result in results.items()
                if result is not None
            ]
            
            # Batch insert
            if data:
                cursor.executemany("""
                    INSERT OR REPLACE INTO message_categories
                    (msg_id, category_slug, is_in_category, explanation)
                    VALUES (?, ?, ?, ?)
                """, data)
            
            conn.commit()
    
    def get_category_message_ids(self, slug: str) -> List[str]:
        """Get the IDs of the messages in the category slug."""
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT msg_id 
                FROM message_categories
                WHERE category_slug = ? AND is_in_category = TRUE
                ORDER BY classified_at DESC
            """, (slug,))
            
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    
    
    def generate_slug(self, name: str) -> str:
        """
        Generate a URL/filesystem-safe slug from category name.
        
        Args:
            name: Category name
            
        Returns:
            Generated slug (lowercase, hyphens, alphanumeric)
        """
        slug = name.lower()
        slug = slug.replace(' ', '-').replace('_', '-')
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        slug = re.sub(r'-+', '-', slug)
        slug = slug.strip('-')
        return slug