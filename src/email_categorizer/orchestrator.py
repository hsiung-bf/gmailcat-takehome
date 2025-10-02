"""
Orchestrator for email categorization.

Coordinates between ingestion, classification, and storage.
"""

from typing import Dict, List, Optional
from dataclasses import asdict
from .types import Message, Category, ClassificationResult
from .data_managers.message_manager import MessageManager
from .data_managers.category_manager import CategoryManager
from .data_managers.faiss_manager import FaissManager
from .classifier.classification_service import ClassificationService
import os
import json


class EmailCategorizationOrchestrator:
    """Orchestrates email categorization workflow."""
    
    def __init__(self, test_mode: bool = False, output_dir: str = "data/outputs"):
        """
        Initialize orchestrator.
        
        Args:
            test_mode: If True, limit to 5 emails for testing
            output_dir: Directory to save classification results
        """
        # Initialize managers
        self.message_manager = MessageManager()
        self.category_manager = CategoryManager()
        self.faiss_manager = FaissManager()
        self.classification_service = ClassificationService(
            message_manager=self.message_manager,
            faiss_manager=self.faiss_manager
        )

        self.test_mode = test_mode
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_messages(self) -> List[Message]:
        """Load processed messages from database (limited to 5 in test mode)."""
        messages = self.message_manager.get_all_messages()
        
        if self.test_mode:
            return messages[:5]
        
        return messages
    
    def has_messages(self) -> bool:
        """Check if ingestion has been run."""
        return self.message_manager.has_messages()
    
    def classify_emails(
        self, 
        messages: List[Message], 
        category: Category
    ) -> Dict[str, Optional[ClassificationResult]]:
        """
        Classify emails for a category.
        
        Args:
            messages: List of messages to classify
            category: Category to classify against
            
        Returns:
            Dictionary mapping msg_id to ClassificationResult (or None if failed)
        """
        results: Dict[str, Optional[ClassificationResult]] = self.classification_service.classify_emails(messages, category)
        self.category_manager.save_classification_results(category.slug, results)
        classification_summary = self.get_classification_summary(results)
        return results, classification_summary
    
    def get_classification_summary(
        self, 
        results: Dict[str, Optional[ClassificationResult]]
    ) -> Dict[str, int]:
        """Get summary statistics for classification results."""
        total = len(results.keys())

        in_category = 0
        not_in_category = 0
        classification_failure = 0
        for result in results.values():
            if result is None:
                classification_failure += 1
            elif result.is_in_category:
                in_category += 1
            else:
                not_in_category += 1
        
        return {
            "total": total,
            "in_category": in_category,
            "not_in_category": not_in_category,
            "classification_failure": classification_failure
        }
        
    def create_category(self, name: str, description: str) -> Category:
        """Create a new category."""
        return self.category_manager.create_category(name, description)
    
    def get_all_categories(self) -> List[Category]:
        """Get all categories."""
        return self.category_manager.get_all_categories()
    
    def delete_category(self, slug: str) -> bool:
        """
        Delete a category.
            
        Returns:
            True if deleted, False if not found
        """
        return self.category_manager.delete_category(slug)
    
    def delete_all_categories(self) -> int:
        """Delete all categories. Returns count of deleted categories."""
        return self.category_manager.delete_all_categories()
    
    def get_emails_in_category(self, slug: str) -> List[str]:
        """
        Get formatted email previews for emails in a category.
        
        Args:
            slug: Category slug
            
        Returns:
            List of formatted email preview strings
        """
        from .utils.utils import format_email_preview
        
        # Get message IDs classified as in the category
        msg_ids = self.category_manager.get_category_message_ids(slug)
        
        if not msg_ids:
            return []
        
        # Fetch the actual messages
        messages_dict = self.message_manager.get_messages_by_ids(msg_ids)
        
        # Format each message as a preview
        return [format_email_preview(msg) for msg in messages_dict.values()]