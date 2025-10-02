from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import settings

from ..types import Message, Category
from ..utils.utils import get_message_text_representation


class CrossEncoder:
    """Cross-encoder for scoring email-category relevance."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder classifier.
        
        Args:
            model_name: Cross-encoder model name (defaults to config)
        """
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        print(f"Loading {self.model_name} cross-encoder model")
        try:
            self.model = CrossEncoder(self.model_name)
            print(f"Cross-encoder loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load cross-encoder: {e}")
    
    def score_candidates(
        self,
        messages: List[Message],
        category: Category
    ) -> List[Tuple[Message, float]]:
        """
        Score email-category pairs using cross-encoder.
        
        Args:
            messages: List of messages to score
            category: Category to score against
            
        Returns:
            List of (Message, score) tuples, sorted by score (highest first)
        """
        if not self.is_enabled():
            raise ValueError("Cross-encoder is not enabled")
        
        if not messages:
            return []
        
        # Prepare query text (category representation)
        query_text = self._get_category_text_representation(category)
        
        # Prepare document texts (email representations)
        doc_texts = []
        for msg in messages:
            doc_text = get_message_text_representation(msg)
            doc_texts.append(doc_text)
        
        # Create query-document pairs
        pairs = [(query_text, doc_text) for doc_text in doc_texts]
        
        # Get scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]
            else:
                scores = [float(score) for score in scores]
        except Exception as e:
            raise ValueError(f"Cross-encoder scoring failed: {e}")
        
        # Combine messages with scores and sort by score (highest first)
        results = list(zip(messages, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _get_category_text_representation(self, category: Category) -> str:
        """
        Get text representation of category for cross-encoder.
        
        Args:
            category: Category object
            
        Returns:
            Text representation combining name, description, and keywords
        """
        parts = [category.name]
        
        if category.description:
            parts.append(category.description)
        
        if category.keywords:
            # Add keywords as additional context
            keywords_text = " ".join(category.keywords)
            parts.append(keywords_text)
        
        return " ".join(parts).strip()
