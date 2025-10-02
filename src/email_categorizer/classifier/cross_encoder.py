from typing import List, Tuple, Optional, Dict
from sentence_transformers import CrossEncoder as SentenceTransformerCrossEncoder
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import settings

from ..types import Message, Category
from ..utils.utils import get_compact_message_representation


class CrossEncoder:
    """Cross-encoder for scoring email-category relevance."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder classifier.
        
        Args:
            model_name: Cross-encoder model name (defaults to config)
        """
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        # Only load model if cross-encoder is enabled
        if settings.CROSS_ENCODER_ENABLED:
            print(f"Loading cross-encoder model: {self.model_name}")
            self.model = SentenceTransformerCrossEncoder(self.model_name)
            print(f"Cross-encoder model {self.model_name} loaded")
        else:
            self.model = None
    
    
    def score_candidates(
        self,
        messages: List[Message],
        category: Category
    ) -> Dict[str, float]:
        """
        Score email-category pairs using cross-encoder.
        
        Args:
            messages: List of messages to score
            category: Category to score against
            
        Returns:
            Dictionary mapping message ID to score
        """
        if self.model is None:
            raise ValueError("Cross-encoder is not enabled")
        
        if not messages:
            return {}
        
        # Prepare query text (category representation)
        query_text = self._get_category_text_representation(category)
        
        # Prepare document texts (email representations)
        doc_texts = []
        msg_ids = []
        for msg in messages:
            doc_text = get_compact_message_representation(msg)
            doc_texts.append(doc_text)
            msg_ids.append(msg.msg_id)
        
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
        
        # Create dictionary mapping message ID to score
        return dict(zip(msg_ids, scores))
    
    def _get_category_text_representation(self, category: Category) -> str:
        """
        Get structured text representation of category for cross-encoder.
        
        Args:
            category: Category object
            
        Returns:
            Structured text representation with category name and description
        """
        lines = [
            f"Category name: {category.name or ''}",
            f"Category description: {category.description or ''}"
        ]
        
        return "\n".join(lines)