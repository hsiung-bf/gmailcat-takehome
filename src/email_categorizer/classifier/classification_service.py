"""
Classification Service - Orchestrates email classification.

Implements configurable multi-tier classification:
1. Candidate generation using BM25 + semantic search
2. Cross-encoder scoring (if enabled)
3. LLM validation for grey area candidates only

High confidence candidates are auto-classified without LLM.
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Set

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import settings

from ..types import Message, Category, ClassificationResult
from .llm_classifier import LLMClassifier
from .cross_encoder import CrossEncoder
from .threshold_partitioner.fixed_threshold_partitioner import FixedThresholdPartitioner 
from ..data_managers.message_manager import MessageManager
from ..data_managers.faiss_manager import FaissManager
from ..utils.utils import get_compact_message_representation


class ClassificationService:
    """Service for classifying emails into categories with intelligent candidate selection."""
    
    def __init__(self, message_manager: Optional[MessageManager] = None, faiss_manager: Optional[FaissManager] = None):
        """
        Initialize the classification service.
        
        Args:
            message_manager: MessageManager instance (creates new one if None)
            faiss_manager: FaissManager instance (creates new one if None)
        """
        self.llm_classifier = LLMClassifier()
        self.cross_encoder = CrossEncoder()
        self.message_manager = message_manager or MessageManager()
        self.faiss_manager = faiss_manager or FaissManager()
        
        self.threshold_partitioner = FixedThresholdPartitioner()
    
    def classify_emails(
        self,
        messages: List[Message],
        category: Category
    ) -> Dict[str, Optional[ClassificationResult]]:
        """
        Classify emails using configurable multi-tier approach:
        1. Generate candidates using BM25 + semantic search
        2. Score candidates with cross-encoder (if enabled)
        3. Auto-classify high/low confidence, LLM for grey area
        
        Args:
            messages: List of messages to classify (typically all messages)
            category: Category to classify against
            
        Returns:
            Dictionary mapping message_id to ClassificationResult
        """        
        # Step 1: Generate candidates using semantic search
        print('\nGenerating candidates...')
        candidates, initial_scores = self._generate_candidates(messages, category)
        print(f'Generated {len(candidates)} candidates')
        
        if not candidates:
            # No candidates found - classify all as FALSE
            return self._classify_all_false(messages, category, "No candidates found")
        
        # Step 2: Score candidates with cross-encoder (if not enabled, use the initial scores)
        if settings.CROSS_ENCODER_ENABLED:
            print("Scoring candidates with cross-encoder...")
            candidate_scores = self.cross_encoder.score_candidates(candidates, category)
            score_type = "cross encoder"
        else:
            print("Cross-encoder disabled, using semantic scores for partitioning...")
            candidate_scores = initial_scores
            score_type = "semantic"

        
        # Step 3: Apply thresholding to partition candidates
        high_confidence_ids, grey_area_ids, low_confidence_ids = self.threshold_partitioner.partition_candidates(
            candidates, candidate_scores
        )
        
        print(f'Partitioned results: {len(high_confidence_ids)} high confidence, {len(grey_area_ids)} grey area, {len(low_confidence_ids)} low confidence')
        
        # Step 4: Build results dict
        results: Dict[str, Optional[ClassificationResult]] = {}
        
        # Auto-classify high confidence as TRUE
        for msg_id in high_confidence_ids:
            score = candidate_scores.get(msg_id, 0.0)
            results[msg_id] = ClassificationResult(
                msg_id=msg_id,
                category_slug=category.slug,
                is_in_category=True,
                explanation=f"High confidence match ({score_type} score: {score:.3f})"
            )
        
        # Auto-classify low confidence as FALSE
        for msg_id in low_confidence_ids:
            score = candidate_scores.get(msg_id, 0.0)
            results[msg_id] = ClassificationResult(
                msg_id=msg_id,
                category_slug=category.slug,
                is_in_category=False,
                explanation=f"Low confidence ({score_type} score: {score:.3f})"
            )
        
        # Step 5: Use LLM to classify grey area candidates
        if grey_area_ids:
            print(f"Sending {len(grey_area_ids)} grey area candidates to LLM...")
            grey_area_messages = [msg for msg in candidates if msg.msg_id in grey_area_ids]
            llm_results = self.llm_classifier.classify_emails(grey_area_messages, category)
            results.update(llm_results)
        
        # Step 6: Handle messages not in candidates
        candidate_ids = {msg.msg_id for msg in candidates}
        for message in messages:
            if message.msg_id not in results and message.msg_id not in candidate_ids:
                results[message.msg_id] = ClassificationResult(
                    msg_id=message.msg_id,
                    category_slug=category.slug,
                    is_in_category=False,
                    explanation="Not in candidate set"
                )
        
        print(f'Finished classifying {len(results)} messages')
        return results
    
    def _generate_candidates(
        self,
        messages: List[Message],
        category: Category
    ) -> Tuple[List[Message], Dict[str, float]]:
        """
        Generate candidates using semantic search only.
        
        Args:
            messages: List of messages to classify
            category: Category to search for
            
        Returns:
            Tuple of (candidate_messages, candidate_scores)
        """
        # TODO: extend to include keyword retrieval
        input_msg_ids = {msg.msg_id for msg in messages}
        
        # Semantic search
        semantic_candidates = self._get_semantic_candidates(category, input_msg_ids)
        candidate_scores = dict(semantic_candidates)
        print(f'Semantic candidates: {len(semantic_candidates)}')
        
        # Convert back to Message objects
        candidate_messages_dict = self.message_manager.get_messages_by_ids(list(candidate_scores.keys()))
        candidate_messages = list(candidate_messages_dict.values())
        return candidate_messages, candidate_scores
    
    def _get_semantic_candidates(
        self,
        category: Category,
        input_msg_ids: Set[str]
    ) -> List[Tuple[str, float]]:
        """Get candidates from semantic search with scores, filtered to input messages."""
        # Get semantic search results
        semantic_results = self.faiss_manager.search_similar(
            category,
            limit=settings.CANDIDATE_LIMIT_SEMANTIC
        )
        
        # Filter to only the messages we're classifying
        return [
            (msg_id, score) for msg_id, score in semantic_results
            if msg_id in input_msg_ids
        ]
    
    def _classify_all_false(
        self,
        messages: List[Message],
        category: Category,
        explanation: str
    ) -> Dict[str, Optional[ClassificationResult]]:
        """Classify all messages as FALSE with given explanation."""
        results = {}
        for message in messages:
            results[message.msg_id] = ClassificationResult(
                msg_id=message.msg_id,
                category_slug=category.slug,
                is_in_category=False,
                explanation=explanation
            )
        return results
    