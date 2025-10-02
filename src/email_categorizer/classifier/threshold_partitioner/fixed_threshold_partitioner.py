"""
Fixed threshold partitioner using sigmoid normalization and percentiles.
"""

from typing import List, Tuple, Dict
import numpy as np
from .threshold_partitioner_interface import ThresholdPartitioner
from ...types import Message


class FixedThresholdPartitioner(ThresholdPartitioner):
    """Fixed threshold partitioner using sigmoid normalization and percentiles."""
    
    def __init__(
        self,
        high_percentile: float = 85.0,
        low_percentile: float = 15.0,
        apply_sigmoid: bool = True
    ):
        """
        Initialize fixed threshold partitioner.
        
        Args:
            high_percentile: Percentile for high confidence threshold (default: 85)
            low_percentile: Percentile for low confidence threshold (default: 15)
            apply_sigmoid: Whether to apply sigmoid normalization (default: True)
        """
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        self.apply_sigmoid = apply_sigmoid
    
    def partition_candidates(
        self,
        candidates: List[Message],
        scores: Dict[str, float]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Partition candidates using fixed percentile thresholds.
        
        Args:
            candidates: List of candidate messages
            scores: Dictionary mapping message ID to score
            
        Returns:
            Tuple of (high_confidence, grey_area, low_confidence) lists of message IDs
        """
        if not scores:
            return [], [], []
        
        # Extract scores and apply sigmoid if requested
        score_values = list(scores.values())
        if self.apply_sigmoid:
            score_values = self._apply_sigmoid(score_values)
        
        # Calculate percentile thresholds
        high_threshold = np.percentile(score_values, self.high_percentile)
        low_threshold = np.percentile(score_values, self.low_percentile)
        
        # Partition candidates
        high_confidence = []
        grey_area = []
        low_confidence = []
        
        for message in candidates:
            score = scores.get(message.msg_id, 0.0)
            if self.apply_sigmoid:
                score = self._sigmoid(score)
            
            if score >= high_threshold:
                high_confidence.append(message.msg_id)
            elif score <= low_threshold:
                low_confidence.append(message.msg_id)
            else:
                grey_area.append(message.msg_id)
        
        return high_confidence, grey_area, low_confidence
    
    def _apply_sigmoid(self, scores: List[float]) -> List[float]:
        """Apply sigmoid function to normalize scores to [0, 1] range."""
        return [self._sigmoid(score) for score in scores]
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function: 1 / (1 + exp(-x))"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
