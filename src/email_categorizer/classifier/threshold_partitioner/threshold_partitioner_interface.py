"""
Base class for threshold-based classification.

Provides a common interface for different thresholding strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import settings

from ...types import Message


class ThresholdPartitioner(ABC):
    """Abstract base class for threshold-based partitioning."""
    
    @abstractmethod
    def partition_candidates(
        self,
        candidates: List[Message],
        scores: Dict[str, float]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Classify candidates into three tiers based on scores.
        
        Args:
            candidates: List of candidate messages
            scores: Dictionary mapping message ID to score
            
        Returns:
            Tuple of (high_confidence, grey_area, low_confidence) lists of ids
        """
        pass