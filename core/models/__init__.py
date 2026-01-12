"""
Luminark Models Package

Production-grade multimodal deepfake detection models.
"""

from .base import BaseDetector, DetectionResult
from .video import VideoAnalyzer
from .audio import AudioAnalyzer
from .rppg import RppgAnalyzer
from .lipsync import LipsyncAnalyzer
from .fusion import LateFusionEnsemble, FusionResult

__all__ = [
    # Base
    'BaseDetector',
    'DetectionResult',
    
    # Detectors
    'VideoAnalyzer',
    'AudioAnalyzer',
    'RppgAnalyzer',
    'LipsyncAnalyzer',
    
    # Fusion
    'LateFusionEnsemble',
    'FusionResult',
]
