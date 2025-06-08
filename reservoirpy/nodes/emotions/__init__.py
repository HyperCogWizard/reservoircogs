"""
Emotion nodes for Differential Emotion Theory Framework.

This module implements affective computing capabilities using Differential Emotion Theory,
integrated with OpenCog AtomSpace symbolic reasoning and reservoir computing core algorithms.
"""

try:
    from .emotion_reservoir import EmotionReservoir
    from .differential_emotion import DifferentialEmotionProcessor
    
    __all__ = [
        "EmotionReservoir",
        "DifferentialEmotionProcessor",
    ]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []