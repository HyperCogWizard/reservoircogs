"""
Differential Emotion Theory Framework - Core Processor

This module implements the core differential emotion processor based on Differential Emotion Theory,
providing symbolic emotion representation and reasoning capabilities integrated with AtomSpace.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ...node import Node
from ...type import Activation


class DifferentialEmotionProcessor(Node):
    """
    Differential Emotion Theory processor for affective computing.
    
    Implements basic emotions based on Differential Emotion Theory:
    - Interest, Joy, Surprise, Sadness, Anger, Disgust, Contempt, Fear, Shame, Guilt
    
    Parameters
    ----------
    emotion_dimensions : int, optional
        Number of emotion dimensions to process. Default is 10 for basic emotions.
    valence_arousal : bool, optional
        Whether to include valence-arousal dimensions. Default is True.
    temporal_dynamics : bool, optional
        Whether to model temporal emotion dynamics. Default is True.
    """
    
    def __init__(
        self,
        emotion_dimensions: int = 10,
        valence_arousal: bool = True,
        temporal_dynamics: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            output_dim=emotion_dimensions + (2 if valence_arousal else 0),
            name=name,
            **kwargs
        )
        
        self.emotion_dimensions = emotion_dimensions
        self.valence_arousal = valence_arousal
        self.temporal_dynamics = temporal_dynamics
        
        # Basic emotion labels from Differential Emotion Theory
        self.emotion_labels = [
            "interest", "joy", "surprise", "sadness", "anger",
            "disgust", "contempt", "fear", "shame", "guilt"
        ][:emotion_dimensions]
        
        if valence_arousal:
            self.emotion_labels.extend(["valence", "arousal"])
        
        # Initialize emotion state
        self.emotion_state = np.zeros(self.output_dim)
        self.previous_state = np.zeros(self.output_dim)
        
        # Temporal parameters
        self.decay_rate = 0.1
        self.adaptation_rate = 0.05
        
    def initialize(self, x: Optional[Activation] = None) -> "DifferentialEmotionProcessor":
        """Initialize the emotion processor."""
        if x is not None:
            input_dim = x.shape[-1] if hasattr(x, 'shape') else len(x)
            # Initialize emotion mapping weights
            self.emotion_weights = np.random.randn(input_dim, self.emotion_dimensions) * 0.1
            
            if self.valence_arousal:
                self.valence_weights = np.random.randn(input_dim, 1) * 0.1
                self.arousal_weights = np.random.randn(input_dim, 1) * 0.1
        
        return self
    
    def forward(self, x: Activation) -> Activation:
        """
        Process input through differential emotion theory framework.
        
        Parameters
        ----------
        x : Activation
            Input activation (e.g., from reservoir states or sensory input)
            
        Returns
        -------
        Activation
            Emotion state vector with basic emotions and optional valence/arousal
        """
        if not hasattr(self, 'emotion_weights'):
            self.initialize(x)
        
        # Compute basic emotion activations
        emotion_raw = np.dot(x, self.emotion_weights)
        emotions = self._apply_emotion_dynamics(emotion_raw)
        
        output = emotions.copy()
        
        # Add valence and arousal if enabled
        if self.valence_arousal:
            valence = np.dot(x, self.valence_weights).flatten()
            arousal = np.dot(x, self.arousal_weights).flatten()
            output = np.concatenate([emotions, valence, arousal])
        
        # Apply temporal dynamics if enabled
        if self.temporal_dynamics:
            output = self._apply_temporal_dynamics(output)
        
        self.previous_state = self.emotion_state.copy()
        self.emotion_state = output.copy()
        
        return output
    
    def _apply_emotion_dynamics(self, raw_emotions: np.ndarray) -> np.ndarray:
        """Apply differential emotion theory dynamics to raw activations."""
        # Handle batched input
        if raw_emotions.ndim > 1:
            raw_emotions = raw_emotions.flatten()
            
        # Apply softmax for competition between emotions
        emotions = np.exp(raw_emotions - np.max(raw_emotions))
        emotions = emotions / np.sum(emotions)
        
        # Apply emotion-specific transformations only if indices exist
        # Interest and joy are approach emotions (positive)
        if len(emotions) > 0:
            emotions[0] = max(0, emotions[0])  # interest
        if len(emotions) > 1:
            emotions[1] = max(0, emotions[1])  # joy
        
        # Fear, sadness, anger are avoidance emotions (be careful with indices)
        if len(emotions) > 3:
            emotions[3] = max(0, emotions[3])  # sadness
        if len(emotions) > 4:
            emotions[4] = max(0, emotions[4])  # anger
        if len(emotions) > 7:
            emotions[7] = max(0, emotions[7])  # fear
        
        return emotions
    
    def _apply_temporal_dynamics(self, current_emotion: np.ndarray) -> np.ndarray:
        """Apply temporal emotion dynamics including decay and adaptation."""
        # Emotion decay
        decayed_previous = self.previous_state * (1 - self.decay_rate)
        
        # Adaptive integration
        adapted_current = current_emotion * self.adaptation_rate
        
        # Combine with momentum
        result = decayed_previous + adapted_current
        
        # Normalize to maintain emotion intensity bounds
        if np.sum(result[:self.emotion_dimensions]) > 0:
            result[:self.emotion_dimensions] = (
                result[:self.emotion_dimensions] / np.sum(result[:self.emotion_dimensions])
            )
        
        return result
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the currently dominant emotion and its intensity."""
        basic_emotions = self.emotion_state[:self.emotion_dimensions]
        dominant_idx = np.argmax(basic_emotions)
        return self.emotion_labels[dominant_idx], basic_emotions[dominant_idx]
    
    def get_valence_arousal(self) -> Optional[Tuple[float, float]]:
        """Get current valence and arousal values if enabled."""
        if not self.valence_arousal:
            return None
        return (
            self.emotion_state[-2],  # valence
            self.emotion_state[-1]   # arousal
        )
    
    def get_emotion_vector(self) -> Dict[str, float]:
        """Get current emotion state as labeled dictionary."""
        return {
            label: value 
            for label, value in zip(self.emotion_labels, self.emotion_state)
        }