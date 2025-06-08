"""
Emotion-aware Reservoir for Differential Emotion Theory Framework

This module implements reservoir computing nodes with integrated emotional processing,
combining reservoir dynamics with differential emotion theory.
"""

import numpy as np
from typing import Optional, Dict, Callable

from ...activationsfunc import tanh
from ...node import Node
from ...type import Activation
from ..reservoirs.base import BaseReservoir
from .differential_emotion import DifferentialEmotionProcessor


class EmotionReservoir(BaseReservoir):
    """
    Emotion-aware reservoir combining reservoir computing with differential emotion theory.
    
    This reservoir incorporates emotional states into its dynamics, allowing for
    affectively-modulated computation and emotion-reservoir state interactions.
    
    Parameters
    ----------
    units : int
        Number of reservoir units.
    emotion_integration : float, optional
        Strength of emotion integration into reservoir dynamics. Default is 0.1.
    emotion_dimensions : int, optional
        Number of emotion dimensions. Default is 10.
    emotion_feedback : bool, optional
        Whether reservoir states influence emotion computation. Default is True.
    lr : float, optional
        Leaking rate of the reservoir. Default is 1.0.
    sr : float, optional  
        Spectral radius of the reservoir. Default is 0.9.
    activation : Callable, optional
        Activation function. Default is tanh.
    """
    
    def __init__(
        self,
        units: int,
        emotion_integration: float = 0.1,
        emotion_dimensions: int = 10,
        emotion_feedback: bool = True,
        lr: float = 1.0,
        sr: float = 0.9,
        activation: Callable = tanh,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            output_dim=units + emotion_dimensions + 2,  # reservoir + emotions + valence/arousal
            lr=lr,
            sr=sr,
            activation=activation,
            name=name,
            **kwargs
        )
        
        self.units = units
        self.emotion_integration = emotion_integration
        self.emotion_dimensions = emotion_dimensions
        self.emotion_feedback = emotion_feedback
        
        # Initialize emotion processor
        self.emotion_processor = DifferentialEmotionProcessor(
            emotion_dimensions=emotion_dimensions,
            valence_arousal=True,
            temporal_dynamics=True
        )
        
        # Reservoir state
        self.state = np.zeros(units)
        self.emotion_state = np.zeros(emotion_dimensions + 2)
        
        # Emotion-reservoir coupling matrices (initialized later)
        self.emotion_to_reservoir = None
        self.reservoir_to_emotion = None
        
    def initialize(self, x: Optional[Activation] = None) -> "EmotionReservoir":
        """Initialize the emotion-aware reservoir."""
        # Initialize base reservoir
        super().initialize(x)
        
        if x is not None:
            input_dim = x.shape[-1] if hasattr(x, 'shape') else len(x)
            
            # Initialize emotion processor
            self.emotion_processor.initialize(x)
            
            # Initialize emotion-reservoir coupling
            self.emotion_to_reservoir = np.random.randn(
                self.emotion_dimensions + 2, self.units
            ) * 0.05
            
            if self.emotion_feedback:
                self.reservoir_to_emotion = np.random.randn(
                    self.units, input_dim
                ) * 0.05
        
        return self
    
    def forward(self, x: Activation) -> Activation:
        """
        Forward pass with emotion-reservoir coupling.
        
        Parameters
        ----------
        x : Activation
            Input activation
            
        Returns
        -------
        Activation
            Combined reservoir and emotion state
        """
        if not hasattr(self, 'W'):
            self.initialize(x)
        
        # Process emotions from input
        if self.emotion_feedback and self.reservoir_to_emotion is not None:
            # Use reservoir state to modulate emotion input
            emotion_input = x + np.dot(self.state, self.reservoir_to_emotion)
            emotions = self.emotion_processor.forward(emotion_input)
        else:
            emotions = self.emotion_processor.forward(x)
        
        # Compute reservoir dynamics with emotion integration
        reservoir_input = np.dot(x, self.Win)
        
        # Add emotional modulation to reservoir input
        if self.emotion_to_reservoir is not None:
            emotion_modulation = np.dot(emotions, self.emotion_to_reservoir)
            reservoir_input += self.emotion_integration * emotion_modulation
        
        # Standard reservoir update
        pre_activation = (
            (1 - self.lr) * self.state +
            self.lr * (np.dot(self.W, self.state) + reservoir_input)
        )
        
        self.state = self.activation(pre_activation)
        self.emotion_state = emotions
        
        # Return combined state
        combined_state = np.concatenate([self.state, self.emotion_state])
        return combined_state
    
    def get_reservoir_state(self) -> np.ndarray:
        """Get current reservoir state (without emotions)."""
        return self.state.copy()
    
    def get_emotion_state(self) -> np.ndarray:
        """Get current emotion state."""
        return self.emotion_state.copy()
    
    def get_dominant_emotion(self) -> tuple:
        """Get the currently dominant emotion."""
        return self.emotion_processor.get_dominant_emotion()
    
    def get_valence_arousal(self) -> tuple:
        """Get current valence and arousal."""
        return self.emotion_processor.get_valence_arousal()
    
    def get_emotion_labels(self) -> list:
        """Get emotion labels."""
        return self.emotion_processor.emotion_labels
    
    def set_emotion_integration(self, strength: float):
        """Set emotion integration strength."""
        self.emotion_integration = strength
    
    def reset_emotion_state(self):
        """Reset emotion processor state."""
        self.emotion_processor.emotion_state = np.zeros(self.emotion_processor.output_dim)
        self.emotion_processor.previous_state = np.zeros(self.emotion_processor.output_dim)
        self.emotion_state = np.zeros(self.emotion_dimensions + 2)