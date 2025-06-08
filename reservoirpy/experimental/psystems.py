# Author: GitHub Copilot at 08/06/2025 <copilot@github.com>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""
P-Systems Membrane Computing implementation for ReservoirCogs

P-Systems (Membrane Computing) enable bio-inspired computational models with
hierarchical processing using membrane-based reservoir partitions.
"""

import sys
from typing import Dict, List, Optional, Union
import numpy as np

from ..node import Node
from ..utils.validation import is_array

if sys.version_info < (3, 8):
    from typing_extensions import Callable
else:
    from typing import Callable


def forward(membrane_comp, x):
    """
    Forward pass through membrane computing system.
    
    Each membrane processes data hierarchically with its own reservoir dynamics.
    """
    hierarchical = membrane_comp.hypers["hierarchical"]
    membrane_states = membrane_comp.get_param("membrane_states")
    
    # Process input through each membrane
    membrane_outputs = []
    
    for i, membrane_state in enumerate(membrane_states):
        if hierarchical and i > 0:
            # Hierarchical processing: use previous membrane output as input
            membrane_input = membrane_outputs[i-1]
        else:
            # Parallel processing: all membranes receive original input
            membrane_input = x
            
        # Simple membrane dynamics (can be extended with P-lingua rules)
        membrane_output = membrane_comp._process_membrane(membrane_input, membrane_state, i)
        membrane_outputs.append(membrane_output)
        
        # Update membrane state
        membrane_states[i] = membrane_comp._update_membrane_state(
            membrane_state, membrane_input, membrane_output, i
        )
    
    membrane_comp.set_param("membrane_states", membrane_states)
    
    # Combine outputs from all membranes
    if hierarchical:
        # In hierarchical mode, return output from final membrane
        return membrane_outputs[-1]
    else:
        # In parallel mode, combine all membrane outputs
        return np.concatenate(membrane_outputs, axis=-1)


def initialize(
    membrane_comp,
    x=None,
    y=None,
    seed=None,
):
    """Initialize membrane computing system."""
    if x is not None:
        membrane_comp.set_input_dim(x.shape[-1])
        
    # Get hyperparameters
    membranes = membrane_comp.hypers["membranes"]
    hierarchical = membrane_comp.hypers["hierarchical"]
    p_lingua_rules = membrane_comp.hypers["p_lingua_rules"]
    membrane_size = membrane_comp.hypers["membrane_size"]
    
    # Initialize membrane states
    membrane_states = []
    for i in range(membranes):
        # Each membrane has its own internal state
        state = {
            "internal_state": np.zeros((1, membrane_size)),
            "weights": np.random.uniform(-1, 1, (membrane_size, membrane_size)),
            "input_weights": np.random.uniform(-1, 1, (membrane_size, x.shape[-1] if x is not None else 1)),
            "output_weights": np.random.uniform(-1, 1, (membrane_size, membrane_size // 2)),
        }
        membrane_states.append(state)
    
    membrane_comp.set_param("membrane_states", membrane_states)
    
    # Set output dimensions
    if hierarchical:
        output_dim = membrane_size // 2  # Final membrane output
    else:
        output_dim = membranes * (membrane_size // 2)  # All membrane outputs combined
        
    membrane_comp.set_output_dim(output_dim)


class MembraneComputing(Node):
    """
    P-Systems Membrane Computing Node for hierarchical reservoir processing.
    
    This implementation provides bio-inspired computational models using
    membrane-based reservoir partitions with optional P-lingua rule integration.
    
    Parameters
    ----------
    membranes : int, default=3
        Number of membranes in the system
    hierarchical : bool, default=True
        Whether to use hierarchical (True) or parallel (False) processing
    p_lingua_rules : str, optional
        Path to P-lingua rules file (for future extension)
    membrane_size : int, default=100
        Size of each membrane's internal reservoir
    name : str, optional
        Node name
        
    Examples
    --------
    >>> from reservoirpy.experimental import MembraneComputing
    >>> 
    >>> # Create hierarchical membrane system
    >>> membrane_reservoir = MembraneComputing(
    ...     membranes=3,
    ...     hierarchical=True,
    ...     membrane_size=100
    ... )
    >>> 
    >>> # Process data through membrane system
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> output = membrane_reservoir.run(X)
    """
    
    def __init__(
        self,
        membranes: int = 3,
        hierarchical: bool = True,
        p_lingua_rules: Optional[str] = None,
        membrane_size: int = 100,
        name: Optional[str] = None,
    ):
        
        # Validate parameters
        if membranes <= 0:
            raise ValueError("Number of membranes must be positive")
        if membrane_size <= 0:
            raise ValueError("Membrane size must be positive")
        
        # Partial initialization function with fixed parameters  
        def _initialize(node, x=None, y=None, seed=None):
            return initialize(node, x=x, y=y, seed=seed)
        
        super(MembraneComputing, self).__init__(
            params={
                "membrane_states": None,
            },
            hypers={
                "membranes": membranes,
                "hierarchical": hierarchical,
                "p_lingua_rules": p_lingua_rules,
                "membrane_size": membrane_size,
            },
            forward=forward,
            initializer=_initialize,
            name=name
        )
        
        # Store parameters as instance attributes for easy access
        self.membranes = membranes
        self.hierarchical = hierarchical
        self.p_lingua_rules = p_lingua_rules
        self.membrane_size = membrane_size
    
    def _process_membrane(self, x, membrane_state, membrane_idx):
        """Process input through a single membrane."""
        # Get membrane components
        internal_state = membrane_state["internal_state"]
        weights = membrane_state["weights"]
        input_weights = membrane_state["input_weights"]
        output_weights = membrane_state["output_weights"]
        
        # Handle input dimension mismatch in hierarchical processing
        if x.shape[1] != input_weights.shape[1]:
            # For hierarchical processing, resize input weights to match current input
            input_weights = np.random.uniform(-1, 1, (input_weights.shape[0], x.shape[1]))
            membrane_state["input_weights"] = input_weights
        
        # Reservoir dynamics for this membrane
        # Simple echo state dynamics (can be extended with P-lingua rules)
        reservoir_input = np.tanh(np.dot(x, input_weights.T) + np.dot(internal_state, weights))
        
        # Apply membrane-specific processing
        # This is where P-lingua rules would be applied in future extensions
        membrane_output = np.dot(reservoir_input, output_weights)
        
        return membrane_output
    
    def _update_membrane_state(self, membrane_state, x, output, membrane_idx):
        """Update internal state of a membrane."""
        # Simple update rule - can be extended with P-lingua semantics
        weights = membrane_state["weights"]
        input_weights = membrane_state["input_weights"]
        
        # Handle input dimension mismatch
        if x.shape[1] != input_weights.shape[1]:
            input_weights = membrane_state["input_weights"]  # Use updated weights from _process_membrane
        
        new_internal_state = np.tanh(
            np.dot(x, input_weights.T) + 
            np.dot(membrane_state["internal_state"], weights)
        )
        
        # Update the membrane state
        updated_state = membrane_state.copy()
        updated_state["internal_state"] = new_internal_state
        
        return updated_state
    
    def get_membrane_states(self):
        """Get current states of all membranes."""
        return self.get_param("membrane_states")
    
    def set_p_lingua_rules(self, rules_file: str):
        """Set P-lingua rules file for advanced membrane processing."""
        self.hypers["p_lingua_rules"] = rules_file
        # TODO: Implement P-lingua rule parsing and application
    
    def get_membrane_info(self):
        """Get information about membrane configuration."""
        return {
            "membranes": self.membranes,
            "hierarchical": self.hierarchical,
            "membrane_size": self.membrane_size,
            "p_lingua_rules": self.p_lingua_rules,
        }