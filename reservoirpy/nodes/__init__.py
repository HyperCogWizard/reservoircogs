"""
============================================
ReservoirPy Nodes (:mod:`reservoirpy.nodes`)
============================================

.. currentmodule:: reservoirpy.nodes

Reservoirs
==========

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Reservoir - Recurrent pool of leaky integrator neurons
   NVAR - Non-linear Vector Autoregressive machine (NG-RC)
   IPReservoir - Reservoir with intrinsic plasticity learning rule
   LocalPlasticityReservoir - Reservoir with weight plasticity
   EmotionReservoir - Emotion-aware reservoir with differential emotion theory

Offline readouts
================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Ridge - Layer of neurons connected through offline linear regression.
   ScikitLearnNode - Interface for linear models from the scikit-learn library.

Online readouts
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   LMS - Layer of neurons connected through least mean squares learning rule.
   RLS - Layer of neurons connected through recursive least squares learning rule.
   FORCE - Layer of neurons connected through online learning rules.

Optimized ESN
=============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ESN - Echo State Network model with distributed offline learning.

Activation functions
====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Tanh - Hyperbolic tangent node.
   Sigmoid - Logistic function node.
   Softmax - Softmax node.
   Softplus - Softplus node.
   ReLU - Rectified Linear Unit node.
   Identity - Identity function node.

Input and Output
================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Input - Input node, used to distribute input data to other nodes.
   Output - Output node, used to gather stated from hidden nodes.

Operators
=========

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Concat - Concatenate vector of data along feature axis.
   Delay - Adds a discrete delay between input and output.

Emotion Processing
==================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   DifferentialEmotionProcessor - Differential emotion theory processor for affective computing.
"""
# Author: Nathan Trouvain at 16/12/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from .activations import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh
from .concat import Concat
from .delay import Delay
from .esn import ESN
from .io import Input, Output
from .readouts import FORCE, LMS, RLS, Ridge, ScikitLearnNode
from .reservoirs import NVAR, IPReservoir, LocalPlasticityReservoir, Reservoir

# Conditional import for emotion nodes (may require additional dependencies)
try:
    from .emotions import DifferentialEmotionProcessor, EmotionReservoir
    _emotion_imports_available = True
except ImportError:
    _emotion_imports_available = False

__all__ = [
    "Reservoir",
    "Input",
    "Output",
    "Ridge",
    "FORCE",
    "LMS",
    "RLS",
    "Tanh",
    "Softmax",
    "Softplus",
    "Identity",
    "Sigmoid",
    "ReLU",
    "NVAR",
    "ESN",
    "Concat",
    "Delay",
    "IPReservoir",
    "ScikitLearnNode",
    "LocalPlasticityReservoir",
]

# Add emotion nodes to __all__ if available
if _emotion_imports_available:
    __all__.extend([
        "DifferentialEmotionProcessor",
        "EmotionReservoir",
    ])
