# ReservoirCogs User Guide

## Overview

ReservoirCogs is a flexible and powerful library for reservoir computing with deep OpenCog AtomSpace integration. This user guide covers both the Python API (compatible with ReservoirPy) and the new C++ AtomSpace integration.

## Table of Contents

1. [Quick Start](quickstart.md) - Get up and running quickly
2. [Node API](node.md) - Understanding the core Node abstraction
3. [Model Creation](model.md) - Building complex reservoir architectures  
4. [Learning Rules](learning_rules.md) - Training and adaptation algorithms
5. [Hyperparameter Optimization](hyper.md) - Automated parameter tuning
6. [AtomSpace Integration](atomspace_integration.md) - Using symbolic AI features
7. [ReservoirChat Playground](playground.md) - Interactive exploration platform
8. [Advanced Examples](advanced_demo.md) - Complex use cases and patterns
9. [Creating Custom Nodes](create_new_node.md) - Extending the framework
10. [Compatibility](compat.md) - Working with legacy ReservoirPy code

## Python API Overview

The Python API provides a familiar interface for reservoir computing research:

```python
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

# Create a simple ESN
reservoir = Reservoir(100, lr=0.3, sr=0.9)
readout = Ridge(ridge=1e-6)

# Connect components
model = reservoir >> readout

# Train the model
model.fit(X_train, y_train)
predictions = model.run(X_test)
```

## C++ AtomSpace API Overview

The C++ API provides high-performance computing with symbolic AI integration:

```cpp
#include <opencog/reservoir/nodes/ReservoirNode.h>
#include <opencog/reservoir/algorithms/ReservoirAlgorithms.h>

using namespace opencog::reservoir;

// Create AtomSpace and reservoir components
auto atomspace = createAtomSpace();
auto esn = std::make_shared<EchoStateNetwork>(atomspace, 100, 3, 1);

// Configure reservoir properties
esn->set_leaking_rate(0.3);
esn->set_spectral_radius(0.9);

// Use algorithms for training
algorithms::ReservoirTrainer trainer(atomspace);
trainer.train_esn_ridge_regression(esn, train_inputs, train_targets);

// Make predictions
auto predictions = esn->predict(test_input);
```

## Key Concepts

### Reservoir Computing

Reservoir computing is a framework for computation with dynamical systems. The main idea is to use a fixed, randomly generated dynamical system (the "reservoir") to transform input signals into a high-dimensional representation, then train only the output weights.

### Echo State Networks (ESNs)

ESNs are the most common type of reservoir computer. They consist of:
- **Input layer**: Connects external inputs to the reservoir
- **Reservoir**: A sparsely connected recurrent network with fixed weights
- **Output layer**: Linear combination of reservoir states (trainable)

### AtomSpace Integration

OpenCog's AtomSpace provides:
- **Symbolic representation** of reservoir states and dynamics
- **Graph-based knowledge** storage and reasoning
- **Temporal logic** for sequence modeling
- **Concept formation** from reservoir patterns

## Getting Help

- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/HyperCogWizard/reservoircogs/issues)
- **Discussions**: Join the conversation on [GitHub Discussions](https://github.com/HyperCogWizard/reservoircogs/discussions)
- **Documentation**: This user guide and API reference
- **Examples**: See the `examples/` directory for complete examples

## Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details on how to:
- Report bugs
- Suggest features  
- Submit pull requests
- Improve documentation