# Getting Started with ReservoirCogs

## Python Version Support

Full support is guaranteed for Python 3.8 and higher. Support is partial for Python 3.7.

## Installation

### Installing Stable Release (v0.3.0)

ReservoirPy can be installed via pip from [PyPI](https://pypi.org/project/reservoirpy):

```bash
pip install reservoirpy
```

### Installing Previous Stable Release (v0.2.4)

```bash
pip install reservoirpy==0.2.4
```

### Complete Installation Guide

For a complete walkthrough for beginners and developers, see our [advanced installation guide](developer_guide/advanced_install.md).

## OpenCog AtomSpace Integration

ReservoirCogs now includes deep integration with OpenCog AtomSpace for symbolic AI capabilities:

### Building with OpenCog Support

```bash
# Install OpenCog dependencies
sudo apt-get install opencog-dev

# Build ReservoirCogs with CMake
mkdir build && cd build
cmake ..
make -j4
make install
```

### Using AtomSpace Integration

```cpp
#include <opencog/reservoir/nodes/ReservoirNode.h>
#include <opencog/atomspace/AtomSpace.h>

// Create AtomSpace and reservoir
auto atomspace = createAtomSpace();
auto reservoir = std::make_shared<EchoStateNetwork>(atomspace, 100, 3, 1);

// Train and use the reservoir
std::vector<double> input = {1.0, 0.5, -0.3};
auto output = reservoir->predict(input);
```

## Project Philosophy

Most machine learning work these days is becoming increasingly complicated. To deal with this complexity, many open source projects have emerged that allow users to go from simple models to over-complicated architectures.

This is not the philosophy of reservoir computing, and thus, not the philosophy of ReservoirCogs.

Because Reservoir Computing aims at making complexity emerge from apparent simplicity, ReservoirCogs provides users with very simple tools that can achieve a wide range of machine learning tasks, particularly when dealing with sequential data.

These tools are based on:
- [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/) for Python components
- [OpenCog AtomSpace](https://github.com/opencog/atomspace) for symbolic AI integration
- Modern C++ for high-performance computing

They can be mastered by any Python or C++ enthusiast, from beginners to experts.

## Learn More

You can now start using ReservoirCogs! Learn more about the software and its capabilities in the [User Guide](user_guide/index.md). You can also find tutorials and examples in the [GitHub repository](https://github.com/HyperCogWizard/reservoircogs/tree/main/tutorials).

## Key Features

- **Easy creation** of complex architectures with multiple reservoirs (e.g. *deep reservoirs*), readouts
- **Feedback loops** for advanced temporal processing
- **Intrinsic plasticity** for adaptive reservoirs
- **Online learning** capabilities
- **OpenCog AtomSpace integration** for symbolic AI
- **High-performance C++ implementation** for production use
- **Python compatibility** for research and prototyping