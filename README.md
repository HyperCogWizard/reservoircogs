<div align="center">
  <img src="static/rpy_banner_light.png#gh-light-mode-only">
  <img src="static/rpy_banner_dark.png#gh-dark-mode-only">

  **ReservoirCogs: Advanced Reservoir Computing with OpenCog AtomSpace Integration**

  Flexible library for Reservoir Computing architectures like Echo State Networks (ESN) with deep symbolic AI integration through OpenCog AtomSpace.

  [![PyPI version](https://badge.fury.io/py/reservoirpy.svg)](https://badge.fury.io/py/reservoirpy)
  [![HAL](https://img.shields.io/badge/HAL-02595026-white?style=flat&logo=HAL&logoColor=white&labelColor=B03532&color=grey)](https://inria.hal.science/hal-02595026)
  ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reservoirpy)
  [![OpenCog Integration](https://img.shields.io/badge/OpenCog-AtomSpace-blue)](https://github.com/opencog/atomspace)
  <br/>
  [![Downloads](https://static.pepy.tech/badge/reservoirpy)](https://pepy.tech/project/reservoirpy)
  [![Documentation Status](https://readthedocs.org/projects/reservoirpy/badge/?version=latest)](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
  [![Testing](https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml)
  [![codecov](https://codecov.io/gh/reservoirpy/reservoirpy/branch/master/graph/badge.svg?token=JC8R1PB5EO)](https://codecov.io/gh/reservoirpy/reservoirpy)
</div>



---

<p> <img src="static/googlecolab.svg" alt="Google Colab icon" width=32 height=32 align="left"><b>Tutorials:</b> <a href="https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/1-Getting_Started.ipynb">Open in Colab</a> </p>
<p> <img src="static/documentation.svg" alt="Open book icon" width=32 height=32 align="left"><b>Documentation:</b> <a href="docs/getting_started.md">Getting Started</a> | <a href="docs/user_guide/index.md">User Guide</a> | <a href="docs/user_guide/atomspace_integration.md">AtomSpace Integration</a></p>
<p> <img src="static/cpp_icon.svg" alt="C++ icon" width=32 height=32 align="left"><b>C++ API:</b> High-performance reservoir computing with OpenCog AtomSpace integration</p>
<p> 📋 <img src="https://img.shields.io/badge/GitHub-Project-blue?logo=github" alt="GitHub Project icon" width=24 height=24 align="left"><b>Project Board:</b> <a href=".github/PROJECT_ROADMAP.md">Development Roadmap</a> | <a href="https://github.com/orgs/HyperCogWizard/projects/1">GitHub Project</a></p>

---

> [!TIP]
> 🎉 Exciting News! We just launched a new beta tool based on a Large Language Model!
> 🚀 You can chat with **ReservoirChat** and ask anything about Reservoir Computing and ReservoirPy! 🤖💡
> Don’t miss out, it’s available for a limited time! ⏳
> 
> https://chat.reservoirpy.inria.fr
> 
> **🌟 NEW: [ReservoirChat Playground](playground/index.html) - Mindbendingly Amazing Interactive Experience!**

<br />

---

## 🚀 NEW: OpenCog AtomSpace Integration

ReservoirCogs now includes deep integration with OpenCog AtomSpace for symbolic AI capabilities:

- **Symbolic Representation**: Store reservoir states and dynamics as AtomSpace knowledge
- **Temporal Logic**: Reason about sequences and temporal patterns  
- **Knowledge Extraction**: Extract learned patterns as symbolic concepts
- **High Performance**: C++ implementation for production systems
- **Hybrid AI**: Combine neural reservoir computing with symbolic reasoning

---

**Feature overview:**
- Easy creation of [complex architectures](docs/user_guide/model.md) with multiple reservoirs (e.g. *deep reservoirs*), readouts
- [Feedback loops](docs/user_guide/advanced_demo.md#Feedback-connections) and advanced temporal processing
- **OpenCog AtomSpace integration** for symbolic AI and knowledge representation
- **Differential Emotion Theory Framework** for affective computing and emotionally aware AI
- **High-performance C++ implementation** alongside Python compatibility
- [Intrinsic plasticity](examples/Improving%20reservoirs%20using%20Intrinsic%20Plasticity/Intrinsic_Plasiticity_Schrauwen_et_al_2008.ipynb) for adaptive reservoir dynamics
- [Online learning](docs/user_guide/learning_rules.md) for real-time applications
- [Evolutionary optimization](docs/user_guide/atomspace_integration.md#evolutionary-optimization) of reservoir topologies
- **Symbolic pattern recognition** and temporal logic reasoning
- **Hybrid neural-symbolic** architectures
- [offline and online training](https://reservoirpy.readthedocs.io/en/latest/user_guide/learning_rules.html)
- [parallel implementation](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.nodes.ESN.html)
- sparse matrix computation
- advanced learning rules (e.g. [*Intrinsic Plasticity*](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.nodes.IPReservoir.html), [*Local Plasticity*](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.nodes.LocalPlasticityReservoir.html) or [*NVAR* (Next-Generation RC)](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.nodes.NVAR.html))
- interfacing with [scikit-learn](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.nodes.ScikitLearnNode.html) models
- and many more!

Moreover, graphical tools are included to **easily explore hyperparameters**
with the help of the *hyperopt* library.

## Quick try ⚡

### Python Installation

```bash
pip install reservoirpy
```

### C++ Installation with OpenCog

```bash
# Install OpenCog dependencies
sudo apt-get install opencog-dev

# Clone and build ReservoirCogs
git clone https://github.com/HyperCogWizard/reservoircogs.git
cd reservoircogs
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### Basic Python Usage

```python
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

# Create a simple ESN
reservoir = Reservoir(100, lr=0.3, sr=0.9)
readout = Ridge(ridge=1e-6)

# Connect and train
model = reservoir >> readout
model.fit(X_train, y_train)
predictions = model.run(X_test)
```

### C++ AtomSpace Usage

```cpp
#include <opencog/reservoir/nodes/ReservoirNode.h>

// Create AtomSpace and reservoir
auto atomspace = createAtomSpace();
auto esn = std::make_shared<EchoStateNetwork>(atomspace, 100, 3, 1);

// Configure and train
esn->set_leaking_rate(0.3);
algorithms::ReservoirTrainer trainer(atomspace);
trainer.train_esn_ridge_regression(esn, inputs, targets);
auto predictions = esn->predict(test_input);
```

### An example on chaotic timeseries prediction (Mackey-Glass)

For a general introduction to reservoir computing and ReservoirPy features, take
a look at the [tutorials](#tutorials)

```python
from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare

### Step 1: Load the dataset

X = mackey_glass(n_timesteps=2000)  # (2000, 1)-shaped array
# create y by shifting X, and train/test split
x_train, x_test, y_train, y_test = to_forecasting(X, test_size=0.2)

### Step 2: Create an Echo State Network

# 100 neurons reservoir, spectral radius = 1.25, leak rate = 0.3
reservoir = Reservoir(units=100, sr=1.25, lr=0.3)
# feed-forward layer of neurons, trained with L2-regularization
readout = Ridge(ridge=1e-5)
# connect the two nodes
esn = reservoir >> readout

### Step 3: Fit, run and evaluate the ESN

esn.fit(x_train, y_train, warmup=100)
predictions = esn.run(x_test)

print(f"RMSE: {rmse(y_test, predictions)}; R^2 score: {rsquare(y_test, predictions)}")
# RMSE: 0.0020282; R^2 score: 0.99992
```


## More examples and tutorials 🎓

### Tutorials

- [**1 - Getting started with ReservoirPy**](./tutorials/1-Getting_Started.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_Getting_started-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/1-Getting_Started.ipynb)
- [**2 - Advanced features**](./tutorials/2-Advanced_Features.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_Advanced_features-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/2-Advanced_Features.ipynb)
- [**3 - General introduction to Reservoir Computing**](./tutorials/3-General_Introduction_to_Reservoir_Computing.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_Introduction_to_RC-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/3-General_Introduction_to_Reservoir_Computing.ipynb)
- [**4 - Understand and optimise hyperparameters**](./tutorials/4-Understand_and_optimize_hyperparameters.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_Hyperparameters-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb)
- [**5 - Classification with reservoir computing**](./tutorials/5-Classification-with-RC.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_Classification-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/5-Classification-with-RC.ipynb)
- [**6 - Interfacing ReservoirPy with scikit-learn**](./tutorials/6-Interfacing_with_scikit-learn.ipynb)
[![Tutorial on Google Colab](https://img.shields.io/badge/Tutorial:_scikit--learn_interface-525252?style=flat&logo=googlecolab&logoColor=%23F9AB00)](https://colab.research.google.com/github/reservoirpy/reservoirpy/blob/master/tutorials/6-Interfacing_with_scikit-learn.ipynb)

### Examples

For advanced users, we also showcase partial reproduction of papers on reservoir computing to demonstrate some features of the library.

- [**Improving reservoir using Intrinsic Plasticity** (Schrauwen et al., 2008)](/examples/Improving%20reservoirs%20using%20Intrinsic%20Plasticity/Intrinsic_Plasiticity_Schrauwen_et_al_2008.ipynb)
- [**Interactive reservoir computing for chunking information streams** (Asabuki et al., 2018)](/examples/Interactive%20reservoir%20computing%20for%20chunking%20information%20streams/Chunking_Asabuki_et_al_2018.ipynb)
- [**Next-Generation reservoir computing** (Gauthier et al., 2021)](/examples/Next%20Generation%20Reservoir%20Computing/NG-RC_Gauthier_et_al_2021.ipynb)
- [**Edge of stability Echo State Network** (Ceni et al., 2023)](/examples/Edge%20of%20Stability%20Echo%20State%20Network/Edge_of_stability_Ceni_Gallicchio_2023.ipynb)


## 🏗️ Project Organization & Development Roadmap

ReservoirCogs development is organized through a comprehensive GitHub Project that orchestrates both short-term and long-term implementation of our feature portfolio.

### 🔧 Current Technical Features
Active development focusing on production-ready capabilities:

- **🕸️ GraphRAG Integration**: Knowledge graph-based retrieval-augmented generation
- **⚡ Codestral AI Engine**: Specialized language model for technical documentation  
- **🧠 AtomSpace Intelligence**: OpenCog symbolic AI reasoning capabilities
- **🔮 Hybrid AI Architecture**: Neural-symbolic fusion implementation

### 🚀 Future Development Roadmap
Research-driven features for long-term innovation:

- **🧬 P-Systems Membrane Computing** with P-lingua integration
- **🌳 B-Series Rooted Tree Gradient Descent** with Runge-Kutta methods ✅ *Research implementation available*
- **💎 J-Surface Julia Differential Equations** with DifferentialEquations.jl
- **💝 Differential Emotion Theory Framework** for affective computing ✅ *Research implementation available*

### 📋 Project Management

- **[📊 GitHub Project Board](https://github.com/orgs/HyperCogWizard/projects/1)**: Complete project tracking and coordination
- **[🗺️ Development Roadmap](.github/PROJECT_ROADMAP.md)**: Detailed timeline and milestones
- **[🏷️ Issue Templates](.github/ISSUE_TEMPLATE/)**: Structured feature requests and bug reports
- **[⚙️ Automation Workflows](.github/workflows/)**: Automated project management and CI/CD

Our project uses advanced GitHub Project features including custom fields, automated workflows, and multiple views (Board, Table, Timeline) to ensure efficient coordination of complex, interdisciplinary development spanning traditional software engineering and cutting-edge AI research.


## Papers and projects using ReservoirPy

*If you want your paper to appear here, please contact us (see contact link below).*

- ( [HAL](https://inria.hal.science/hal-04354303) | [PDF](https://arxiv.org/pdf/2312.06695) | [Code](https://github.com/corentinlger/ER-MRL) ) Leger et al. (2024) *Evolving Reservoirs for Meta Reinforcement Learning.* EvoAPPS 2024
- ( [arXiv](https://arxiv.org/abs/2204.02484) | [PDF](https://arxiv.org/pdf/2204.02484) ) Chaix-Eichel et al. (2022) *From implicit learning to explicit representations.* arXiv preprint arXiv:2204.02484.
- ( [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_6) | [HAL](https://hal.inria.fr/hal-03203374) | [PDF](https://hal.inria.fr/hal-03203374/document) ) Trouvain & Hinaut (2021) *Canary Song Decoder: Transduction and Implicit Segmentation with ESNs and LTSMs.* ICANN 2021
- ( [HTML](https://ieeexplore.ieee.org/abstract/document/9515607) ) Pagliarini et al. (2021) *Canary Vocal Sensorimotor Model with RNN Decoder and Low-dimensional GAN Generator.* ICDL 2021.
- ( [HAL](https://hal.inria.fr/hal-03244723/) | [PDF](https://hal.inria.fr/hal-03244723/document) ) Pagliarini et al. (2021) *What does the Canary Say? Low-Dimensional GAN Applied to Birdsong.* HAL preprint.
- ( [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) | [HAL](https://hal.inria.fr/hal-03203318) | [PDF](https://hal.inria.fr/hal-03203318) ) Hinaut & Trouvain (2021) *Which Hype for My New Task? Hints and Random Search for Echo State Networks Hyperparameters.* ICANN 2021

## Awesome Reservoir Computing

We also provide a curated list of tutorials, papers, projects and tools for Reservoir Computing (not necessarily related to ReservoirPy) here!:

**https://github.com/reservoirpy/awesome-reservoir-computing**

## Contact
If you have a question regarding the library, please open an issue.

If you have more general question or feedback you can contact us by email to **xavier dot hinaut the-famous-home-symbol inria dot fr**.

## Citing ReservoirPy

Trouvain, N., Pedrelli, L., Dinh, T. T., Hinaut, X. (2020) *ReservoirPy: an efficient and user-friendly library to design echo state networks. In International Conference on Artificial Neural Networks* (pp. 494-505). Springer, Cham. ( [HTML](https://link.springer.com/chapter/10.1007/978-3-030-61616-8_40) | [HAL](https://hal.inria.fr/hal-02595026) | [PDF](https://hal.inria.fr/hal-02595026/document) )

If you're using ReservoirPy in your work, please cite our package using the following bibtex entry:

```
@incollection{Trouvain2020,
  doi = {10.1007/978-3-030-61616-8_40},
  url = {https://doi.org/10.1007/978-3-030-61616-8_40},
  year = {2020},
  publisher = {Springer International Publishing},
  pages = {494--505},
  author = {Nathan Trouvain and Luca Pedrelli and Thanh Trung Dinh and Xavier Hinaut},
  title = {{ReservoirPy}: An Efficient and User-Friendly Library to Design Echo State Networks},
  booktitle = {Artificial Neural Networks and Machine Learning {\textendash} {ICANN} 2020}
}
```


## Acknowledgement

<div align="left">
  <img src="./static/inria_red.svg" width=300><br>
</div>


This package is developed and supported by Inria at Bordeaux, France in [Mnemosyne](https://team.inria.fr/mnemosyne/) group. [Inria](https://www.inria.fr/en) is a French Research Institute in Digital Sciences (Computer Science, Mathematics, Robotics, ...).
