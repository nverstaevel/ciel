# CIEL ☁️

## Overview 🌍

CIEL (Contextual Interactive Ensemble Learning) is a multiagent ensemble learning system designed for supervised learning tasks. It leverages multiple learning agents that collaborate to solve supervised learning tasks.

### Prerequisites

- **Python 3.x** installed on your machine
- **pip** (Python package installer)

## Installation 💾

To install the dependencies of the project:

```bash
pip install -r requirements.txt
```

To install the library:

```bash
pip install git+https://github.com/nverstaevel/ciel.git
```

## Repository 🗂️

The repository is organized as follows:

```
.
├── examples/       # Example notebooks demonstrating code recipes
│   └── <example_notebook>.ipynb        # Notebooks with usage examples and tutorials
│
└── torch_mas/      # Core implementation of the multi-agent algorithms
    ├── batch/      # Implementation of batch mode
    │   ├── activation_function/        # Implementations of various activation functions
    │   │   └── <activation>.py     # Code for specific activation functions
    │   │
    │   ├── internal_model/     # Implementations of internal models
    │   │   └── <model>.py      # Code for specific types of internal models
    │   │
    │   └── trainer/        # Implementation of various trainer
    │       ├── <trainer>.py        # Code for specific trainers
    │       └── learning_rules.py       # Definitions of learning rules for trainers
    │
    ├── common/     # Utilities shared between batch and sequential modes
    │   ├── models/     # Utilities for machine learning models
    │   │   └── <model_utilities>.py        # Code for model utility functions, layers, etc.
    │   │
    │   └── orthotopes/     # Utilities for orthotope (n-dimensional rectangle) manipulation
    │       └── <orthotope_utilities>.py        # Code for orthotope operations and utilities
    │
    └── sequential/     # Implementation of sequential mode
        ├── activation_function/        # Implementations of various activation functions
        │   └── <activation>.py     # Code for specific activation functions
        │
        ├── internal_model/     # Implementations of internal models
        │   └── <model>.py      # Code for specific types of internal models
        │
        └── trainer/        # Implementation of various trainer
            └── <trainer>.py        # Code for specific trainers
```

## Context Learning 🤖

### Context Agents 

_A context agent is an expert entity on the function to be approximated in a small area inside the input space._

A context agent has 2 core components:

- **Validity Area**: a context agent positions itself in the input variable space in the shape of an [https://en.wikipedia.org/wiki/Hyperrectangle](orthotope) → To know **when** to predict.
- **Internal Model**: a context agent has an internal model in the form of a simple machine learning model (_linear regression_, _svm_, ...) so it can map the input space to the output space → To know **what** to predict.

<p align="center"><image src="images/context_agent_structure.png"></p>

### Learning 

A learning step follows the 5 following steps:

- Define neighborhood of X (input data point)
- Select neighboring agents
- Selected agents suggest predictions
- Error on propositions is calculated
- Agents are updated according to errors (feedbacks)

<p align="center"><image src="images/learning_with_context_agents.gif"></p>

## Usage 🧑‍💻

CIEL library features 2 learning modes:

- **Sequential**: data are fed sequentially one by one to the model during training
- **Batch**: data points are fed in batches to the model during training

Here is a simple code snippet to run batch learning:

```python
import time
from torch_mas.sequential.trainer import BaseTrainer as Trainer
from torch_mas.sequential.internal_model import LinearWithMemory
from torch_mas.sequential.activation_function import BaseActivation

...

dataset = DataBuffer(X, y, device=device)

validity = BaseActivation(
    dataset.input_dim, 
    dataset.output_dim, 
    alpha=0.1, 
)

internal_model = LinearWithMemory(
    dataset.input_dim, 
    dataset.output_dim, 
    l1=0.1, 
    memory_length=10, 
)

model = Trainer(
    validity,
    internal_model,
    R=0.5,
    imprecise_th=0.01,
    bad_th=0.1,
    n_epochs=5,
)

t = time.time()
model.fit(dataset)
tt = time.time() - t
print(f"Total training time: {tt}s")

print("Number of agents created:", model.n_agents)
```

Complete examples of learning (regression and classification) are available in [examples/](https://github.com/nverstaevel/ciel/tree/main/examples).

## TODO Works 📝

- [x] GPU Batch Training
- [x] GPU Sequential Training
- [x] Multiclass classification
- [ ] Refine destruction of agents
- [ ] Explainability Metrics
- [ ] Compute SHAPley and LIME values
- [ ] Benchmark performances on higher dimensional problems


## References 📚

- _Boes, Jérémy, Julien Nigon, Nicolas Verstaevel, Marie-Pierre Gleizes, and Frédéric Migeon. 2015. “The Self-Adaptive Context Learning Pattern: Overview and Proposal.” In SpringerLink, 91–104. Cham, Switzerland: Springer. https://doi.org/10.1007/978-3-319-25591-0_7._
- _Verstaevel, Nicolas, Jérémy Boes, Julien Nigon, Dorian D’Amico, and Marie-Pierre Gleizes. 2017. “Lifelong Machine Learning with Adaptive Multi-Agent Systems” 1 (February):275–86. https://doi.org/10.5220/0006247302750286._
- _Fourez, Thibault, Nicolas Verstaevel, Frédéric Migeon, Frédéric Schettini, and Frederic Amblard. 2022. “An Ensemble Multi-Agent System for Non-Linear Classification.” ArXiv E-Prints, September. https://doi.org/10.48550/arXiv.2209.06824._
