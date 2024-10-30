# CIEL ☁️

## Overview 🌍

CIEL (Contextual Interactive Ensemble Learning) is a multiagent ensemble learning system designed for supervised learning tasks. It leverages multiple learning agents that collaborate to solve supervised learning tasks.

### Prerequisites 🔑

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

## Repository

## Context Learning

### Context Agents 🤖

_A context agent is an expert entity on the function to be approximated in a small area inside the input space._

A context agent has 2 core components:

- **Validity Area**: a context agent positions itself in the input variable space in the shape of an [https://en.wikipedia.org/wiki/Hyperrectangle](orthotope) → To know **when** to predict.
- **Internal Model**: a context agent has an internal model in the form of a simple machine learning model (_linear regression_, _svm_, ...) so it can map the input space to the output space → To know **what** to predict.

<p align="center"><image src="images/context_agent_structure.png"></p>

### Learning 🎓

A learning step follows the 5 following steps:

- Define neighborhood of X (input data point)
- Select neighboring agents
- Selected agents suggest predictions
- Error on propositions is calculated
- Agents are updated according to errors (feedbacks)

<p align="center"><image src="images/learning_with_context_agents.gif"></p>

## Usage

CIEL library features 2 learning modes:

- **Sequential**: data are fed sequentially one by one to the model during training
- **Batch**: data points are fed in batches to the model during training

Here is a simple code snippet to run batch learning:

```python
import time
from torch_mas.batch_head import BatchHead
from torch_mas.agents.batch_agents_linear_reg import BatchLinearAgent

...

dataset = DataBuffer(X, y, device=device)

model = BatchHead(
    dataset.input_dim,
    dataset.output_dim,
    R=[0.5, 0.4],
    imprecise_th=0.01,
    bad_th=0.1,
    n_epochs=20,
    batch_size=256,
    agents=BatchLinearAgent,
    agents_kwargs={
        "l1": 0.1,
        "alpha": 0.3,
        "memory_length": 10
    },
    device=device
)

t = time.time()
model.fit(dataset)
tt = time.time() - t
print(f"Total training time: {tt}s")

print("Number of agents created:", model.agents.n_agents)
```

A complete learning example is available in the following notebook: `examples/batch_simple_learning.ipynb` (`examples/simple_learning.ipynb` for sequential learning).

## TODO Works

- [x] GPU Batch Training
- [ ] GPU Sequential Training
- [ ] Explainability Metrics
- [ ] Compute SHAPley and LIME values
- [ ] Refine destruction of agents
- [ ] Benchmark performances on higher dimensional problems
- [ ] Multiclass classification

## References
