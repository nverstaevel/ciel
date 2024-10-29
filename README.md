# CIEL ☁️

## Overview 🌍

CIEL (Contextual Interactive Ensemble Learning) is a multiagent ensemble learning system designed for supervised learning tasks. It leverages multiple learning agents that collaborate to solve supervised learning tasks.

### Prerequisites 🔑

- **Python 3.x** installed on your machine
- **pip** (Python package installer)

## Installation 💾

Follow the instructions below to install and set up the project:

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

## Future Works

## References
