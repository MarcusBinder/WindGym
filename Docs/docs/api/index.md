---
sidebar_position: 1
title: API Reference
description: Complete API documentation for the WindGym package
---

# API Reference

Complete API documentation for the WindGym reinforcement learning environment.

---

## Environments

The core environment classes for wind farm control.

| Class | Module | Description |
|:------|:-------|:------------|
| [`WindFarmEnv`](./modules/windgym#windfarmenv) | `WindGym` | Single-agent wind farm environment |
| [`WindFarmEnvMulti`](./modules/windgym#windfarmenvmulti) | `WindGym` | Multi-agent environment (PettingZoo) |
| [`FarmEval`](./modules/windgym#farmeval) | `WindGym` | Evaluation wrapper with tracking |
| [`AgentEval`](./modules/windgym#agenteval) | `WindGym` | Agent evaluation utility |

---

## Agents

Built-in agent implementations for baseline comparisons.

| Class | Module | Description |
|:------|:-------|:------------|
| [`BaseAgent`](./modules/agents#baseagent) | `WindGym.Agents` | Abstract base class |
| [`PyWakeAgent`](./modules/agents#pywakeagent) | `WindGym.Agents` | PyWake optimization baseline |
| [`GreedyAgent`](./modules/agents#greedyagent) | `WindGym.Agents` | Greedy wind-following strategy |
| [`RandomAgent`](./modules/agents#randomagent) | `WindGym.Agents` | Random action baseline |
| [`ConstantAgent`](./modules/agents#constantagent) | `WindGym.Agents` | Fixed yaw angles |

---

## Wrappers

Gymnasium wrappers to extend environment functionality.

| Class | Module | Description |
|:------|:-------|:------------|
| [`RecordEpisodeVals`](./modules/wrappers#recordepisodevals) | `WindGym.wrappers` | Episode statistics recording |
| [`CurriculumWrapper`](./modules/wrappers#curriculumwrapper) | `WindGym.wrappers` | Curriculum learning |
| [`PowerWrapper`](./modules/wrappers#powerwrapper) | `WindGym.wrappers` | Power normalization |
| [`AdversaryWrapper`](./modules/wrappers#adversarywrapper) | `WindGym.wrappers` | Adversarial training |

---

## Core Components

Internal components for environment customization.

| Class | Module | Description |
|:------|:-------|:------------|
| [`RewardCalculator`](./modules/core#reward-calculator) | `WindGym.core` | Reward computation |
| [`WindManager`](./modules/core#wind-manager) | `WindGym.core` | Wind condition management |
| [`TurbulenceManager`](./modules/core#turbulence-manager) | `WindGym.core` | Turbulence modeling |
| [`MeasurementManager`](./modules/core#measurement-manager) | `WindGym.core` | Noise and measurements |

---

## Utilities

Helper functions and evaluation tools.

| Function/Class | Module | Description |
|:---------------|:-------|:------------|
| [`Coliseum`](./modules/utils#evaluate_ppo) | `WindGym.utils` | Multi-agent evaluation framework |
| [`grid_layout`](./modules/utils#generate_layouts) | `WindGym.utils` | Generate grid turbine layouts |
| [`circular_layout`](./modules/utils#generate_layouts) | `WindGym.utils` | Generate circular layouts |

---

## Quick Start

```python
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

# Create environment
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="path/to/config.yaml",
)

# Standard Gymnasium interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Module Index

- [**WindGym**](./modules/windgym) - Main environment classes
- [**WindGym.core**](./modules/core) - Core components
- [**WindGym.Agents**](./modules/agents) - Agent implementations
- [**WindGym.wrappers**](./modules/wrappers) - Gymnasium wrappers
- [**WindGym.visualization**](./modules/visualization) - Plotting utilities
- [**WindGym.utils**](./modules/utils) - Helper functions
