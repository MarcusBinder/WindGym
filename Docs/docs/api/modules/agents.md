---
sidebar_position: 3
title: WindGym.Agents Module
---

# WindGym.Agents Module

*This documentation is auto-generated from Python docstrings using Sphinx.*

This module contains the agent implementations for WindGym.

## BaseAgent

Abstract base class for all agents.

```python
from WindGym.Agents import BaseAgent

class MyAgent(BaseAgent):
    def predict(self, obs):
        # Your control logic here
        action = ...
        return action, None
```

**Methods:**
- `predict(obs)` - Generate action from observation
- `scale_yaw(yaw_angles)` - Scale yaw angles to [-1, 1]
- `unscale_yaw(scaled_actions)` - Convert scaled actions back to degrees

## PyWakeAgent

Optimal static yaw control using PyWake optimization.

```python
from WindGym.Agents import PyWakeAgent

agent = PyWakeAgent(
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    turbine=V80(),
)
```

## NoisyPyWakeAgent

Robust variant of PyWakeAgent for noisy observations.

```python
from WindGym.Agents import NoisyPyWakeAgent
```

## GreedyAgent

Simple reactive agent that aligns turbines with wind.

```python
from WindGym.Agents import GreedyAgent

agent = GreedyAgent(env, use_global_wind=True)
```

## RandomAgent

Takes random actions within action space.

```python
from WindGym.Agents import RandomAgent

agent = RandomAgent(env)
```

## ConstantAgent

Maintains fixed yaw angles.

```python
from WindGym.Agents import ConstantAgent

agent = ConstantAgent(env, yaw_angles=np.array([0.0, 5.0, -5.0]))
```

---

*Run `./build-api-docs.sh` to regenerate this documentation from source.*
