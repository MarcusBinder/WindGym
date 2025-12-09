---
sidebar_position: 3
title: WindGym.Agents
sidebar_label: Agents
---

# WindGym.Agents

Built-in agent implementations for baseline comparisons and custom agent development.

```python
from WindGym.Agents import BaseAgent, PyWakeAgent, GreedyAgent, RandomAgent, ConstantAgent
```

---

## BaseAgent

Abstract base class for all agents. Extend this class to create custom agents.

### Import

```python
from WindGym.Agents import BaseAgent
```

### Creating a Custom Agent

```python
from WindGym.Agents import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def predict(self, obs):
        """Generate action from observation.

        Args:
            obs: Current observation array

        Returns:
            action: Action array
            state: Optional agent state (can be None)
        """
        # Your control logic here
        action = np.zeros(self.n_wt)
        return action, None
```

### Helper Methods

| Method | Description |
|:-------|:------------|
| `scale_yaw(yaw_angles)` | Scale yaw angles to [-1, 1] action space |
| `unscale_yaw(scaled_actions)` | Convert scaled actions back to degrees |

---

## PyWakeAgent

Optimal static yaw control using PyWake's wake steering optimization.

### Import

```python
from WindGym.Agents import PyWakeAgent
```

### Constructor

```python
agent = PyWakeAgent(
    x_pos,                # Turbine x positions (required)
    y_pos,                # Turbine y positions (required)
    turbine=V80(),        # PyWake turbine model
    wind_speed=8,         # Default wind speed for optimization
    wind_dir=270,         # Default wind direction
    TI=0.07,              # Turbulence intensity
    yaw_max=45,           # Maximum yaw angle
    yaw_min=-45,          # Minimum yaw angle
    env=None,             # Optional environment reference
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `x_pos` | list | *required* | Turbine x positions |
| `y_pos` | list | *required* | Turbine y positions |
| `turbine` | PyWake turbine | `V80()` | Turbine model |
| `wind_speed` | float | `8` | Wind speed for optimization |
| `wind_dir` | float | `270` | Wind direction |
| `TI` | float | `0.07` | Turbulence intensity |
| `yaw_max` | float | `45` | Maximum yaw angle (degrees) |
| `yaw_min` | float | `-45` | Minimum yaw angle (degrees) |

### Example

```python
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80

agent = PyWakeAgent(
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    turbine=V80(),
)

# Use with environment
obs, info = env.reset()
action, _ = agent.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)
```

### Behavior

- Computes optimal yaw angles based on wind conditions
- Uses PyWake's internal optimization routines
- Static optimization (does not adapt during episode)

---

## NoisyPyWakeAgent

Robust variant of PyWakeAgent designed for noisy observations.

### Import

```python
from WindGym.Agents import NoisyPyWakeAgent
```

### Constructor

```python
agent = NoisyPyWakeAgent(
    x_pos,
    y_pos,
    turbine=V80(),
    # ... same parameters as PyWakeAgent
)
```

### Behavior

- Averages multiple wind measurements to estimate true conditions
- More robust to measurement noise than standard PyWakeAgent

---

## GreedyAgent

Simple reactive agent that aligns turbines with the current wind direction.

### Import

```python
from WindGym.Agents import GreedyAgent
```

### Constructor

```python
agent = GreedyAgent(
    env,                  # Environment reference (required)
    use_global_wind=True, # Use global vs local wind direction
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `env` | WindFarmEnv | *required* | Environment reference |
| `use_global_wind` | bool | `True` | Use global wind direction |

### Example

```python
from WindGym.Agents import GreedyAgent

agent = GreedyAgent(env, use_global_wind=True)

obs, info = env.reset()
action, _ = agent.predict(obs)
```

---

## RandomAgent

Takes random actions within the action space. Useful as a baseline.

### Import

```python
from WindGym.Agents import RandomAgent
```

### Constructor

```python
agent = RandomAgent(env)
```

### Example

```python
from WindGym.Agents import RandomAgent

agent = RandomAgent(env)

obs, info = env.reset()
action, _ = agent.predict(obs)  # Random action
```

---

## ConstantAgent

Maintains fixed yaw angles throughout the episode.

### Import

```python
from WindGym.Agents import ConstantAgent
```

### Constructor

```python
agent = ConstantAgent(
    env,                  # Environment reference
    yaw_angles=None,      # Fixed yaw angles in degrees
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `env` | WindFarmEnv | *required* | Environment reference |
| `yaw_angles` | array | `None` | Fixed yaw angles (degrees). If None, uses zeros |

### Example

```python
from WindGym.Agents import ConstantAgent
import numpy as np

# Set all turbines to 0 degrees yaw
agent = ConstantAgent(env, yaw_angles=np.array([0.0, 0.0, 0.0]))

# Or with specific angles
agent = ConstantAgent(env, yaw_angles=np.array([5.0, 0.0, -5.0]))
```
