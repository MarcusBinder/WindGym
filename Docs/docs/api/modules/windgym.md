---
sidebar_position: 1
title: WindGym
sidebar_label: WindGym
---

# WindGym

Main package containing the core environment classes.

```python
from WindGym import WindFarmEnv, WindFarmEnvMulti, FarmEval, AgentEval
```

---

## WindFarmEnv

The base single-agent wind farm environment implementing the Gymnasium interface.

### Import

```python
from WindGym import WindFarmEnv
```

### Constructor

```python
env = WindFarmEnv(
    turbine,              # PyWake turbine model (required)
    x_pos,                # Turbine x positions in meters (required)
    y_pos,                # Turbine y positions in meters (required)
    config=None,          # Path to YAML config or dict
    n_passthrough=5,      # Number of flow passthroughs
    dt_sim=1,             # Simulation timestep (seconds)
    dt_env=1,             # Environment timestep (seconds)
    backend="dynamiks",   # "dynamiks" or "pywake"
    seed=None,            # Random seed
    render_mode=None,     # None, "human", or "rgb_array"
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `turbine` | PyWake turbine | *required* | Wind turbine model (e.g., `V80()`) |
| `x_pos` | list | *required* | Turbine x positions in meters |
| `y_pos` | list | *required* | Turbine y positions in meters |
| `config` | str, Path, dict | `None` | Environment configuration |
| `n_passthrough` | int | `5` | Number of flow passthroughs |
| `dt_sim` | float | `1` | Simulation timestep in seconds |
| `dt_env` | float | `1` | Environment timestep in seconds |
| `backend` | str | `"dynamiks"` | Simulation backend |
| `seed` | int | `None` | Random seed for reproducibility |
| `render_mode` | str | `None` | Rendering mode |
| `yaw_step_sim` | float | `1` | Max yaw change per sim step (degrees) |
| `Baseline_comp` | bool | `False` | Enable baseline comparison |
| `sample_site` | PyWake Site | `None` | Site for wind sampling |

### Methods

| Method | Returns | Description |
|:-------|:--------|:------------|
| `reset(seed=None)` | `(obs, info)` | Reset environment, return initial observation |
| `step(action)` | `(obs, reward, terminated, truncated, info)` | Execute action |
| `render()` | array or None | Visualize the environment |
| `close()` | None | Clean up resources |

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `observation_space` | `gym.Space` | Observation space specification |
| `action_space` | `gym.Space` | Action space specification |
| `n_wt` | int | Number of turbines |
| `dt_env` | float | Environment timestep |

### Example

```python
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    n_passthrough=5,
    dt_env=1,
)

obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## WindFarmEnvMulti

Multi-agent wind farm environment compatible with PettingZoo.

### Import

```python
from WindGym import WindFarmEnvMulti
```

### Constructor

```python
env = WindFarmEnvMulti(
    turbine,              # PyWake turbine model
    x_pos,                # Turbine x positions
    y_pos,                # Turbine y positions
    # ... same parameters as WindFarmEnv
)
```

### Key Differences from WindFarmEnv

- Returns dict of observations (one per agent/turbine)
- Requires dict of actions (one per agent/turbine)
- Compatible with PettingZoo parallel API

### Example

```python
from WindGym import WindFarmEnvMulti
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnvMulti(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
)

observations, infos = env.reset()

# Each agent gets its own observation and action
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

---

## FarmEval

Evaluation wrapper providing detailed performance tracking and analysis.

### Import

```python
from WindGym import FarmEval
```

### Constructor

```python
env = FarmEval(
    turbine,
    x_pos,
    y_pos,
    Baseline_comp=True,   # Enable baseline comparison
    # ... other WindFarmEnv parameters
)
```

### Additional Methods

| Method | Returns | Description |
|:-------|:--------|:------------|
| `get_results()` | `xarray.Dataset` | Episode results as xarray |
| `plot_flow_field(time_idx=-1)` | Figure | Visualize flow field |
| `get_power_time_series()` | array | Power production over time |

---

## AgentEval

Utility class for evaluating agent performance.

### Import

```python
from WindGym import AgentEval
```

### Usage

```python
from WindGym import AgentEval

results = AgentEval(
    env=env,
    agent=agent,
    n_episodes=10,
)
```

---

## AgentEvalFast

Fast evaluation function for quick agent assessment.

### Import

```python
from WindGym import AgentEvalFast
```

### Usage

```python
from WindGym import AgentEvalFast

result = AgentEvalFast(env, agent, n_steps=1000)
```
