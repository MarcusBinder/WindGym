---
sidebar_position: 6
title: WindGym.utils
sidebar_label: Utils
---

# WindGym.utils

Utility functions for evaluation and layout generation.

---

## Coliseum

Multi-agent evaluation framework for comparing agent performance.

### Import

```python
from WindGym.utils.evaluate_PPO import Coliseum
```

### Constructor

```python
coliseum = Coliseum(
    env_factory,          # Function that returns a new environment
    agents,               # Dictionary of {name: agent}
)
```

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `env_factory` | callable | Function returning new environment instance |
| `agents` | dict | Dictionary mapping agent names to agent objects |

### Methods

| Method | Returns | Description |
|:-------|:--------|:------------|
| `run_time_series_evaluation(n_episodes)` | DataFrame | Stochastic evaluation |
| `run_wind_grid_evaluation(ws, wd, TI)` | xarray.Dataset | Grid evaluation |
| `plot_time_series_rewards(results)` | Figure | Plot time series |
| `plot_summary_bar_chart(results)` | Figure | Compare agents |

### Example

```python
from WindGym.utils.evaluate_PPO import Coliseum

def create_env():
    return WindFarmEnv(...)

agents = {
    "PyWake": pywake_agent,
    "Greedy": greedy_agent,
    "Random": random_agent,
}

coliseum = Coliseum(
    env_factory=create_env,
    agents=agents,
)

# Run stochastic evaluation
results = coliseum.run_time_series_evaluation(n_episodes=100)

# Run grid evaluation
grid_results = coliseum.run_wind_grid_evaluation(
    wind_speeds=[6, 8, 10, 12],
    wind_directions=[240, 270, 300],
    turbulence_intensities=[0.06, 0.08],
)

# Visualize results
coliseum.plot_summary_bar_chart(results, metric='mean_reward')
```

---

## generate_layouts

Functions for generating wind farm turbine layouts.

### Import

```python
from WindGym.utils import generate_layouts
```

### grid_layout

Generate a rectangular grid layout.

```python
x_pos, y_pos = generate_layouts.grid_layout(
    n_rows=2,
    n_cols=3,
    spacing_x=500,    # meters
    spacing_y=500,    # meters
)
```

#### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `n_rows` | int | Number of rows |
| `n_cols` | int | Number of columns |
| `spacing_x` | float | Spacing between columns (meters) |
| `spacing_y` | float | Spacing between rows (meters) |

#### Returns

| Return | Type | Description |
|:-------|:-----|:------------|
| `x_pos` | list | X coordinates of turbines |
| `y_pos` | list | Y coordinates of turbines |

### circular_layout

Generate a circular layout.

```python
x_pos, y_pos = generate_layouts.circular_layout(
    n_turbines=6,
    radius=1000,      # meters
)
```

#### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `n_turbines` | int | Number of turbines |
| `radius` | float | Circle radius (meters) |

### Example

```python
from WindGym.utils import generate_layouts
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

# Create a 3x2 grid layout
x_pos, y_pos = generate_layouts.grid_layout(
    n_rows=2,
    n_cols=3,
    spacing_x=500,
    spacing_y=500,
)

# Use with environment
env = WindFarmEnv(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
)
```
