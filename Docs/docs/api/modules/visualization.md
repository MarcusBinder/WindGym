---
sidebar_position: 5
title: WindGym.visualization
sidebar_label: Visualization
---

# WindGym.visualization

Plotting and visualization utilities for wind farm data.

```python
from WindGym.visualization import farm_plots, turbine_plots, plot_utils
```

---

## farm_plots

Functions for farm-level visualization.

### Import

```python
from WindGym.visualization import farm_plots
```

### Available Functions

| Function | Description |
|:---------|:------------|
| `plot_farm_layout(x_pos, y_pos)` | Plot turbine positions |
| `plot_power_time_series(results)` | Plot farm power over time |
| `plot_wind_field(env)` | Visualize wind flow field |

### Example

```python
from WindGym.visualization import farm_plots

# Plot turbine layout
farm_plots.plot_farm_layout(
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
)

# Plot power production over time
farm_plots.plot_power_time_series(episode_results)
```

---

## turbine_plots

Functions for turbine-level visualization.

### Import

```python
from WindGym.visualization import turbine_plots
```

### Available Functions

| Function | Description |
|:---------|:------------|
| `plot_turbine_power(turbine_id, results)` | Plot individual turbine power |
| `plot_yaw_angles(results)` | Plot yaw angle trajectories |
| `plot_turbine_comparison(results)` | Compare turbine performance |

### Example

```python
from WindGym.visualization import turbine_plots

# Plot yaw angles over time
turbine_plots.plot_yaw_angles(episode_results)

# Plot individual turbine power
turbine_plots.plot_turbine_power(turbine_id=0, results=episode_results)
```

---

## plot_utils

Shared plotting utilities and styling.

### Import

```python
from WindGym.visualization import plot_utils
```

### Available Functions

| Function | Description |
|:---------|:------------|
| `set_style()` | Set consistent plot styling |
| `save_figure(fig, path)` | Save figure with proper settings |
| `create_subplot_grid(n_plots)` | Create subplot layout |

### Example

```python
from WindGym.visualization import plot_utils
import matplotlib.pyplot as plt

# Apply consistent styling
plot_utils.set_style()

# Create figure and save
fig, ax = plt.subplots()
# ... plotting code ...
plot_utils.save_figure(fig, "output.png")
```

---

## Integration with FarmEval

The visualization module integrates with `FarmEval` for easy plotting:

```python
from WindGym import FarmEval
from WindGym.visualization import farm_plots

# Create evaluation environment
env = FarmEval(...)

# Run episode
obs, info = env.reset()
for _ in range(100):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

# Get results and visualize
results = env.get_results()
farm_plots.plot_power_time_series(results)
```
