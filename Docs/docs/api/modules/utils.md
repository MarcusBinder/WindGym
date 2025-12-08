---
sidebar_position: 6
title: WindGym.utils Module
---

# WindGym.utils Module

*This documentation is auto-generated from Python docstrings using Sphinx.*

This module contains utility functions for WindGym.

## evaluate_PPO

PPO evaluation utilities including the Coliseum multi-agent evaluation framework.

```python
from WindGym.utils.evaluate_PPO import Coliseum

coliseum = Coliseum(
    env_factory=create_env,
    agents=agent_dict
)
```

## generate_layouts

Wind farm layout generation utilities.

```python
from WindGym.utils import generate_layouts

# Generate grid layout
x_pos, y_pos = generate_layouts.grid_layout(
    n_rows=2,
    n_cols=3,
    spacing_x=500,
    spacing_y=500
)

# Generate circular layout
x_pos, y_pos = generate_layouts.circular_layout(
    n_turbines=6,
    radius=1000
)
```

---

*Run `./build-api-docs.sh` to regenerate this documentation from source.*
