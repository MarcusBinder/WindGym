---
sidebar_position: 1
title: WindGym Package
---

# WindGym Package

*This documentation is auto-generated from Python docstrings using Sphinx.*

## Main Environment Classes

### WindFarmEnv

The base wind farm environment class implementing the Gymnasium interface.

```python
from WindGym import WindFarmEnv
```

**Parameters:**

- `turbine` - PyWake turbine model (required)
- `x_pos` - Turbine x positions in meters (required)
- `y_pos` - Turbine y positions in meters (required)
- `n_passthrough` (int) - Number of flow passthroughs (default: 5)
- `dt_sim` (float) - Simulation timestep in seconds (default: 1)
- `dt_env` (float) - Environment timestep in seconds (default: 1)
- `config` (str | Path | dict) - Environment configuration
- `backend` (str) - Simulation backend: "dynamiks" or "pywake" (default: "dynamiks")
- `seed` (int) - Random seed for reproducibility
- `render_mode` (str) - Render mode: None, "human", or "rgb_array"

### WindFarmEnvMulti

Multi-agent wind farm environment compatible with PettingZoo.

```python
from WindGym import WindFarmEnvMulti
```

### FarmEval

Evaluation wrapper for detailed performance tracking.

```python
from WindGym import FarmEval
```

### AgentEval

Agent evaluation utility class.

```python
from WindGym import AgentEval
```

---

*Run `./build-api-docs.sh` to regenerate this documentation from source.*
