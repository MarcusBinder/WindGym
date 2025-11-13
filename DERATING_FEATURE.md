# Derating Feature in WindGym

## Overview

This document describes the new derating feature that allows wind turbines to reduce their power output (derate) as part of the control strategy. Derating can be used alone or in combination with yaw control for more sophisticated wind farm optimization.

## What is Derating?

Derating (or power curtailment) is the intentional reduction of a wind turbine's power output below its maximum capacity. This is typically done by:
- Adjusting the blade pitch angle
- Modifying the generator torque
- Other control mechanisms

In this implementation, derating is represented as a **power fraction** where:
- `1.0` = Full power (no derating)
- `0.5` = 50% power
- `0.0` = No power (complete shutdown)

## Configuration

### Basic Configuration

Add the following parameters to your environment configuration YAML file:

```yaml
# Action type configuration
enable_yaw: True          # Enable yaw control (default: True)
enable_derate: False      # Enable derating control (default: False)
DerateMethod: "wind"      # Derating method: "yaw" (incremental) or "wind" (setpoint)

# Farm parameters
farm:
  yaw_min: -45
  yaw_max: 45
  derate_min: 0.0       # Minimum derating (0.0 = no power)
  derate_max: 1.0       # Maximum derating (1.0 = full power)
  # ... other farm parameters
```

### Action Modes

You can configure three different action modes:

#### 1. Yaw Only (Default/Backward Compatible)
```yaml
enable_yaw: True
enable_derate: False
```
- Action space: `[yaw_t1, yaw_t2, ..., yaw_tn]`
- Same behavior as before the derating feature was added

#### 2. Derate Only
```yaml
enable_yaw: False
enable_derate: True
```
- Action space: `[derate_t1, derate_t2, ..., derate_tn]`
- Only control power output, yaw angles remain fixed

#### 3. Both Yaw and Derate
```yaml
enable_yaw: True
enable_derate: True
```
- Action space: `[yaw_t1, yaw_t2, ..., yaw_tn, derate_t1, derate_t2, ..., derate_tn]`
- Full control over both yaw and power output
- **Action Structure**: Yaw actions for all turbines, then derating actions for all turbines (concatenated)

### Control Methods

Both `ActionMethod` (for yaw) and `DerateMethod` (for derating) support two control approaches:

#### Setpoint Control (`"wind"`)
- **Mapping**: Action values `[-1, 1]` map directly to the target range
- For yaw: `-1` → `yaw_min`, `+1` → `yaw_max`, `0` → `0°`
- For derating: `-1` → `derate_min` (minimum power), `+1` → `derate_max` (maximum power), `0` → `0.5` (50% power)
- **Rate limiting**: Changes per step are limited by `yaw_step_sim`/`derate_step_sim`
- **Recommended** for most applications

#### Incremental Control (`"yaw"`)
- **Mapping**: Action values represent changes relative to current state
- `0` → No change
- `+1` → Increase by maximum step
- `-1` → Decrease by maximum step
- **Budget system**: Total change is distributed across simulation sub-steps

## Action Space

The action space shape depends on the enabled actions:

| Configuration | Action Space Shape | Example (2 turbines) |
|---------------|-------------------|----------------------|
| Yaw only | `(n_turbines,)` | `(2,)` |
| Derate only | `(n_turbines,)` | `(2,)` |
| Both | `(2 * n_turbines,)` | `(4,)` |

## Example Usage

### Example 1: Yaw + Derate with Setpoint Control

```python
import numpy as np
from py_wake.examples.data.iea37 import IEA37_WindTurbines
from WindGym import WindFarmEnv

# Create configuration
config = {
    "enable_yaw": True,
    "enable_derate": True,
    "DerateMethod": "wind",
    "ActionMethod": "wind",
    "farm": {
        "yaw_min": -30,
        "yaw_max": 30,
        "derate_min": 0.0,
        "derate_max": 1.0,
    },
    # ... other config parameters
}

# Create environment
env = WindFarmEnv(
    turbine=IEA37_WindTurbines(),
    x_pos=np.array([0, 1000]),
    y_pos=np.array([0, 0]),
    config=config,
    backend="pywake",
)

# Reset
obs, info = env.reset()

# Create action: [yaw_t1, yaw_t2, derate_t1, derate_t2]
action = np.array([
    0.5,   # Turbine 1: yaw to +15° (halfway between 0 and max)
    -0.5,  # Turbine 2: yaw to -15°
    1.0,   # Turbine 1: full power (100%)
    0.0,   # Turbine 2: 50% power
])

# Take step
obs, reward, terminated, truncated, info = env.step(action)
```

### Example 2: Derate Only

```python
config = {
    "enable_yaw": False,
    "enable_derate": True,
    "DerateMethod": "wind",
    "farm": {
        "derate_min": 0.5,  # Minimum 50% power
        "derate_max": 1.0,  # Maximum 100% power
    },
    # ... other config
}

env = WindFarmEnv(...)
obs, info = env.reset()

# Action: [derate_t1, derate_t2]
action = np.array([1.0, 0.0])  # T1: 100%, T2: 75%
obs, reward, terminated, truncated, info = env.step(action)
```

## Implementation Details

### Backend Support

The derating feature is supported on both backends:

- **PyWake Backend**: Derating is applied as a multiplier to the power output after wake calculations
- **Dynamiks Backend**: The power method is wrapped to apply derating transparently

### Power Calculation

Power output is calculated as:
```
actual_power = base_power * derate
```

Where:
- `base_power`: Power from the turbine's power curve and wake model
- `derate`: Current derating factor (0.0 to 1.0)

### Baseline Comparison

When baseline comparison is enabled, the baseline farm always operates at full power (derate = 1.0) to provide a fair reference.

## Use Cases

1. **Load Balancing**: Reduce power from upstream turbines to increase total farm output
2. **Grid Services**: Provide frequency regulation or reserve capacity
3. **Fatigue Reduction**: Reduce loads on specific turbines
4. **Optimization Research**: Study combined yaw and power control strategies
5. **Reinforcement Learning**: Expand action space for more sophisticated control policies

## Related Configuration Parameters

```yaml
# Optional: Custom step sizes (if not specified, derived from yaw settings)
derate_step_sim: 0.1   # Max change per simulation step (optional)
derate_step_env: 0.5   # Max change per environment step (optional)
```

## Backward Compatibility

The feature is **fully backward compatible**:
- Default configuration: `enable_yaw=True`, `enable_derate=False`
- Existing configurations without these parameters will work as before
- Action space shape remains `(n_turbines,)` when only yaw is enabled

## References

This implementation is inspired by:
- PyWake optimization examples: [Optimization with TOPFARM](https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/Optimization.html)
- Research on combined wake steering and derating strategies
- Wind farm control literature on power curtailment

## Testing

A test script is provided at `test_derating.py` that verifies:
1. Yaw-only mode (backward compatibility)
2. Derate-only mode
3. Combined yaw and derate mode

Run tests with:
```bash
python test_derating.py
```
