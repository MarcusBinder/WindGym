# WindGym Configuration System

This guide explains the new configuration schema and validation system introduced to WindGym.

## Overview

WindGym now uses **strongly-typed configuration** with automatic validation using Python dataclasses. This provides:

- ✅ **Early error detection** - Configuration errors are caught at load time, not during execution
- ✅ **Clear error messages** - Validation errors explain exactly what's wrong
- ✅ **Type safety** - IDEs can provide autocomplete and type checking
- ✅ **Documentation** - Config structure is self-documenting via dataclass fields
- ✅ **Backward compatibility** - All existing YAML configs continue to work

## Quick Start

### Loading a Configuration

```python
from WindGym import WindFarmEnv
from py_wake.wind_turbines import V80

# Option 1: Load from YAML file (recommended)
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500],
    y_pos=[0, 0],
    config="path/to/config.yaml"
)

# Option 2: Load from dict
config_dict = {
    "farm": {"yaw_min": -45, "yaw_max": 45},
    "wind": {
        "ws_min": 7, "ws_max": 15,
        "TI_min": 0.02, "TI_max": 0.15,
        "wd_min": 255, "wd_max": 285
    },
    # ... rest of config
}
env = WindFarmEnv(turbine=V80(), x_pos=[0, 500], y_pos=[0, 0], config=config_dict)

# Option 3: Use EnvConfig object directly (for advanced usage)
from WindGym.config_schema import EnvConfig
env_config = EnvConfig.from_dict(config_dict)
env = WindFarmEnv(turbine=V80(), x_pos=[0, 500], y_pos=[0, 0], config=env_config)
```

## Configuration Structure

### Required Sections

Every configuration must include these sections:

#### 1. Farm Configuration (`farm`)
```yaml
farm:
  yaw_min: -45  # Minimum yaw angle (degrees)
  yaw_max: 45   # Maximum yaw angle (degrees)
```

#### 2. Wind Configuration (`wind`)
```yaml
wind:
  ws_min: 7     # Minimum wind speed (m/s)
  ws_max: 15    # Maximum wind speed (m/s)
  TI_min: 0.02  # Minimum turbulence intensity (0-1)
  TI_max: 0.15  # Maximum turbulence intensity (0-1)
  wd_min: 255   # Minimum wind direction (degrees)
  wd_max: 285   # Maximum wind direction (degrees)
```

**Note:** `min` == `max` values are allowed for fixed (non-random) conditions.

#### 3. Measurement Level (`mes_level`)
```yaml
mes_level:
  turb_ws: True      # Per-turbine wind speed
  turb_wd: False     # Per-turbine wind direction
  turb_TI: False     # Per-turbine turbulence intensity
  turb_power: False  # Per-turbine power output
  farm_ws: False     # Farm-wide wind speed
  farm_wd: False     # Farm-wide wind direction
  farm_TI: False     # Farm-wide turbulence intensity
  farm_power: False  # Farm-wide power output
```

#### 4. Measurement Details

Each measurement type needs detail configuration:

```yaml
ws_mes:
  ws_current: False          # Include current value
  ws_rolling_mean: True      # Include rolling mean
  ws_history_N: 1           # Number of history samples
  ws_history_length: 25     # History window length
  ws_window_length: 25      # Rolling mean window length

wd_mes:
  wd_current: False
  wd_rolling_mean: False
  wd_history_N: 1
  wd_history_length: 20
  wd_window_length: 20

yaw_mes:
  yaw_current: False
  yaw_rolling_mean: True
  yaw_history_N: 1
  yaw_history_length: 10
  yaw_window_length: 10

power_mes:
  power_current: False
  power_rolling_mean: False
  power_history_N: 1
  power_history_length: 10
  power_window_length: 10
```

### Optional Sections

#### Action Penalty (`act_pen`)
```yaml
act_pen:
  action_penalty: 0.0           # Penalty weight
  action_penalty_type: "Change"  # "Change", "Absolute", or "None"
```

#### Power Reward (`power_def`)
```yaml
power_def:
  Power_reward: "Baseline"  # "Baseline", "Power_avg", "Power_diff", or "None"
  Power_avg: 10            # Averaging window size
  Power_scaling: 1.0       # Reward scaling factor
```

#### Top-Level Options
```yaml
yaw_init: "Random"           # "Random", "Defined", or "Zeros"
noise: "None"                # Noise type
BaseController: "Local"      # "Local", "Global", or "PyWake"
ActionMethod: "wind"         # "yaw", "wind", or "absolute"
Track_power: False           # Whether to track power setpoint
```

#### Wind Probes (`probes`)
```yaml
probes:
  - name: probe_0
    turbine_index: 0
    relative_position: [-100, 0, 0]  # [x, y, z] relative to turbine
    include_wakes: true
    probe_type: WS  # "WS", "WD", or "TI"
```

## Simplified Constructor

The `WindFarmEnv.__init__` constructor has been simplified:

### New Recommended Usage

```python
from WindGym import WindFarmEnv
from WindGym.config_schema import SimulationConfig, ScalingConfig

env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="config.yaml",
    backend="dynamiks",  # or "pywake"

    # Optional: Use config objects for simulation parameters
    sim_config=SimulationConfig(
        dt_sim=1.0,
        dt_env=1.0,
        n_passthrough=5,
        burn_in_passthroughs=2,
        max_turb_move=2.0
    ),

    # Optional: Use config objects for scaling
    scaling_config=ScalingConfig(
        ws_min=0.0, ws_max=30.0,
        wd_min=0.0, wd_max=360.0,
        ti_min=0.0, ti_max=1.0,
        yaw_min=-45.0, yaw_max=45.0
    )
)
```

### Backward Compatibility

All old constructor parameters still work:

```python
# This still works (deprecated but supported)
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500],
    y_pos=[0, 0],
    config="config.yaml",
    dt_sim=1.0,              # Deprecated: use sim_config
    dt_env=1.0,              # Deprecated: use sim_config
    ws_scaling_min=0.0,      # Deprecated: use scaling_config
    ws_scaling_max=30.0,     # Deprecated: use scaling_config
    # ... etc
)
```

## Validation Examples

The configuration system validates inputs and provides clear error messages:

### Invalid Ranges
```python
# This will raise: ValueError: ws_min (15.0) must be less than or equal to ws_max (7.0)
WindConfig(ws_min=15, ws_max=7, TI_min=0.02, TI_max=0.15, wd_min=0, wd_max=360)
```

### Invalid Values
```python
# This will raise: ValueError: TI_min must be in [0, 1], got 1.5
WindConfig(ws_min=7, ws_max=15, TI_min=1.5, TI_max=2.0, wd_min=0, wd_max=360)
```

### Missing Required Fields
```python
# This will raise: ValueError: Key 'ws_min' is required in section 'wind'.
EnvConfig.from_dict({"farm": {"yaw_min": -45, "yaw_max": 45}})
```

## Programmatic Config Creation

For programmatic configuration:

```python
from WindGym.config_schema import (
    EnvConfig, FarmConfig, WindConfig,
    MeasurementLevelConfig, MeasurementDetailsConfig
)

# Create config programmatically
config = EnvConfig(
    farm=FarmConfig(yaw_min=-45, yaw_max=45),
    wind=WindConfig(
        ws_min=7, ws_max=15,
        TI_min=0.02, TI_max=0.15,
        wd_min=255, wd_max=285
    ),
    mes_level=MeasurementLevelConfig(turb_ws=True),
    ws_mes=MeasurementDetailsConfig(rolling_mean=True),
    wd_mes=MeasurementDetailsConfig(),
    yaw_mes=MeasurementDetailsConfig(rolling_mean=True),
    power_mes=MeasurementDetailsConfig(),
)

# Use in environment
env = WindFarmEnv(turbine=V80(), x_pos=[0, 500], y_pos=[0, 0], config=config)
```

## Export Config to YAML

```python
import yaml
from WindGym.config_schema import EnvConfig

# Load existing config
config = EnvConfig.from_dict(config_dict)

# Modify it
config.wind.ws_min = 8.0

# Export to YAML
with open('modified_config.yaml', 'w') as f:
    yaml.dump(config.to_dict(), f)
```

## Migration Guide

### For Existing Code

No changes required! All existing code continues to work. The config schema is used internally.

### For New Code

Use the new patterns for better type safety and IDE support:

**Before:**
```python
env = WindFarmEnv(
    turbine, x_pos, y_pos,
    config="config.yaml",
    dt_sim=1, dt_env=1,
    ws_scaling_min=0, ws_scaling_max=30,
    # ... 20+ more parameters
)
```

**After:**
```python
env = WindFarmEnv(
    turbine, x_pos, y_pos,
    config="config.yaml",  # Config handles most settings
    sim_config=SimulationConfig(dt_sim=1, dt_env=1),
    scaling_config=ScalingConfig(ws_min=0, ws_max=30)
)
```

## Benefits

1. **Early Error Detection**: Config errors caught at load time
2. **Clear Error Messages**: Know exactly what's wrong and where
3. **IDE Support**: Autocomplete and type checking
4. **Self-Documenting**: Type hints and docstrings explain everything
5. **Validation**: Ranges, types, and constraints automatically checked
6. **Backward Compatible**: All existing code continues to work

## Implementation Details

### Config Schema Location

- `WindGym/config_schema.py` - All configuration dataclasses
- `WindGym/wind_farm_env.py` - Updated to use schema

### Key Changes

1. **`_normalize_config_input()`** - Now returns validated `EnvConfig` object
2. **`_apply_config()`** - Works with `EnvConfig` instead of raw dict
3. **`__init__()`** - Simplified with grouped parameters
4. **`_init_managers()`** - Extracted manager initialization for clarity

### Testing

Run the configuration validation tests:

```bash
python test_config_validation.py
```

This tests:
- ✅ Basic config schema creation
- ✅ Config validation (catches errors)
- ✅ YAML file loading
- ✅ All example configs
- ✅ Config export/import roundtrip
