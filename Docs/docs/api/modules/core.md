---
sidebar_position: 2
title: WindGym.core
sidebar_label: Core
---

# WindGym.core

Core components for environment customization and internal functionality.

```python
from WindGym.core import (
    RewardCalculator,
    WindManager,
    TurbulenceManager,
    MeasurementManager,
    WindFarmRenderer,
)
```

---

## RewardCalculator

Handles reward computation based on power production and action penalties.

### Import

```python
from WindGym.core import RewardCalculator
```

### Configuration

Rewards are configured via the YAML config file:

```yaml
# Reward configuration
power_reward: "Baseline"    # "Baseline", "Power_avg", "Power_diff", "None"
Power_scaling: 1.0          # Reward scaling factor
action_penalty: "Change"    # "Change", "Total", "None"
penalty_scaling: 0.01       # Penalty scaling factor
```

### Reward Types

| Type | Description |
|:-----|:------------|
| `Baseline` | Reward relative to baseline (no yaw) performance |
| `Power_avg` | Reward based on average power production |
| `Power_diff` | Reward based on power difference from previous step |
| `None` | No power-based reward |

### Action Penalty Types

| Type | Description |
|:-----|:------------|
| `Change` | Penalize large changes in yaw angles |
| `Total` | Penalize total yaw deviation from zero |
| `None` | No action penalty |

---

## WindManager

Manages wind conditions and sampling for the environment.

### Import

```python
from WindGym.core import WindManager, WindConditions
```

### WindConditions

Data class holding wind state:

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `ws` | float | Wind speed (m/s) |
| `wd` | float | Wind direction (degrees) |
| `TI` | float | Turbulence intensity |

### Configuration

Wind conditions are configured via YAML:

```yaml
# Wind sampling configuration
wind_speed:
  min: 4
  max: 25
  mean: 10
  std: 3

wind_direction:
  min: 0
  max: 360
  mean: 270
  std: 30

turbulence_intensity:
  min: 0.04
  max: 0.16
  mean: 0.08
  std: 0.02
```

---

## TurbulenceManager

Manages turbulence modeling and generation.

### Import

```python
from WindGym.core import TurbulenceManager
```

### Turbulence Types

| Type | Description |
|:-----|:------------|
| `Random` | Random turbulence generation |
| `MannGenerate` | Mann turbulence box generation |
| `MannLoad` | Load pre-generated Mann turbulence |
| `MannFixed` | Fixed Mann turbulence |
| `None` | No turbulence |

### Configuration

```python
env = WindFarmEnv(
    ...,
    turbtype="Random",        # Turbulence type
    TurbBox="Default",        # Path to turbulence box files
)
```

---

## MeasurementManager

Manages noise application to observations.

### Import

```python
from WindGym.core import MeasurementManager, MeasurementType
```

### Usage

```python
from WindGym.core import MeasurementManager, WhiteNoiseModel, MeasurementType

# Create measurement manager
manager = MeasurementManager(env)

# Configure noise for specific measurements
wd_noise = WhiteNoiseModel({MeasurementType.WIND_DIRECTION: 2.0})
ws_noise = WhiteNoiseModel({MeasurementType.WIND_SPEED: 0.5})

manager.set_noise_model(MeasurementType.WIND_DIRECTION, wd_noise)
manager.set_noise_model(MeasurementType.WIND_SPEED, ws_noise)
```

### Measurement Types

| Type | Description |
|:-----|:------------|
| `WIND_SPEED` | Wind speed measurements |
| `WIND_DIRECTION` | Wind direction measurements |
| `POWER` | Power output measurements |
| `YAW` | Yaw angle measurements |

### Methods

| Method | Description |
|:-------|:------------|
| `set_noise_model(type, model)` | Set noise model for measurement type |
| `apply_noise(obs, reset=False)` | Apply noise to observations |
| `get_measurement_spec()` | Get measurement specifications |

---

## Noise Models

### WhiteNoiseModel

Adds independent Gaussian noise at each timestep.

```python
from WindGym.core import WhiteNoiseModel, MeasurementType

noise_model = WhiteNoiseModel({
    MeasurementType.WIND_DIRECTION: 2.0,  # 2 degrees std dev
    MeasurementType.WIND_SPEED: 0.5,      # 0.5 m/s std dev
})
```

### EpisodicBiasNoiseModel

Adds consistent bias throughout an episode.

```python
from WindGym.core import EpisodicBiasNoiseModel, MeasurementType

noise_model = EpisodicBiasNoiseModel({
    MeasurementType.WIND_DIRECTION: 5.0  # 5 degrees bias std dev
})
```

### HybridNoiseModel

Combines white noise and episodic bias.

```python
from WindGym.core import HybridNoiseModel, MeasurementType

noise_model = HybridNoiseModel(
    white_noise_std={MeasurementType.WIND_DIRECTION: 2.0},
    episodic_bias_std={MeasurementType.WIND_DIRECTION: 5.0}
)
```

---

## WindFarmRenderer

Visualization rendering for the environment.

### Import

```python
from WindGym.core import WindFarmRenderer
```

### Usage

The renderer is used internally by the environment when `render_mode` is set:

```python
env = WindFarmEnv(
    ...,
    render_mode="human",  # or "rgb_array"
)

# Render the current state
env.render()
```

---

## Other Components

### BaselineManager

Manages baseline agent comparison for reward computation.

```python
from WindGym.core import BaselineManager
```

### ProbeManager

Manages wind probes in the environment.

```python
from WindGym.core import ProbeManager
```

### WindProbe

Wind probe utility for sampling wind conditions at specific locations.

```python
from WindGym.core import WindProbe
```

### Measurement Classes

Classes for handling different types of measurements:

```python
from WindGym.core import Mes, TurbMes, FarmMes
```

| Class | Description |
|:------|:------------|
| `Mes` | Base measurement class |
| `TurbMes` | Turbine-level measurements |
| `FarmMes` | Farm-level measurements |
