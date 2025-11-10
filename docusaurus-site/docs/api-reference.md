# API Reference

This page provides a reference for the main classes and functions in WindGym.

---

## Environment Classes

### `WindFarmEnv`

The base wind farm environment class.

```python
from WindGym.Wind_Farm_Env import WindFarmEnv

env = WindFarmEnv(
    n_wt=3,                  # Number of wind turbines
    ws=10.0,                 # Wind speed (m/s)
    wd=270.0,                # Wind direction (degrees)
    TI=0.06,                 # Turbulence intensity
    x_pos=None,              # Turbine x positions (meters)
    y_pos=None,              # Turbine y positions (meters)
    dt_sim=0.1,              # Simulation timestep (seconds)
    dt_env=1.0,              # Environment timestep (seconds)
    n_passthrough=3,         # Number of flow passthroughs
    burn_in_passthroughs=1,  # Burn-in passthroughs
    turbtype='mann',         # Turbulence type ('mann' or 'random')
    TurbBox=None,            # Path to turbulence box file
    sample_site=None,        # PyWake Site for sampling wind conditions
    yaml_path=None,          # Path to configuration YAML
)
```

**Key Methods:**

- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return next state
- `render()`: Visualize the environment (optional)
- `close()`: Clean up resources

**Attributes:**

- `observation_space`: Gymnasium Space object defining observations
- `action_space`: Gymnasium Space object defining actions
- `n_wt`: Number of turbines
- `dt_env`: Environment timestep

---

### `FarmEval`

Evaluation wrapper for detailed performance tracking.

```python
from WindGym.FarmEval import FarmEval

env = FarmEval(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    TI=0.06,
    Baseline_comp=True,  # Enable baseline comparison
    # ... other WindFarmEnv parameters
)
```

**Additional Methods:**

- `get_results()`: Return xarray.Dataset with episode results
- `plot_flow_field(time_idx=-1, save_path=None)`: Visualize flow field
- `get_power_time_series()`: Get power production over time

---

### `WindEnvMulti`

Multi-agent wind farm environment.

```python
from WindGym.WindEnvMulti import WindEnvMulti

env = WindEnvMulti(
    n_wt=6,
    n_agents=2,  # Number of agents
    ws=10.0,
    wd=270.0,
    TI=0.06,
)
```

**Key Differences:**

- Returns dict of observations (one per agent)
- Requires dict of actions (one per agent)
- Compatible with PettingZoo interface

---

## Wrappers

### `NoisyWindFarmEnv`

Wrapper that adds measurement noise to observations.

```python
from WindGym.wrappers.NoisyWindFarmEnv import NoisyWindFarmEnv
from WindGym.noise.measurement_manager import MeasurementManager

manager = MeasurementManager(base_env)
noisy_env = NoisyWindFarmEnv(base_env, manager)
```

**Info Dictionary Additions:**

- `info['clean_obs']`: Clean (ground truth) observations
- `info['noise_info']`: Dictionary of applied noise for each measurement

---

### `RecordEpisodeVals`

Records episode statistics.

```python
from WindGym.wrappers import RecordEpisodeVals

env = RecordEpisodeVals(base_env)
```

---

### `CurriculumWrapper`

Implements curriculum learning by gradually increasing difficulty.

```python
from WindGym.wrappers import CurriculumWrapper

env = CurriculumWrapper(
    base_env,
    initial_difficulty=0.5,
    max_difficulty=1.0,
    increase_rate=0.01
)
```

---

## Agent Classes

### `BaseAgent`

Base class for all agents.

```python
from WindGym.Agents.BaseAgent import BaseAgent

class MyAgent(BaseAgent):
    def predict(self, obs):
        """
        Generate action from observation.

        Args:
            obs (np.ndarray): Current observation

        Returns:
            action (np.ndarray): Action to take
            state: Optional agent state
        """
        # Your control logic here
        action = ...
        return action, None
```

**Helper Methods:**

- `scale_yaw(yaw_angles)`: Scale yaw angles to [-1, 1] action space
- `unscale_yaw(scaled_actions)`: Convert scaled actions back to degrees

---

### `PyWakeAgent`

Optimal static yaw control using PyWake optimization.

```python
from WindGym.Agents.PyWakeAgent import PyWakeAgent

agent = PyWakeAgent(env)
action, _ = agent.predict(obs)
```

**Behavior:**
- Computes optimal yaw angles for current wind conditions
- Uses PyWake's internal optimization
- Does not adapt to observation changes during episode

---

### `NoisyPyWakeAgent`

Robust variant of PyWakeAgent for noisy observations.

```python
from WindGym.Agents.NoisyPyWakeAgent import NoisyPyWakeAgent

agent = NoisyPyWakeAgent(noisy_env)
action, _ = agent.predict(obs)
```

**Behavior:**
- Averages multiple wind measurements to estimate true conditions
- More robust to measurement noise than standard PyWakeAgent

---

### `GreedyAgent`

Simple reactive agent that aligns turbines with wind.

```python
from WindGym.Agents.GreedyAgent import GreedyAgent

agent = GreedyAgent(env, use_global_wind=True)
action, _ = agent.predict(obs)
```

**Parameters:**
- `use_global_wind` (bool): Use global wind direction vs local measurements

---

### `RandomAgent`

Takes random actions within action space.

```python
from WindGym.Agents.RandomAgent import RandomAgent

agent = RandomAgent(env)
action, _ = agent.predict(obs)
```

---

### `ConstantAgent`

Maintains fixed yaw angles.

```python
from WindGym.Agents.ConstantAgent import ConstantAgent
import numpy as np

yaw_angles = np.array([0.0, 5.0, -5.0])  # degrees
agent = ConstantAgent(env, yaw_angles=yaw_angles)
action, _ = agent.predict(obs)
```

---

## Noise Models

### `WhiteNoiseModel`

Adds independent Gaussian noise at each timestep.

```python
from WindGym.noise.noise_models import WhiteNoiseModel

noise_model = WhiteNoiseModel(std=2.0)  # Standard deviation in physical units
```

---

### `EpisodicBiasNoiseModel`

Adds consistent bias throughout an episode.

```python
from WindGym.noise.noise_models import EpisodicBiasNoiseModel

noise_model = EpisodicBiasNoiseModel(bias_std=5.0)  # Bias distribution std
```

---

### `HybridNoiseModel`

Combines white noise and episodic bias.

```python
from WindGym.noise.noise_models import HybridNoiseModel

noise_model = HybridNoiseModel(
    white_noise_std=2.0,
    episodic_bias_std=5.0
)
```

---

### `MeasurementManager`

Manages noise application to observations.

```python
from WindGym.noise.measurement_manager import MeasurementManager

manager = MeasurementManager(env)
manager.set_noise_model('wd', WhiteNoiseModel(std=2.0))
manager.set_noise_model('ws', WhiteNoiseModel(std=0.5))
```

**Methods:**

- `set_noise_model(measurement_type, noise_model)`: Configure noise for a measurement
- `apply_noise(clean_obs, reset=False)`: Apply noise to clean observations
- `get_measurement_spec()`: Get specification of all measurements

---

## Evaluation Tools

### `Coliseum`

Multi-agent evaluation framework.

```python
from WindGym.utils.evaluate_PPO import Coliseum

coliseum = Coliseum(
    env_factory=create_env,  # Function that returns new environment
    agents=agent_dict         # Dictionary of {name: agent}
)
```

**Methods:**

- `run_time_series_evaluation(n_episodes, save_histories=False)`: Stochastic evaluation
- `run_wind_grid_evaluation(wind_speeds, wind_directions, turbulence_intensities)`: Grid evaluation
- `plot_time_series_rewards(results, save_path=None)`: Plot time series
- `plot_summary_bar_chart(results, metric='mean_reward')`: Compare agents

**Returns:**
- Time series: pandas.DataFrame with episode statistics
- Grid: xarray.Dataset with results across wind conditions

---

## Utility Functions

### `generate_layouts`

Generate wind farm turbine layouts.

```python
from WindGym.utils import generate_layouts

# Generate grid layout
x_pos, y_pos = generate_layouts.grid_layout(
    n_rows=2,
    n_cols=3,
    spacing_x=500,  # meters
    spacing_y=500
)

# Generate circular layout
x_pos, y_pos = generate_layouts.circular_layout(
    n_turbines=6,
    radius=1000  # meters
)
```

---

## Configuration

### YAML Configuration File

WindGym uses YAML files to configure observations, actions, and rewards.

**Example `config.yaml`:**

```yaml
# Observation configuration
observations:
  turbine_level:
    - ws
    - wd
    - yaw
    - power
  farm_level:
    - ws_mean
    - total_power
  include_history: true
  history_length: 10

# Action configuration
ActionMethod: "yaw"  # or "wind"
yaw_min: -30
yaw_max: 30
yaw_step_env: 3.0

# Reward configuration
power_reward: "Baseline"  # or "Power_avg", "Power_diff", "None"
Power_scaling: 1.0
action_penalty: "Change"  # or "Total", "None"
penalty_scaling: 0.01
```

---

## Data Structures

### Observation Space

Observations are returned as flattened numpy arrays with values scaled to [-1, 1].

**Structure depends on YAML configuration:**
- Turbine-level measurements: repeated for each turbine
- Farm-level measurements: single values
- History: past N observations concatenated

**Example:**
```
obs = [ws_t1, wd_t1, yaw_t1, power_t1, ws_t2, wd_t2, yaw_t2, power_t2, ..., ws_farm, total_power]
```

---

### Action Space

Actions are numpy arrays of length `n_wt`, with values in [-1, 1] representing:
- **"yaw" method**: Change in yaw angle (scaled)
- **"wind" method**: Target yaw offset from wind direction (scaled)

---

### Info Dictionary

The `info` dict returned by `step()` contains:

**Standard keys:**
- `'episode'`: Episode statistics (if terminated)
- `'TimeLimit.truncated'`: Whether episode was truncated by time limit

**FarmEval additions:**
- `'power'`: Current power output per turbine
- `'yaw'`: Current yaw angles
- `'ws'`, `'wd'`: Current wind conditions

**NoisyWindFarmEnv additions:**
- `'clean_obs'`: Ground truth observations
- `'noise_info'`: Applied noise details

---

## Type Definitions

```python
# Observation
ObsType = np.ndarray  # Shape: (obs_dim,)

# Action
ActType = np.ndarray  # Shape: (n_wt,)

# Info
InfoDict = Dict[str, Any]

# Step return
StepReturn = Tuple[ObsType, float, bool, bool, InfoDict]

# Reset return
ResetReturn = Tuple[ObsType, InfoDict]
```

---

## Related Pages

- [Core Concepts](concepts.md) - Detailed explanations of key concepts
- [Agents](agents.md) - Agent development guide
- [Simulations](simulations.md) - Running simulations
- [Evaluations](evaluations.md) - Evaluation tools and methods
