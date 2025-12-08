# API Reference

This page provides an auto-generated reference for the main classes and functions in WindGym.

---

## Environment Classes

### `WindFarmEnv`

```python
from WindGym import WindFarmEnv

WindFarmEnv(
    turbine,
    x_pos,
    y_pos,
    n_passthrough,
    ws_scaling_min: float,
    ws_scaling_max: float,
    wd_scaling_min: float,
    wd_scaling_max: float,
    ti_scaling_min: float,
    ti_scaling_max: float,
    yaw_scaling_min: float,
    yaw_scaling_max: float,
    TurbBox,
    turbtype,
    backend: str,
    config,
    Baseline_comp,
    yaw_init,
    render_mode,
    seed,
    dt_sim,
    dt_env,
    yaw_step_sim,
    yaw_step_env,
    fill_window,
    sample_site,
    HTC_path,
    reset_init,
    burn_in_passthroughs,
    cleanup_on_time_limit: bool,
    wd_function,
    max_turb_move
)
```

**Key Methods:**

- `init_render()`: Initialize rendering - delegates to renderer.
- `reset(seed: Optional[int] = None, options: Optional[dict] = None)`: Reset the environment. This is called at the start of every episode. - The wind conditions are sampled, and the site is set. - The flow simulation is run for the time it takes for the flow to develop. - The measurements are filled up with the initial values.
- `step(action)`: The step function 1. Adjust the yaw angles of the turbines 2. Take a step in the flow simulation 3. Update the measurements 4. Calculate the reward 5. Return the observation, reward, terminated, truncated and info
- `render()`: Render method required by Gymnasium API - delegates to renderer.
- `close()`: Close the environment and clean up resources.
- `plot_farm(baseline = False, fix_turbines = False)`: Plot the entire farm layout - delegates to renderer.
- `plot_frame(baseline = False)`: Plot a single frame - delegates to renderer.
- `pywake_agent()`: Expose pywake_agent from baseline_manager for backward compatibility.
- `py_agent_mode()`: Expose py_agent_mode from baseline_manager for backward compatibility.

---

### `FarmEval`

```python
from WindGym import FarmEval

FarmEval(
    turbine,
    x_pos,
    y_pos,
    finite_episode: bool,
    ws_scaling_min: float,
    ws_scaling_max: float,
    wd_scaling_min: float,
    wd_scaling_max: float,
    ti_scaling_min: float,
    ti_scaling_max: float,
    yaw_scaling_min: float,
    yaw_scaling_max: float,
    yaw_init,
    TurbBox,
    config,
    Baseline_comp,
    render_mode,
    turbtype,
    seed,
    dt_sim,
    dt_env,
    yaw_step_sim,
    yaw_step_env,
    n_passthrough,
    HTC_path,
    reset_init,
    fill_window,
    sample_site,
    burn_in_passthroughs
)
```

**Key Methods:**

- `reset(seed = None, options = None)`
- `set_wind_vals(ws = None, ti = None, wd = None)`: Set the wind values to be used in the evaluation
- `set_yaw_vals(yaw_vals)`: Set the yaw values to be used in the evaluation
- `update_tf(path)`: Overwrite the _def_site method to set the turbulence field to the path given

---

### `WindFarmEnvMulti`

```python
from WindGym import WindFarmEnvMulti

WindFarmEnvMulti(
    turbine,
    x_pos,
    y_pos,
    n_passthrough,
    ws_scaling_min: float,
    ws_scaling_max: float,
    wd_scaling_min: float,
    wd_scaling_max: float,
    ti_scaling_min: float,
    ti_scaling_max: float,
    yaw_scaling_min: float,
    yaw_scaling_max: float,
    TurbBox,
    turbtype,
    config,
    Baseline_comp,
    yaw_init,
    render_mode,
    seed,
    dt_sim,
    dt_env,
    yaw_step_sim,
    yaw_step_env,
    fill_window,
    sample_site,
    HTC_path,
    reset_init,
    burn_in_passthroughs
)
```

**Key Methods:**

- `render()`
- `reset(seed = None, options = None)`
- `step(actions)`: The step function. We unpack the actions, and call the step function of the parent class.
- `observation_space(agent)`
- `action_space(agent)`

---

## Wrappers

### `CurriculumWrapper`

Curriculum wrapper for the WindGym environment. This wrapper adds a curriculum-based similarity reward between the agent's yaw vector and a reference ("good") yaw vector produced by a PyWakeAgent. yaw_check options: - 'current': use the current yaw angles of the agent - 'goal': use the yaw angles that would have been used, with no yaw step limits (only for wind actions)  similarity_type options: - 'l2': negative L2 distance - 'l1': negative mean absolute error - 'mse': negative mean squared error - 'normalized_l2': 1 - (L2 distance / max_distance) - 'exponential': exp(-alpha * L2 distance) - 'cosine': cosine similarity - 'huber': negative Huber loss weight_function: function(step: int) -> float in [0,1], weighting env reward vs. similarity 1 = env reward, 0 = similarity

```python
from WindGym.wrappers import CurriculumWrapper

CurriculumWrapper(
    env: gym.Env,
    n_envs: int,
    similarity_type: str,
    yaw_check: str,
    weight_function,
    huber_kappa: float,
    exp_alpha: float
)
```

**Key Methods:**

- `reset(**kwargs)`: Reset the environment and the pywake agent.
- `step(action)`: Take a step in the environment and calculate the reward based on the similarity between the yaw angles of the agent and the pywake agent.

---

### `RecordEpisodeVals`

This wraps the RecordEpisodeStatistics Wrapper. It also adds a queue to store the mean power of the episodes. This is used for the logging during training. Could also be expanded upon to include more statistics if wanted.

```python
from WindGym.wrappers import RecordEpisodeVals

RecordEpisodeVals(
    env: VectorEnv,
    buffer_length
)
```

**Key Methods:**

- `reset(seed: int | list[int] | None = None, options: dict | None = None)`
- `step(actions: ActType)`: Steps through the environment, recording the episode statistics.

---

### `NoisyWindFarmEnv`

A Gym wrapper that applies measurement errors to a base WindFarm environment.

```python
from WindGym.core import NoisyWindFarmEnv

NoisyWindFarmEnv(
    base_env_class,
    measurement_manager: MeasurementManager
)
```

**Key Methods:**

- `reset()`
- `step(action: np.ndarray)`
- `close()`

---

## Agent Classes

### `BaseAgent`

```python
from WindGym.Agents import BaseAgent

BaseAgent(
    yaw_max,
    yaw_min
)
```

**Key Methods:**

- `predict(*args, **kwargs)`
- `scale_yaw(yaws)`: Scale the yaw angles to be between -1 and 1.
- `unscale_yaw(action)`: Unscale the action to the yaw range.

---

### `PyWakeAgent`

```python
from WindGym.Agents import PyWakeAgent

PyWakeAgent(
    x_pos,
    y_pos,
    wind_speed,
    wind_dir,
    TI,
    yaw_max,
    yaw_min,
    refine_pass_n,
    yaw_n,
    look_up,
    turbine,
    env
)
```

**Key Methods:**

- `update_wind(wind_speed, wind_direction, TI)`: Update the wind conditions for the agent.
- `make_lookup()`: Create a lookup table for the yaw angles. This is done as we can save time by doing it once and then use it later.
- `use_lookup()`: Use the lookup table to get the yaw angles for the current wind conditions.
- `reset()`: Reset the wind things for the objective.
- `optimize()`: Optimizes the yaw angles of the wind farm.
- `predict(*args, **kwargs)`: This class pretends to be an agent, so we need to have a predict function. If we havent called the optimize function, we do that now, and return the action Note that we dont use the obs or the deterministic arguments. Note that the command yaw offset is __always__ defined relative to the incoming wind direction
- `calc_power(yaws)`: Calculates the power of the farm, given the yaw angles. Inputs are the yaw angles in degrees. Returns the total power of the farm.
- `plot_flow()`: Plot the flowfield of the wind farm.

---

### `NoisyPyWakeAgent`

A version of the PyWakeAgent that makes decisions based on noisy observations.  Unlike the base PyWakeAgent which gets perfect global wind conditions, this agent must estimate the wind conditions from the observation vector it receives at each step. It then re-runs its optimization based on this imperfect, noisy information.

```python
from WindGym.Agents import NoisyPyWakeAgent

NoisyPyWakeAgent(
    measurement_manager: MeasurementManager
)
```

**Key Methods:**

- `predict(obs, deterministic = None)`: This method now uses the observation to make a decision.

---

### `GreedyAgent`

```python
from WindGym.Agents import GreedyAgent

GreedyAgent(
    type,
    yaw_max,
    yaw_min,
    yaw_step,
    env
)
```

**Key Methods:**

- `predict(*args, **kwargs)`: This class pretends to be an agent, so we need to have a predict function. If we havent called the optimize function, we do that now, and return the action Note that we dont use the obs or the deterministic arguments.

---

### `RandomAgent`

```python
from WindGym.Agents import RandomAgent

RandomAgent(
    env
)
```

**Key Methods:**

- `predict(*args, **kwargs)`: This class pretends to be an agent, so we need to have a predict function. If we havent called the optimize function, we do that now, and return the action Note that we dont use the obs or the deterministic arguments.

---

### `ConstantAgent`

```python
from WindGym.Agents import ConstantAgent

ConstantAgent(
    yaw_angles,
    yaw_max,
    yaw_min
)
```

**Key Methods:**

- `predict(*args, **kwargs)`: This class pretends to be an agent, so we need to have a predict function. If we havent called the optimize function, we do that now, and return the action Note that we dont use the obs or the deterministic arguments.

---

## Noise Models

### `WhiteNoiseModel`

Applies Gaussian white noise defined in physical units (e.g., m/s, degrees).

```python
from WindGym.core import WhiteNoiseModel

WhiteNoiseModel(
    noise_std_devs: Dict[MeasurementType, float]
)
```

**Key Methods:**

- `apply_noise(observations: np.ndarray, specs: List[MeasurementSpec], rng: np.random.Generator)`
- `get_info()`

---

### `EpisodicBiasNoiseModel`

Applies a consistent bias for an entire episode, defined in physical units.

```python
from WindGym.core import EpisodicBiasNoiseModel

EpisodicBiasNoiseModel(
    bias_ranges: Dict[MeasurementType, Tuple[float, float]]
)
```

**Key Methods:**

- `reset_noise(specs: List[MeasurementSpec], rng: np.random.Generator)`
- `apply_noise(observations: np.ndarray, specs: List[MeasurementSpec], rng: np.random.Generator)`: Applies the sampled episodic bias to the given observations.
- `get_info()`

---

### `HybridNoiseModel`

```python
from WindGym.core import HybridNoiseModel

HybridNoiseModel(
    models: List[NoiseModel]
)
```

**Key Methods:**

- `reset_noise(specs: List[MeasurementSpec], rng: np.random.Generator)`
- `apply_noise(observations: np.ndarray, specs: List[MeasurementSpec], rng: np.random.Generator)`
- `get_info()`

---

### `MeasurementManager`

Orchestrates measurement specifications and the application of noise.

```python
from WindGym.core import MeasurementManager

MeasurementManager(
    env,
    seed
)
```

**Key Methods:**

- `seed(seed: Optional[int] = None)`: Reseeds the random number generator for the noise model.
- `set_noise_model(noise_model: NoiseModel)`
- `reset_noise()`
- `apply_noise(clean_observations: np.ndarray)`

---

## Evaluation Tools

### `Coliseum`

Enhanced evaluation framework to compare multiple agents in WindFarm environments.  Features: - Time series evaluation with detailed episode history - Wind condition grid evaluation with NetCDF export - Mean cumulative reward tracking - Flexible agent management with custom labels - Comprehensive plotting capabilities

```python
from WindGym.utils.evaluate_PPO import Coliseum

Coliseum(
    env_factory: Callable,
    agents: Union[Dict[str, object], List[object]],
    agent_labels: Optional[List[str]],
    n_passthrough: float,
    burn_in_passthroughs: float
)
```

**Key Methods:**

- `run_time_series_evaluation(num_episodes: int = 10, seed: int = 42, deterministic: bool = True, save_detailed_history: bool = True)`: Run time series evaluation with stochastic wind conditions using sample_site.  This method relies on the environment's sample_site for realistic wind sampling. Each episode will have different wind conditions sampled from the site's wind resource distributions (Weibull for wind speed, frequency for direction).
- `run_wind_grid_evaluation(wd_step: int = 10, ws_step: int = 2, ti_points: int = 3, wd_min: Optional[float] = None, wd_max: Optional[float] = None, ws_min: Optional[float] = None, ws_max: Optional[float] = None, ti_min: Optional[float] = None, ti_max: Optional[float] = None, deterministic: bool = True, save_netcdf: Optional[str] = None)`: Run evaluation over a grid of wind conditions and return as xarray Dataset.
- `plot_time_series_comparison(episodes_to_plot: Optional[List[int]] = None, save_path: str = 'time_series_comparison.png')`: Plot time series comparison of mean cumulative rewards.
- `plot_summary_comparison(save_path: str = 'summary_comparison.png')`: Plot summary comparison showing average performance across all episodes.
- `plot_wind_grid_results(dataset: xr.Dataset, agent_name: Optional[str] = None, save_path: str = 'wind_grid_results.png')`: Plot wind grid evaluation results as heatmaps.
- `get_summary_statistics()`: Get summary statistics for all agents across all episodes.
- `create_env_factory_with_site(env_class, site, **env_kwargs)`: Helper method to create an environment factory with sample_site configured.

---

## Utility Functions

### Layout Generation

Generate wind farm turbine layouts.

---

## Related Pages

- [Core Concepts](concepts.md) - Detailed explanations of key concepts
- [Agents](agents.md) - Agent development guide
- [Simulations](simulations.md) - Running simulations
- [Evaluations](evaluations.md) - Evaluation tools and methods
