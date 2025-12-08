# WindGym API Reference

This page provides an auto-generated reference for the main classes and functions in WindGym.

## Environment Classes

### *class* WindGym.WindFarmEnv

Bases: `Env`

#### metadata *= {'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, n_passthrough=5, ws_scaling_min=0.0, ws_scaling_max=30.0, wd_scaling_min=0, wd_scaling_max=360, ti_scaling_min=0.0, ti_scaling_max=1.0, yaw_scaling_min=-45, yaw_scaling_max=45, TurbBox='Default', turbtype='Random', backend='dynamiks', config=None, Baseline_comp=False, yaw_init=None, render_mode=None, seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, fill_window=True, sample_site=None, HTC_path=None, reset_init=True, burn_in_passthroughs=2, cleanup_on_time_limit=True, wd_function=None, max_turb_move=2, \*\*kwargs)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** (*bool*) – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.
  * **ws_scaling_min** (*float*)
  * **ws_scaling_max** (*float*)
  * **wd_scaling_min** (*float*)
  * **wd_scaling_max** (*float*)
  * **ti_scaling_min** (*float*)
  * **ti_scaling_max** (*float*)
  * **yaw_scaling_min** (*float*)
  * **yaw_scaling_max** (*float*)
  * **backend** (*str*)

#### init_render()

Initialize rendering - delegates to renderer.

#### reset(seed=None, options=None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

* **Parameters:**
  * **seed** (*int* *|* *None*)
  * **options** (*dict* *|* *None*)

#### step(action)

The step function
1. Adjust the yaw angles of the turbines
2. Take a step in the flow simulation
3. Update the measurements
4. Calculate the reward
5. Return the observation, reward, terminated, truncated and info

#### render()

Render method required by Gymnasium API - delegates to renderer.

#### close()

Close the environment and clean up resources.

#### plot_farm(baseline=False, fix_turbines=False)

Plot the entire farm layout - delegates to renderer.

#### plot_frame(baseline=False)

Plot a single frame - delegates to renderer.

#### *property* pywake_agent

Expose pywake_agent from baseline_manager for backward compatibility.

#### *property* py_agent_mode

Expose py_agent_mode from baseline_manager for backward compatibility.

### *class* WindGym.FarmEval

Bases: [`WindFarmEnv`](#WindGym.WindFarmEnv)

#### metadata *= {'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, finite_episode=False, ws_scaling_min=0.0, ws_scaling_max=30.0, wd_scaling_min=0, wd_scaling_max=360, ti_scaling_min=0.0, ti_scaling_max=1.0, yaw_scaling_min=-45, yaw_scaling_max=45, yaw_init='Zeros', TurbBox='Default', config=None, Baseline_comp=False, render_mode=None, turbtype='MannGenerate', seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, n_passthrough=5, HTC_path=None, reset_init=True, fill_window=True, sample_site=None, burn_in_passthroughs=2)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.
  * **finite_episode** (*bool*)
  * **ws_scaling_min** (*float*)
  * **ws_scaling_max** (*float*)
  * **wd_scaling_min** (*float*)
  * **wd_scaling_max** (*float*)
  * **ti_scaling_min** (*float*)
  * **ti_scaling_max** (*float*)
  * **yaw_scaling_min** (*float*)
  * **yaw_scaling_max** (*float*)

#### reset(seed=None, options=None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

#### set_wind_vals(ws=None, ti=None, wd=None)

Set the wind values to be used in the evaluation

#### set_yaw_vals(yaw_vals)

Set the yaw values to be used in the evaluation

#### update_tf(path)

Overwrite the \_def_site method to set the turbulence field to the path given

### *class* WindGym.WindFarmEnvMulti

Bases: `ParallelEnv`, [`WindFarmEnv`](#WindGym.WindFarmEnv)

#### metadata *= {'name': 'MultiFarm_environment_v0', 'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, n_passthrough=20, ws_scaling_min=0.0, ws_scaling_max=30.0, wd_scaling_min=0, wd_scaling_max=360, ti_scaling_min=0.0, ti_scaling_max=1.0, yaw_scaling_min=-45, yaw_scaling_max=45, TurbBox='Default', turbtype='MannGenerate', config=None, Baseline_comp=False, yaw_init=None, render_mode=None, seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=1, fill_window=True, sample_site=None, HTC_path=None, reset_init=False, burn_in_passthroughs=2)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.
  * **ws_scaling_min** (*float*)
  * **ws_scaling_max** (*float*)
  * **wd_scaling_min** (*float*)
  * **wd_scaling_max** (*float*)
  * **ti_scaling_min** (*float*)
  * **ti_scaling_max** (*float*)
  * **yaw_scaling_min** (*float*)
  * **yaw_scaling_max** (*float*)

#### render()

Render method required by Gymnasium API - delegates to renderer.

#### reset(seed=None, options=None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

#### step(actions)

The step function.
We unpack the actions, and call the step function of the parent class.

#### observation_space(agent)

#### action_space(agent)

## Wrappers

### *class* WindGym.wrappers.CurriculumWrapper

Bases: `Wrapper`

Curriculum wrapper for the WindGym environment.
This wrapper adds a curriculum-based similarity reward between the agent’s yaw
vector and a reference (“good”) yaw vector produced by a PyWakeAgent.
yaw_check options:

> - ‘current’: use the current yaw angles of the agent
> - ‘goal’: use the yaw angles that would have been used, with no yaw step limits (only for wind actions)

similarity_type options:
: - ‘l2’: negative L2 distance
  - ‘l1’: negative mean absolute error
  - ‘mse’: negative mean squared error
  - ‘normalized_l2’: 1 - (L2 distance / max_distance)
  - ‘exponential’: exp(-alpha \* L2 distance)
  - ‘cosine’: cosine similarity
  - ‘huber’: negative Huber loss

weight_function:
: function(step: int) -> float in [0,1], weighting env reward vs. similarity
  1 = env reward, 0 = similarity

#### \_\_init_\_(env, n_envs, similarity_type='normalized_l2', yaw_check='current', weight_function=<function CurriculumWrapper.<lambda>>, huber_kappa=1.0, exp_alpha=1.0)

* **Parameters:**
  * **env** (*gymnasium.Env*)
  * **n_envs** (*int*)
  * **similarity_type** (*str*)
  * **yaw_check** (*str*)
  * **huber_kappa** (*float*)
  * **exp_alpha** (*float*)

#### reset(\*\*kwargs)

Reset the environment and the pywake agent.

#### step(action)

Take a step in the environment and calculate the reward based on the similarity between the yaw angles of the agent and the pywake agent.

### *class* WindGym.wrappers.RecordEpisodeVals

Bases: `RecordEpisodeStatistics`

This wraps the RecordEpisodeStatistics Wrapper.
It also adds a queue to store the mean power of the episodes. This is used for the logging during training.
Could also be expanded upon to include more statistics if wanted.

#### \_\_init_\_(env, buffer_length=100)

* **Parameters:**
  **env** (*gymnasium.vector.vector_env.VectorEnv*)

#### reset(seed=None, options=None)

* **Parameters:**
  * **seed** (*int* *|* *list* *[**int* *]*  *|* *None*)
  * **options** (*dict* *|* *None*)

#### step(actions)

Steps through the environment, recording the episode statistics.

* **Parameters:**
  **actions** (*gymnasium.core.ActType*)
* **Return type:**
  tuple[gymnasium.core.ObsType, gymnasium.vector.vector_env.ArrayType, gymnasium.vector.vector_env.ArrayType, gymnasium.vector.vector_env.ArrayType, dict]

### *class* WindGym.core.NoisyWindFarmEnv

Bases: `Wrapper`

A Gym wrapper that applies measurement errors to a base WindFarm environment.

#### \_\_init_\_(base_env_class, measurement_manager, \*\*env_kwargs)

* **Parameters:**
  **measurement_manager** ([*MeasurementManager*](#WindGym.core.MeasurementManager))

#### reset(, seed=None, options=None)

* **Parameters:**
  * **seed** (*int* *|* *None*)
  * **options** (*dict* *|* *None*)
* **Return type:**
  tuple[numpy.ndarray, dict]

#### step(action)

* **Parameters:**
  **action** (*numpy.ndarray*)
* **Return type:**
  tuple[numpy.ndarray, float, bool, bool, dict]

#### close()

## Agent Classes

### *class* WindGym.Agents.BaseAgent.BaseAgent

Bases: `object`

#### \_\_init_\_(yaw_max=45, yaw_min=-45)

#### predict(\*args, \*\*kwargs)

#### scale_yaw(yaws)

Scale the yaw angles to be between -1 and 1.

#### unscale_yaw(action)

Unscale the action to the yaw range.

### *class* WindGym.Agents.PyWakeAgent

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(x_pos, y_pos, wind_speed=8, wind_dir=270, TI=0.07, yaw_max=45, yaw_min=-45, refine_pass_n=6, yaw_n=7, look_up=False, turbine=py_wake.examples.data.hornsrev1.V80, env=None)

#### update_wind(wind_speed, wind_direction, TI)

Update the wind conditions for the agent.

#### make_lookup()

Create a lookup table for the yaw angles.
This is done as we can save time by doing it once and then use it later.

#### use_lookup()

Use the lookup table to get the yaw angles for the current wind conditions.

#### reset()

Reset the wind things for the objective.

#### optimize()

Optimizes the yaw angles of the wind farm.

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.
Note that the command yaw offset is \_\_always_\_ defined relative to the incoming wind direction

#### calc_power(yaws)

Calculates the power of the farm, given the yaw angles.
Inputs are the yaw angles in degrees.
Returns the total power of the farm.

#### plot_flow()

Plot the flowfield of the wind farm.

### *class* WindGym.Agents.PyWakeAgent.NoisyPyWakeAgent

Bases: [`PyWakeAgent`](#WindGym.Agents.PyWakeAgent)

A version of the PyWakeAgent that makes decisions based on noisy observations.

Unlike the base PyWakeAgent which gets perfect global wind conditions, this
agent must estimate the wind conditions from the observation vector it
receives at each step. It then re-runs its optimization based on this
imperfect, noisy information.

#### \_\_init_\_(measurement_manager, \*\*kwargs)

Initializes the agent.

* **Parameters:**
  * **measurement_manager** ([*MeasurementManager*](#WindGym.core.MeasurementManager)) – The MeasurementManager instance
    from the environment. This is required for the agent to understand
    the structure of the observation vector.
  * **\*\*kwargs** – Keyword arguments to be passed to the parent PyWakeAgent,
    such as x_pos, y_pos, turbine, etc.

#### predict(obs, deterministic=None)

This method now uses the observation to make a decision.

### *class* WindGym.Agents.GreedyAgent.GreedyAgent

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(type='local', yaw_max=45, yaw_min=-45, yaw_step=1, env=None)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.

### *class* WindGym.Agents.RandomAgent.RandomAgent

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(env=None)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.

### *class* WindGym.Agents.ConstantAgent.ConstantAgent

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(yaw_angles, yaw_max=45, yaw_min=-45)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.

## Noise Models

### *class* WindGym.core.WhiteNoiseModel

Bases: `NoiseModel`

Applies Gaussian white noise defined in physical units (e.g., m/s, degrees).

#### \_\_init_\_(noise_std_devs)

* **Parameters:**
  **noise_std_devs** (*Dict* *[**MeasurementType* *,* *float* *]*)

#### apply_noise(observations, specs, rng)

* **Parameters:**
  * **observations** (*numpy.ndarray*)
  * **specs** (*List* *[**MeasurementSpec* *]*)
  * **rng** (*numpy.random.Generator*)
* **Return type:**
  numpy.ndarray

#### get_info()

* **Return type:**
  *Dict*

### *class* WindGym.core.EpisodicBiasNoiseModel

Bases: `NoiseModel`

Applies a consistent bias for an entire episode, defined in physical units.

#### \_\_init_\_(bias_ranges)

* **Parameters:**
  **bias_ranges** (*Dict* *[**MeasurementType* *,* *Tuple* *[**float* *,* *float* *]* *]*)

#### reset_noise(specs, rng)

* **Parameters:**
  * **specs** (*List* *[**MeasurementSpec* *]*)
  * **rng** (*numpy.random.Generator*)

#### apply_noise(observations, specs, rng)

Applies the sampled episodic bias to the given observations.

* **Parameters:**
  * **observations** (*numpy.ndarray*)
  * **specs** (*List* *[**MeasurementSpec* *]*)
  * **rng** (*numpy.random.Generator*)
* **Return type:**
  numpy.ndarray

#### get_info()

* **Return type:**
  *Dict*

### *class* WindGym.core.HybridNoiseModel

Bases: `NoiseModel`

#### \_\_init_\_(models)

* **Parameters:**
  **models** (*List* *[**NoiseModel* *]*)

#### reset_noise(specs, rng)

* **Parameters:**
  * **specs** (*List* *[**MeasurementSpec* *]*)
  * **rng** (*numpy.random.Generator*)

#### apply_noise(observations, specs, rng)

* **Parameters:**
  * **observations** (*numpy.ndarray*)
  * **specs** (*List* *[**MeasurementSpec* *]*)
  * **rng** (*numpy.random.Generator*)
* **Return type:**
  numpy.ndarray

#### get_info()

* **Return type:**
  *Dict*

### *class* WindGym.core.MeasurementManager

Bases: `object`

Orchestrates measurement specifications and the application of noise.

#### \_\_init_\_(env, seed=None)

#### seed(seed=None)

Reseeds the random number generator for the noise model.

* **Parameters:**
  **seed** (*int* *|* *None*)

#### set_noise_model(noise_model)

* **Parameters:**
  **noise_model** (*NoiseModel*)

#### reset_noise()

#### apply_noise(clean_observations)

* **Parameters:**
  **clean_observations** (*numpy.ndarray*)
* **Return type:**
  *Tuple*[numpy.ndarray, *Dict*]

## Evaluation Tools

### *class* WindGym.utils.evaluate_PPO.Coliseum

Bases: `object`

Enhanced evaluation framework to compare multiple agents in WindFarm environments.

Features:
- Time series evaluation with detailed episode history
- Wind condition grid evaluation with NetCDF export
- Mean cumulative reward tracking
- Flexible agent management with custom labels
- Comprehensive plotting capabilities

#### \_\_init_\_(env_factory, agents, agent_labels=None, n_passthrough=1.0, burn_in_passthroughs=2.0)

Initialize the Coliseum evaluation framework.

* **Parameters:**
  * **env_factory** (*Callable*) – Function that returns a new environment instance.
    Example: lambda: WindFarmEnv(…)
  * **agents** (*Union* *[**Dict* *[**str* *,* *object* *]* *,* *List* *[**object* *]* *]*) – Either a dictionary {name: agent} or list of agent objects.
    All agents must have a .predict(obs, deterministic) method.
  * **agent_labels** (*Optional* *[**List* *[**str* *]* *]*) – Custom labels for agents when using list input.
    If None, defaults to “Agent_0”, “Agent_1”, etc.
  * **n_passthrough** (*float* *,* *optional*) – Number of flow passthroughs for episode length.
    Defaults to 1.0.
  * **burn_in_passthroughs** (*float* *,* *optional*) – Number of flow passthroughs before episode

#### run_time_series_evaluation(num_episodes=10, seed=42, deterministic=True, save_detailed_history=True)

Run time series evaluation with stochastic wind conditions using sample_site.

This method relies on the environment’s sample_site for realistic wind sampling.
Each episode will have different wind conditions sampled from the site’s
wind resource distributions (Weibull for wind speed, frequency for direction).

* **Parameters:**
  * **num_episodes** (*int*) – Number of episodes to run
  * **seed** (*int*) – Master seed for reproducibility
  * **deterministic** (*bool*) – Whether to use deterministic agent policies
  * **save_detailed_history** (*bool*) – Whether to save detailed time series data
* **Returns:**
  Summary results with mean cumulative rewards
* **Return type:**
  pd.DataFrame

#### run_wind_grid_evaluation(wd_step=10, ws_step=2, ti_points=3, wd_min=None, wd_max=None, ws_min=None, ws_max=None, ti_min=None, ti_max=None, deterministic=True, save_netcdf=None)

Run evaluation over a grid of wind conditions and return as xarray Dataset.

* **Parameters:**
  * **wd_step** (*int*) – Wind direction step size in degrees
  * **ws_step** (*int*) – Wind speed step size in m/s
  * **ti_points** (*int*) – Number of turbulence intensity points
  * **wd_min** (*Optional* *[**float* *]*) – Minimum wind direction. If None, uses env.wd_min
  * **wd_max** (*Optional* *[**float* *]*) – Maximum wind direction. If None, uses env.wd_max
  * **ws_min** (*Optional* *[**float* *]*) – Minimum wind speed. If None, uses env.ws_min
  * **ws_max** (*Optional* *[**float* *]*) – Maximum wind speed. If None, uses env.ws_max
  * **ti_min** (*Optional* *[**float* *]*) – Minimum turbulence intensity. If None, uses env.TI_min
  * **ti_max** (*Optional* *[**float* *]*) – Maximum turbulence intensity. If None, uses env.TI_max
  * **deterministic** (*bool*) – Whether to use deterministic policies
  * **save_netcdf** (*Optional* *[**str* *]*) – Path to save NetCDF file
* **Returns:**
  Results with dimensions (wd, ws, ti) and variables for each agent
* **Return type:**
  xr.Dataset

#### plot_time_series_comparison(episodes_to_plot=None, save_path='time_series_comparison.png')

Plot time series comparison of mean cumulative rewards.

* **Parameters:**
  * **episodes_to_plot** (*Optional* *[**List* *[**int* *]* *]*) – Specific episodes to plot.
    If None, plots first 3 episodes.
  * **save_path** (*str*) – Path to save the figure

#### plot_summary_comparison(save_path='summary_comparison.png')

Plot summary comparison showing average performance across all episodes.

* **Parameters:**
  **save_path** (*str*) – Path to save the figure

#### plot_wind_grid_results(dataset, agent_name=None, save_path='wind_grid_results.png')

Plot wind grid evaluation results as heatmaps.

* **Parameters:**
  * **dataset** (*xr.Dataset*) – Results from wind grid evaluation
  * **agent_name** (*Optional* *[**str* *]*) – Specific agent to plot. If None, plots all agents.
  * **save_path** (*str*) – Path to save the figure

#### get_summary_statistics()

Get summary statistics for all agents across all episodes.

* **Return type:**
  pandas.DataFrame

#### *static* create_env_factory_with_site(env_class, site, \*\*env_kwargs)

Helper method to create an environment factory with sample_site configured.

* **Parameters:**
  * **env_class** – Environment class (e.g., WindFarmEnv, EvaluationEnv)
  * **site** – PyWake site object for realistic wind sampling
  * **\*\*env_kwargs** – Additional environment parameters
* **Returns:**
  Environment factory function
* **Return type:**
  Callable

### Example

from py_wake.examples.data.hornsrev1 import Hornsrev1Site

site = Hornsrev1Site()
env_factory = Coliseum.create_env_factory_with_site(

> WindFarmEnv, site,
> turbine=V80(), x_pos=x_pos, y_pos=y_pos,
> config=”config.yaml”

)
coliseum = Coliseum(env_factory, agents)

## Utility Functions

### Layout Generation

### WindGym.utils.generate_layouts.generate_square_grid(turbine, nx, ny, xDist, yDist)

Create a square grid of turbines.

* **Parameters:**
  * **turbine** (*WindTurbine*) – The wind turbine object.
  * **nx** (*int*) – Number of turbines in the x-direction.
  * **ny** (*int*) – Number of turbines in the y-direction.
  * **xDist** (*float*) – Diameter distance between turbines in the x-direction.
  * **yDist** (*float*) – Diameter distance between turbines in the y-direction.
* **Returns:**
  Array of turbine positions.
* **Return type:**
  np.ndarray

### WindGym.utils.generate_layouts.generate_circle(n, r, angle_offset=0)

Generate a circular grid of n points with radius r.

### WindGym.utils.generate_layouts.generate_cirular_farm(n_list, turbine, r_dist=5, angle_offset_list=None)

Generate a circular farm of n circular grids with radius r and m points.

* **Parameters:**
  * **n_list** (*numpy.typing.ArrayLike*)
  * **r_dist** (*float*)
  * **angle_offset_list** (*numpy.typing.ArrayLike*)

### WindGym.utils.generate_layouts.generate_staggered_grid(turbine, nx, ny, xDist, yDist, x_stagger_offset=None, y_stagger_offset=None)

Create a staggered grid of turbines with column- or row-based offsets.

* **Parameters:**
  * **turbine** (*WindTurbine*) – The wind turbine object.
  * **nx** (*int*) – Number of turbines in the x-direction.
  * **ny** (*int*) – Number of turbines in the y-direction.
  * **xDist** (*float*) – Distance between turbines in the x-direction, in rotor diameters.
  * **yDist** (*float*) – Distance between turbines in the y-direction, in rotor diameters.
  * **x_stagger_offset** (*list* *[**float* *] or* *None*) – List of horizontal offsets (in rotor diameters) per column.
  * **y_stagger_offset** (*list* *[**float* *] or* *None*) – List of vertical offsets (in rotor diameters) per column.
* **Returns:**
  Array of turbine positions.
* **Return type:**
  np.ndarray

### WindGym.utils.generate_layouts.plot_farm(x, y, turbine=None, D=None)

Plots the turbines in the farm layout, and their minimum distance to the closest turbine

## Related Pages

- [Core Concepts](concepts.md) - Detailed explanations of key concepts
- [Agents](agents.md) - Agent development guide
- [Simulations](simulations.md) - Running simulations
- [Evaluations](evaluations.md) - Evaluation tools and methods
