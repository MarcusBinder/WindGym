# Running Simulations

This guide covers how to run simulations in WindGym, from basic environment setup to advanced configuration options.

---

## Quick Start

### Creating Your First Environment

The simplest way to create a WindGym environment:

```python
from WindGym.Wind_Farm_Env import WindFarmEnv

# Create a basic 3-turbine wind farm
env = WindFarmEnv(
    n_wt=3,          # Number of turbines
    ws=10.0,         # Wind speed (m/s)
    wd=270.0,        # Wind direction (degrees)
    TI=0.06,         # Turbulence intensity
)

# Reset the environment
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

### Running a Simple Episode

```python
# Run one episode with random actions
obs, info = env.reset()
total_reward = 0

for step in range(100):
    # Take a random action
    action = env.action_space.sample()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # Check if episode is done
    if terminated or truncated:
        print(f"Episode finished after {step + 1} steps")
        break

print(f"Total reward: {total_reward:.2f}")
env.close()
```

---

## Environment Configuration

### Wind Conditions

WindGym supports various ways to configure wind conditions:

#### Fixed Wind Conditions

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,        # Fixed wind speed
    wd=270.0,       # Fixed wind direction (270° = from West)
    TI=0.06,        # Turbulence intensity
)
```

#### Sampled Wind Conditions

Use PyWake's Site object to sample realistic wind distributions:

```python
from py_wake.site import UniformSite
from py_wake.wind_turbines import WindTurbine

# Create a site with wind resource distribution
site = UniformSite(
    p_wd=[0.1, 0.2, 0.3, 0.4],  # Probability for each direction sector
    a=[9.0],                     # Weibull A parameter (wind speed)
    k=[2.0],                     # Weibull k parameter (shape)
    ti=0.06
)

env = WindFarmEnv(
    n_wt=3,
    sample_site=site,  # Sample wind conditions from site
)
```

### Turbine Layout

Configure the wind farm layout:

```python
import numpy as np

# Create a 3x2 grid layout
x_positions = np.array([0, 500, 1000, 0, 500, 1000])  # meters
y_positions = np.array([0, 0, 0, 500, 500, 500])      # meters

env = WindFarmEnv(
    n_wt=6,
    x_pos=x_positions,
    y_pos=y_positions,
    ws=10.0,
    wd=270.0,
)
```

### Simulation Timesteps

Control the simulation granularity:

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    dt_sim=0.1,       # Internal simulation timestep (seconds)
    dt_env=1.0,       # Agent decision timestep (seconds)
)
```

**Note**: `dt_env` must be a multiple of `dt_sim`.

### Episode Length

Configure episode duration using flow passthroughs:

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    n_passthrough=3,           # Number of times flow passes through farm
    burn_in_passthroughs=1,    # Initial stabilization period
)
```

---

## Turbulence Models

### Mann Turbulence

Use high-fidelity Mann turbulence boxes:

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    TI=0.06,
    turbtype='mann',          # Use Mann turbulence model
    TurbBox='path/to/turbulence/box.hdf5',  # Optional: pre-generated box
)
```

### Random Turbulence

For faster simulations, use simpler turbulence:

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    TI=0.06,
    turbtype='random',  # Simple random turbulence
)
```

---

## Observation Configuration

Customize what the agent observes using YAML configuration:

```python
env = WindFarmEnv(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    yaml_path='path/to/config.yaml',  # Custom observation config
)
```

**Example `config.yaml`**:

```yaml
obs_config:
  turbine_level:
    - ws          # Wind speed
    - wd          # Wind direction
    - yaw         # Yaw angle
    - power       # Power output

  farm_level:
    - ws_mean     # Average farm wind speed
    - total_power # Total farm power

  include_history: true
  history_length: 10
```

See [Core Concepts](concepts.md#observations-and-measurement-system-mesclass) for more details on observations.

---

## Action Methods

Configure how agents control turbines:

### Yaw Change Method

Agent specifies change in yaw angle:

```python
# In your YAML config
ActionMethod: "yaw"
yaw_step_env: 3.0  # Maximum yaw change per step (degrees)
```

```python
# In your agent
action = np.array([1.0, -0.5, 0.0])  # Change yaw by [+3°, -1.5°, 0°]
```

### Wind-Relative Method

Agent specifies target yaw relative to wind direction:

```python
# In your YAML config
ActionMethod: "wind"
yaw_step_env: 3.0
```

```python
# In your agent
action = np.array([5.0, 10.0, 0.0])  # Target yaw: [+5°, +10°, 0°] from wind
```

---

## Rewards

Configure the reward function:

```python
# In your YAML config
power_reward: "Baseline"    # Compare to baseline
Power_scaling: 1.0          # Reward scaling factor
action_penalty: "Change"    # Penalize yaw changes
penalty_scaling: 0.01       # Penalty weight
```

**Reward Types**:
- `"Baseline"`: Reward based on improvement over baseline
- `"Power_avg"`: Normalized average power
- `"Power_diff"`: Power improvement over time
- `"None"`: No power reward

See [Core Concepts](concepts.md#rewards) for more details.

---

## Using Evaluation Wrapper

For detailed evaluation, use the `FarmEval` wrapper:

```python
from WindGym.FarmEval import FarmEval

env = FarmEval(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    TI=0.06,
    Baseline_comp=True,  # Run parallel baseline simulation
)

# Run episode
obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Get detailed results
results = env.get_results()
print(f"Power improvement: {results['power_improvement'].mean():.2%}")
```

---

## Multi-Agent Environments

For cooperative or competitive scenarios:

```python
from WindGym.WindEnvMulti import WindEnvMulti

env = WindEnvMulti(
    n_wt=6,
    n_agents=2,  # Number of agents
    ws=10.0,
    wd=270.0,
)

# Each agent controls a subset of turbines
observations = env.reset()
for agent_id, obs in observations.items():
    print(f"Agent {agent_id} observation shape: {obs.shape}")
```

---

## Visualization

### Plot Flow Field

```python
# After running simulation
env.plot_flow_field(
    time_idx=-1,  # Last timestep
    save_path='flowfield.png'
)
```

### Animate Episode

```python
from WindGym.utils.visualization import create_animation

create_animation(
    env,
    results,
    output_path='episode_animation.mp4',
    fps=10
)
```

---

## Complete Example

Putting it all together:

```python
from WindGym.Wind_Farm_Env import WindFarmEnv
from WindGym.Agents.PyWakeAgent import PyWakeAgent
import numpy as np

# Configure environment
env = WindFarmEnv(
    n_wt=3,
    x_pos=np.array([0, 500, 1000]),
    y_pos=np.array([0, 0, 0]),
    ws=10.0,
    wd=270.0,
    TI=0.06,
    dt_env=1.0,
    n_passthrough=3,
    turbtype='mann',
)

# Create agent
agent = PyWakeAgent(env)

# Run episode
obs, info = env.reset()
episode_rewards = []

for step in range(100):
    # Agent predicts action
    action, _ = agent.predict(obs)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    episode_rewards.append(reward)

    if terminated or truncated:
        break

print(f"Episode length: {len(episode_rewards)} steps")
print(f"Average reward: {np.mean(episode_rewards):.3f}")
print(f"Total reward: {np.sum(episode_rewards):.3f}")

env.close()
```

---

## Next Steps

- Learn about [different agent types](agents.md)
- Explore [evaluation tools](evaluations.md)
- Understand [noise and uncertainty](noise-and-uncertainty.md)
- Check out [complete examples](../examples/README.md)

---

## Troubleshooting

### Simulation runs slowly
- Use larger `dt_sim` and `dt_env` values
- Use `turbtype='random'` instead of `'mann'`
- Reduce `n_passthrough`

### Memory issues
- Reduce `history_length` in observation config
- Use fewer turbines
- Decrease turbulence box resolution

### Unstable rewards
- Increase `burn_in_passthroughs`
- Use reward smoothing or moving averages
- Check action penalty scaling

For more help, see the [Troubleshooting Guide](troubleshooting.md).
