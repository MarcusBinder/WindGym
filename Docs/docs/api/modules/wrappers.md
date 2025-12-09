---
sidebar_position: 4
title: WindGym.wrappers
sidebar_label: Wrappers
---

# WindGym.wrappers

Gymnasium wrappers to extend environment functionality.

```python
from WindGym.wrappers import RecordEpisodeVals, CurriculumWrapper, PowerWrapper, AdversaryWrapper
```

---

## RecordEpisodeVals

Records episode statistics for analysis and logging.

### Import

```python
from WindGym.wrappers import RecordEpisodeVals
```

### Constructor

```python
env = RecordEpisodeVals(env)
```

### Usage

```python
from WindGym.wrappers import RecordEpisodeVals

# Wrap the base environment
env = RecordEpisodeVals(base_env)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        # Episode statistics available in info
        print(f"Episode reward: {info.get('episode', {}).get('r')}")
        obs, info = env.reset()
```

### Recorded Statistics

The wrapper adds episode statistics to the `info` dict when an episode ends:

| Key | Description |
|:----|:------------|
| `info['episode']['r']` | Total episode reward |
| `info['episode']['l']` | Episode length (steps) |
| `info['episode']['t']` | Episode time (seconds) |

---

## CurriculumWrapper

Implements curriculum learning by gradually increasing environment difficulty.

### Import

```python
from WindGym.wrappers import CurriculumWrapper
```

### Constructor

```python
env = CurriculumWrapper(
    env,
    initial_difficulty=0.5,   # Starting difficulty (0-1)
    max_difficulty=1.0,       # Maximum difficulty
    increase_rate=0.01,       # Difficulty increase per episode
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `env` | WindFarmEnv | *required* | Base environment |
| `initial_difficulty` | float | `0.5` | Starting difficulty level |
| `max_difficulty` | float | `1.0` | Maximum difficulty level |
| `increase_rate` | float | `0.01` | Increase per episode |

### Example

```python
from WindGym.wrappers import CurriculumWrapper

# Start with easier conditions, gradually increase difficulty
env = CurriculumWrapper(
    base_env,
    initial_difficulty=0.3,
    max_difficulty=1.0,
    increase_rate=0.005,
)

# Train agent - difficulty increases automatically
for episode in range(1000):
    obs, info = env.reset()
    # ... training loop
```

---

## PowerWrapper

Normalizes power output for reward scaling.

### Import

```python
from WindGym.wrappers import PowerWrapper
```

### Constructor

```python
env = PowerWrapper(env)
```

### Usage

```python
from WindGym.wrappers import PowerWrapper

env = PowerWrapper(base_env)
```

---

## AdversaryWrapper

Wrapper for adversarial training and robustness testing.

### Import

```python
from WindGym.wrappers import AdversaryWrapper
```

### Constructor

```python
env = AdversaryWrapper(env)
```

### Usage

```python
from WindGym.wrappers import AdversaryWrapper

# Create adversarial environment for robust training
env = AdversaryWrapper(base_env)
```

---

## Combining Wrappers

Wrappers can be stacked to combine functionality:

```python
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, CurriculumWrapper

# Create base environment
base_env = WindFarmEnv(...)

# Stack wrappers (order matters - outermost wrapper applied last)
env = RecordEpisodeVals(
    CurriculumWrapper(
        base_env,
        initial_difficulty=0.5,
    )
)
```
