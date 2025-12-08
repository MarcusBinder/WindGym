---
sidebar_position: 4
title: WindGym.wrappers Module
---

# WindGym.wrappers Module

*This documentation is auto-generated from Python docstrings using Sphinx.*

This module contains Gymnasium wrappers for WindGym environments.

## RecordEpisodeVals

Records episode statistics for analysis.

```python
from WindGym.wrappers import RecordEpisodeVals

env = RecordEpisodeVals(base_env)
```

## CurriculumWrapper

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

## PowerWrapper

Power normalization wrapper.

```python
from WindGym.wrappers import PowerWrapper
```

## AdversaryWrapper

Adversarial wrapper for robust training.

```python
from WindGym.wrappers import AdversaryWrapper
```

---

*Run `./build-api-docs.sh` to regenerate this documentation from source.*
