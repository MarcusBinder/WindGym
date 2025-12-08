---
sidebar_position: 1
title: API Reference
---

# API Reference

This section contains the auto-generated API documentation for the WindGym package. The documentation is generated from Python docstrings using Sphinx.

## Modules

- [**WindGym**](./modules/windgym.md) - Main package with environment classes
- [**WindGym.core**](./modules/core.md) - Core components (reward calculator, wind manager, etc.)
- [**WindGym.Agents**](./modules/agents.md) - Agent implementations
- [**WindGym.wrappers**](./modules/wrappers.md) - Gymnasium wrappers
- [**WindGym.visualization**](./modules/visualization.md) - Visualization utilities
- [**WindGym.utils**](./modules/utils.md) - Utility functions

## Quick Links

### Environment Classes

| Class | Description |
|-------|-------------|
| `WindFarmEnv` | Base wind farm environment |
| `WindFarmEnvMulti` | Multi-agent environment (PettingZoo compatible) |
| `FarmEval` | Evaluation wrapper with detailed tracking |
| `AgentEval` | Agent evaluation utility |

### Agent Classes

| Class | Description |
|-------|-------------|
| `BaseAgent` | Abstract base class for agents |
| `PyWakeAgent` | PyWake optimization baseline |
| `GreedyAgent` | Greedy control strategy |
| `RandomAgent` | Random control baseline |
| `ConstantAgent` | Constant action baseline |

### Wrappers

| Class | Description |
|-------|-------------|
| `RecordEpisodeVals` | Records episode statistics |
| `CurriculumWrapper` | Curriculum learning wrapper |
| `PowerWrapper` | Power normalization wrapper |
| `AdversaryWrapper` | Adversarial wrapper |

---

*This documentation is auto-generated. Run `./build-api-docs.sh` to regenerate.*
