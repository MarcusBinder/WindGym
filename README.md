# WindGym

A dynamic wind farm environment for developing and evaluating reinforcement learning agents for wind farm control.

[![coverage report](https://gitlab.windenergy.dtu.dk/sys/windgym/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/sys/windgym/-/commits/main)

📚 **[View the full documentation here](https://sys.pages.windenergy.dtu.dk/windgym/)**

## Overview

WindGym provides realistic wind farm simulations built on [DYNAMIKS](https://gitlab.windenergy.dtu.dk/DYNAMIKS/dynamiks) and [PyWake](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake). Agents learn to optimize turbine yaw control for power maximization and load reduction in complex wake interactions.

## Installation

```{bash}
git clone https://gitlab.windenergy.dtu.dk/sys/WindGym.git
cd WindGym
pixi install
pixi shell
```

## Examples

See the [examples directory](https://gitlab.windenergy.dtu.dk/sys/windgym/-/tree/main/examples) for complete usage demonstrations.


![Animation of flowfield](examples/images/Flowfield_gif.gif)




