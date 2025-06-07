"""
===================================================================================
WindGym Coliseum Example: Custom Wind Grid Evaluation
===================================================================================

This script demonstrates how to use the Coliseum evaluation framework to compare
agent performance over a specific, custom grid of wind conditions.

It is configured to run for:
- Wind Directions: [260, 265, 270, 275] degrees
- Wind Speeds: [7] m/s

This is achieved by setting the min/max values and step size in the
`run_wind_grid_evaluation` method call. The script also handles the `xarray.Dataset`
output and generates a heatmap plot of the results.
"""

import tempfile
import os
import matplotlib.pyplot as plt
import xarray as xr

# --- Imports from the WindGym project ---
from WindGym.FarmEval import FarmEval  # Use the flexible FarmEval class
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.Agents import ConstantAgent, PyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Configuration Section ---

# This multi-line string constant provides the environment's configuration.
YAML_CONFIG = """
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
farm: {yaw_min: -30, yaw_max: 30}
wind: {ws_min: 7, ws_max: 10, TI_min: 0.07, TI_max: 0.1, wd_min: 260, wd_max: 280}
act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
mes_level: {turb_ws: True, turb_wd: True, turb_TI: True, turb_power: True, farm_ws: True, farm_wd: True, farm_TI: True, farm_power: True}
ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""
x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

agents = {
    "Steering_Agent_Plus30": ConstantAgent(yaw_angles=[30, 0]),
    "PyWake Controller": PyWakeAgent(x_pos=x_pos, y_pos=y_pos)
}

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tmp_file:
   tmp_file.write(YAML_CONFIG)
   yaml_path = tmp_file.name

env_factory = lambda: FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    yaml_path=yaml_path,
    turbtype="None",
    Baseline_comp=True,
    reset_init=True,
    finite_episode=True # Ensures n_passthrough is respected for each grid point
)

coliseum = Coliseum(
    env_factory=env_factory,
    agents=agents,
    n_passthrough=4 # A short episode length is fine for each grid point
)

# Configure the grid parameters to match the request:
results_dataset = coliseum.run_wind_grid_evaluation(
    wd_min=260,
    wd_max=280,
    wd_step=5,
    ws_min=7,
    ws_max=8,
    ws_step=1,
    ti_points=1, # Use a single TI value
    save_netcdf="custom_grid_results.nc"
)

print("\n--- Examining Grid Evaluation Results (xarray.Dataset) ---")
print(results_dataset)

coliseum.plot_wind_grid_results(
   dataset=results_dataset,
   save_path="coliseum_custom_grid.png"
)
