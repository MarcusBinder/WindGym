"""
===================================================================================
WindGym Coliseum Example: Time Series Evaluation with Controlled Episode Length
===================================================================================

This script demonstrates how to use the Coliseum evaluation framework from WindGym
to compare the performance of multiple agents.

Key features demonstrated:
- Use of an "environment factory" to create fresh, identical environments for each
  agent trial, ensuring a fair comparison.
- How to use the base `WindFarmEnv` class to have the episode length correctly
  determined by the `n_passthrough` parameter (the number of times the wind
  flows across the entire length of the farm).
- How to run a time series evaluation and plot the results.

This script compares two simple agents:
1. Steering_Agent_Plus10: A ConstantAgent that holds the first turbine at a +10 degree yaw.
2. Steering_Agent_Minus10: A ConstantAgent that holds the first turbine at a -10 degree yaw.
"""

import tempfile
import os
import matplotlib.pyplot as plt

# --- Imports from the WindGym project ---
from WindGym import WindFarmEnv  # Use the base environment for standard episode length
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.Agents import ConstantAgent, PyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Configuration Section ---

# This multi-line string constant provides the environment's configuration.
# It is written to a temporary file during execution.
YAML_CONFIG = """
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
farm: {yaw_min: -30, yaw_max: 30}
wind: {ws_min: 8, ws_max: 12, TI_min: 0.07, TI_max: 0.1, wd_min: 265, wd_max: 275}
act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
mes_level: {turb_ws: True, turb_wd: True, turb_TI: True, turb_power: True, farm_ws: True, farm_wd: True, farm_TI: True, farm_power: True}
ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""

# Create a temporary file for the YAML configuration.
with tempfile.NamedTemporaryFile(
    mode="w", delete=False, suffix=".yaml", encoding="utf-8"
) as tmp_file:
    tmp_file.write(YAML_CONFIG)
    yaml_filepath = tmp_file.name

print(f"Created temporary YAML config at: {yaml_filepath}")

# Define the layout for the wind farm
x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

# These agents will be evaluated. The environment will have 2 turbines.
agent_a = ConstantAgent(yaw_angles=[20, 0])  # Steers the first turbine
agent_b = ConstantAgent(yaw_angles=[-10, 0])  # Steers the first turbine the other way
agent_c = PyWakeAgent(x_pos=x_pos, y_pos=y_pos)

agents = {
    "Steering_Agent_Plus20": agent_a,
    "Steering_Agent_Minus10": agent_b,
    "pywake": agent_c,
}
print(f"Agents created: {list(agents.keys())}")


# The "environment factory" is a function (here, a lambda) that creates a
# new, fresh environment instance every time it's called. This ensures that
# each agent's evaluation trial is isolated and fair.
def env_factory():
    return WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        yaml_path=yaml_filepath,
        turbtype="None",
        Baseline_comp=True,
        reset_init=False,
    )


# Initialize Coliseum. The `n_passthrough` argument here will be passed to
# each environment created by the factory, controlling the episode length.
coliseum = Coliseum(
    env_factory=env_factory,
    agents=agents,
    n_passthrough=6,  # Let episodes last for half a flow passthrough
)
print("Coliseum instance created successfully.")

num_episodes = 2
summary_df = coliseum.run_time_series_evaluation(
    num_episodes=num_episodes, seed=123, save_detailed_history=True
)

print("\n--- Examining Evaluation Results ---")

# 1. Check the summary DataFrame
print("\n[1. Summary DataFrame]")
print(f"Type of summary_df: {type(summary_df)}")
print(f"Shape of summary_df: {summary_df.shape} (Expected: {num_episodes} rows)")
print(f"Columns in summary_df: {summary_df.columns.tolist()}")
print("First few rows of the summary:")
print(summary_df.head())

# 2. Check the detailed time_series_results
ts_results = coliseum.time_series_results
print("\n[2. Detailed Time Series History]")
print(f"Type of time_series_results: {type(ts_results)}")
print(f"Agents in history: {list(ts_results.keys())}")

# This will create and save a plot comparing the agents' performance.
coliseum.plot_time_series_comparison(save_path="coliseum_timeseries_comparison.png")
print("\nPlotting function executed successfully.")
print("Comparison plot saved as 'coliseum_timeseries_comparison.png'")
