# tests/test_wind_farm_env.py

import pytest
import numpy as np
import yaml
from WindGym import WindFarmEnv  # Assuming your project structure
from py_wake.examples.data.hornsrev1 import V80  # Using a sample turbine
from unittest.mock import PropertyMock

# A minimal YAML config for testing purposes
TEST_CONFIG = """
farm:
  yaw_min: -30
  yaw_max: 30
wind:
  ws_min: 8
  ws_max: 8
  wd_min: 270
  wd_max: 270
  TI_min: 0.1
  TI_max: 0.1
mes_level:
  turb_ws: True
  turb_wd: False
  turb_TI: False
  turb_power: False
  farm_ws: False
  farm_wd: False
  farm_TI: True  # Must be True to activate the buffer logging
  farm_power: False
ws_mes:
  ws_history_length: 10
  ws_window_length: 5
  ws_history_N: 1
  ws_current: True
  ws_rolling_mean: True
# Add other minimal required configs...
yaw_mes:
  yaw_history_length: 10
  yaw_window_length: 1
  yaw_history_N: 1
  yaw_current: False
  yaw_rolling_mean: False
power_mes:
  power_history_length: 1
  power_window_length: 1
  power_history_N: 1
  power_current: False
  power_rolling_mean: False
wd_mes:
  wd_history_length: 1
  wd_window_length: 1
  wd_history_N: 1
  wd_current: False
  wd_rolling_mean: False
power_def:
    Power_scaling: 1.0
    Power_avg: 10
    Power_reward: "Baseline"
act_pen:
    action_penalty: 0.0
    action_penalty_type: "Total"
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
Track_power: False
"""


@pytest.fixture
def test_config_file(tmp_path):
    """Create a temporary YAML config file for the test."""
    p = tmp_path / "config.yaml"
    p.write_text(TEST_CONFIG)
    return p


def test_step_workflow_runs_without_crashing_and_logs_plausible_data(test_config_file):
    """
    This is an integration test or "smoke test". It runs the full step workflow
    without any mocks to ensure that the integration between the simulation
    library and the buffer logging code does not raise an exception and that
    the logged values are of the correct type and in a plausible range.
    """
    # ==========================================================================
    # 1. ARRANGE: Set up the environment with a real simulation.
    # ==========================================================================
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=np.array([0, 500]),
        y_pos=np.array([0, 0]),
        Baseline_comp=True,
        yaml_path=str(test_config_file),
        dt_sim=1,
        dt_env=1,
        turbtype="None",  # Use the simplest simulation type
        fill_window=2,
        reset_init=True,  # Let it fully initialize
    )

    # ==========================================================================
    # 2. ACT: Perform a step using the real, un-mocked workflow.
    # ==========================================================================
    action = np.zeros(env.action_space.shape)

    # We wrap the `step` call in a try/except block to give a clear
    # error message if the workflow itself crashes.
    env.step(action)

    # ==========================================================================
    # 3. ASSERT: Check for plausible data, not specific values.
    # ==========================================================================

    # --- Check Farm-Level Buffer ---
    farm_hf_buffer = env.farm_measurements.farm_mes.ws_hf_buffer
    assert len(farm_hf_buffer) == 3, "Farm buffer was not populated correctly."
    last_farm_ws = farm_hf_buffer[-1]

    # Sanity checks: Is it a valid number? Is it non-negative?
    assert isinstance(last_farm_ws, float), "Farm buffer logged a non-float value."
    assert not np.isnan(last_farm_ws), "Farm buffer logged a NaN value."
    assert last_farm_ws >= 0, "Farm buffer logged a negative wind speed."

    # --- Check Turbine-Specific Buffers ---
    for i in range(env.n_turb):
        turb_hf_buffer = env.farm_measurements.turb_mes[i].ws_hf_buffer
        assert (
            len(turb_hf_buffer) == 3
        ), f"Turbine {i} buffer was not populated correctly."
        last_turb_ws = turb_hf_buffer[-1]

        assert isinstance(
            last_turb_ws, float
        ), f"Turbine {i} buffer logged a non-float value."
        assert not np.isnan(last_turb_ws), f"Turbine {i} buffer logged a NaN value."
        assert last_turb_ws >= 0, f"Turbine {i} buffer logged a negative wind speed."

    env.close()
