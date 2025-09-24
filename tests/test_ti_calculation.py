import pytest
import numpy as np
import yaml
import tempfile
import os

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid
from dynamiks.dwm import DWMFlowSimulation  # <-- ADD THIS IMPORT

# --- Test-Specific YAML Configuration (Unchanged) ---
TI_TEST_YAML_CONFIG = """
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
farm: {yaw_min: -30, yaw_max: 30}
wind: {ws_min: 10, ws_max: 10, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
power_def: {Power_reward: "None", Power_avg: 1, Power_scaling: 1.0}

mes_level:
  turb_ws: True
  turb_wd: False
  turb_TI: True
  turb_power: False
  farm_ws: False
  farm_wd: False
  farm_TI: False
  farm_power: False
  ti_sample_count: 12

ws_mes: {ws_current: True, ws_rolling_mean: True, ws_history_N: 1, ws_history_length: 5, ws_window_length: 3}
wd_mes: {wd_current: False, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: False, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""


@pytest.fixture
def ti_test_env(monkeypatch):
    """
    This fixture creates a WindFarmEnv instance and mocks both the measurements
    and the initial wake development run for deterministic testing.
    """
    # This list now only needs to be long enough for the reset loop and step calls.
    predefined_wind_speeds = [
        # Values for reset pre-fill (5 agent steps * 5 sim steps = 25 values)
        [10.0, 9.5],
        [10.1, 9.6],
        [9.9, 9.4],
        [10.5, 9.9],
        [9.5, 9.1],
        [10.2, 9.7],
        [10.3, 9.8],
        [9.7, 9.2],
        [10.8, 10.2],
        [9.2, 8.8],
        [10.0, 9.5],
        [10.0, 9.5],
        [10.0, 9.5],
        [10.0, 9.5],
        [11.0, 10.5],
        [9.0, 8.5],
        [9.5, 9.0],
        [10.5, 10.0],
        [10.0, 9.5],
        [10.0, 9.5],
        [10.1, 9.6],
        [10.2, 9.7],
        [10.3, 9.8],
        [10.4, 9.9],
        [10.5, 10.0],
        # Values for the actual test steps (3 agent steps * 5 sim steps = 15 values)
        [10.0, 9.5],
        [9.8, 9.3],
        [10.2, 9.7],
        [9.9, 9.4],
        [10.1, 9.6],
        [10.5, 10.0],
        [9.5, 9.0],
        [10.6, 10.1],
        [9.4, 8.9],
        [10.3, 9.8],
        [9.9, 9.4],
        [10.1, 9.6],
        [9.8, 9.3],
        [10.2, 9.7],
        [10.0, 9.5],
    ]
    step_counter = 0

    def mock_take_measurements(self):
        nonlocal step_counter
        self.current_ws = np.array(predefined_wind_speeds[step_counter])
        step_counter += 1
        self.current_wd = np.array([270.0, 270.0])
        self.current_yaw = self.fs.windTurbines.yaw
        self.current_powers = np.array([1e6, 0.8e6])

    # --- FIX: Apply mocks ---
    monkeypatch.setattr(WindFarmEnv, "_take_measurements", mock_take_measurements)
    # This new line prevents the initial `fs.run(t_developed)` from consuming our mock data
    monkeypatch.setattr(DWMFlowSimulation, "run", lambda self, t: None)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_file:
        tmp_file.write(TI_TEST_YAML_CONFIG)
        yaml_filepath = tmp_file.name

    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=1)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        dt_sim=1,
        dt_env=5,
        reset_init=False,
        turbtype="None",
    )

    yield env, predefined_wind_speeds

    env.close()
    os.remove(yaml_filepath)


def test_decoupled_ti_calculation(ti_test_env):
    """
    Tests that the TI calculation is correctly decoupled from the agent time step
    for a multi-turbine environment.
    """
    env, full_ws_sequence = ti_test_env

    # --- 1. Execution ---
    # With fs.run() mocked, reset() now consumes exactly 25 values (5 agent steps * 5 sim steps)
    env.reset(seed=42)

    # Take three more agent steps, consuming the next 15 values
    env.step(action=np.array([0.0, 0.0]))
    env.step(action=np.array([0.0, 0.0]))
    _, _, _, _, info = env.step(action=np.array([0.0, 0.0]))

    # --- 2. Verification ---
    # Total sim steps = 25 (reset) + 15 (3 steps) = 40.

    # A. Verify the High-Frequency TI Calculation
    # The TI buffer (maxlen=12) should contain the last 12 values consumed: from index 28 to 40.
    expected_hf_content_all = np.array(full_ws_sequence[28:40])

    expected_ti_turb0 = np.std(expected_hf_content_all[:, 0]) / np.mean(
        expected_hf_content_all[:, 0]
    )
    expected_ti_turb1 = np.std(expected_hf_content_all[:, 1]) / np.mean(
        expected_hf_content_all[:, 1]
    )

    actual_ti = env.farm_measurements.get_TI_turb()

    assert np.allclose(
        actual_ti[0], expected_ti_turb0
    ), "TI for turbine 0 is incorrect."
    assert np.allclose(
        actual_ti[1], expected_ti_turb1
    ), "TI for turbine 1 is incorrect."

    # B. Verify the Low-Frequency Agent Observation Buffer
    # This buffer should contain the AVERAGED values from each agent step.
    low_freq_buffer_content_turb0 = list(
        env.farm_measurements.turb_mes[0].ws.measurements
    )

    full_ws_array = np.array(full_ws_sequence)
    # The averages for the 3 steps we took AFTER the reset
    avg_step1 = np.mean(full_ws_array[25:30, 0])
    avg_step2 = np.mean(full_ws_array[30:35, 0])
    avg_step3 = np.mean(full_ws_array[35:40, 0])

    actual_last_three_averages = low_freq_buffer_content_turb0[-3:]

    assert np.isclose(
        actual_last_three_averages[0], avg_step1
    ), "Low-freq buffer has incorrect average for step 1."
    assert np.isclose(
        actual_last_three_averages[1], avg_step2
    ), "Low-freq buffer has incorrect average for step 2."
    assert np.isclose(
        actual_last_three_averages[2], avg_step3
    ), "Low-freq buffer has incorrect average for step 3."
