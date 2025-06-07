# tests/test_coliseum_timeseries.py

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os
from unittest.mock import patch

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym import WindFarmEnv  # FIX: Import the base WindFarmEnv
from WindGym.Agents import ConstantAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Fixtures for setting up the test environment ---

@pytest.fixture(scope="module")
def simple_yaml_config_for_coliseum():
    """Provides a minimal, fully valid YAML configuration string for coliseum tests."""
    return """
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

@pytest.fixture(scope="module")
def temp_yaml_file_for_coliseum(simple_yaml_config_for_coliseum):
    """Creates a temporary YAML file and yields its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(simple_yaml_config_for_coliseum)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

@pytest.fixture(scope="module")
def coliseum_agents():
    """Provides a dictionary of simple, deterministic agents for testing."""
    # Using 2 turbines in the env_factory
    agent_a = ConstantAgent(yaw_angles=[10, 0])
    agent_b = ConstantAgent(yaw_angles=[-10, 0])
    return {"Steering_Agent": agent_a, "Non_Steering_Agent": agent_b}

@pytest.fixture
def coliseum_instance(temp_yaml_file_for_coliseum, coliseum_agents):
    """
    This fixture provides a Coliseum instance with a fast environment factory.
    It uses WindFarmEnv so that the episode length is controlled by n_passthrough.
    """
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    # FIX: Use WindFarmEnv to ensure n_passthrough controls the episode length.
    env_factory = lambda: WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        yaml_path=temp_yaml_file_for_coliseum,
        turbtype="None",
        Baseline_comp=True,
        reset_init=False,
    )

    return Coliseum(
        env_factory=env_factory,
        agents=coliseum_agents,
        n_passthrough=0.05  # A very short episode for fast testing
    )


# --- Test Functions ---

def test_coliseum_run_time_series_evaluation(coliseum_instance, coliseum_agents):
    """
    Tests the time series evaluation mode of the Coliseum, checking the format
    and content of the returned summary and detailed history data.
    """
    num_episodes = 2
    summary_df = coliseum_instance.run_time_series_evaluation(
        num_episodes=num_episodes,
        seed=123,
        save_detailed_history=True
    )

    # --- 1. Assertions on the summary DataFrame ---
    assert isinstance(summary_df, pd.DataFrame), "The returned summary should be a pandas DataFrame."
    assert summary_df.shape[0] == num_episodes, f"Expected {num_episodes} rows in the summary DataFrame."

    expected_columns = ["episode"] + list(coliseum_agents.keys())
    assert all(col in summary_df.columns for col in expected_columns), "Summary DataFrame is missing expected columns."
    assert not summary_df.drop(columns=['episode']).isnull().values.any(), "Summary DataFrame should not contain NaN values."

    # --- 2. Assertions on the detailed time_series_results ---
    ts_results = coliseum_instance.time_series_results
    assert isinstance(ts_results, dict), "time_series_results should be a dictionary."
    assert len(ts_results) == len(coliseum_agents), "Number of agents in time_series_results should match input."

    for agent_name, results_list in ts_results.items():
        assert agent_name in coliseum_agents, f"Unexpected agent '{agent_name}' found in results."
        assert len(results_list) == num_episodes, f"Expected {num_episodes} episode results for agent '{agent_name}'."

        single_episode_result = results_list[0]
        assert isinstance(single_episode_result, dict)
        expected_keys = ['history', 'final_mean_reward', 'total_reward', 'episode_length', 'agent_name']
        assert all(key in single_episode_result for key in expected_keys), "An episode result dict is missing keys."

        history = single_episode_result['history']
        assert isinstance(history, dict)
        expected_history_keys = ['step', 'reward', 'mean_cumulative_reward', 'info']
        assert all(key in history for key in expected_history_keys), "The history dict is missing keys."

        assert len(history['step']) > 0, "History 'step' array should not be empty."
        assert len(history['step']) == single_episode_result['episode_length'], "History length should match episode length."
        assert 'Power agent' in history['info'][0], "Info dict within history is missing 'Power agent' key."


# FIX: Patch the correct plotting function, which is 'savefig', not 'show'.
@patch('matplotlib.pyplot.savefig')
def test_coliseum_plot_time_series_runs_without_error(mock_savefig, coliseum_instance):
    """
    Tests that the time series plotting function can be called without crashing
    after an evaluation has been run. It correctly mocks `plt.savefig`.
    """
    # First, run an evaluation to generate the necessary data.
    coliseum_instance.run_time_series_evaluation(num_episodes=1, save_detailed_history=True)

    # Now, attempt to plot the results.
    try:
        coliseum_instance.plot_time_series_comparison(save_path="test_plot.png")
    except Exception as e:
        pytest.fail(f"plot_time_series_comparison raised an unexpected exception: {e}")

    # Check that the matplotlib 'savefig' function was called, indicating the plot was generated and saved.
    mock_savefig.assert_called_once()
    # Optional: More robustly check if it was called with the correct filename.
    mock_savefig.assert_called_once_with("test_plot.png", dpi=150, bbox_inches='tight')
