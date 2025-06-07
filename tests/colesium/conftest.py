# tests/conf2/conftest.py
import pytest
import tempfile
import os
import yaml

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.FarmEval import FarmEval
from WindGym.Agents import ConstantAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Fixture for YAML configuration ---
@pytest.fixture(scope="module")
def simple_yaml_config_for_coliseum():
    """Provides a minimal, valid YAML configuration string for Coliseum tests."""
    return """
    yaw_init: "Zeros"
    noise: "None"
    BaseController: "Local"
    ActionMethod: "wind"
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
    """Creates a temporary YAML file from the config and yields its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(simple_yaml_config_for_coliseum)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

# --- Fixture for Agents ---
@pytest.fixture(scope="module")
def coliseum_agents():
    """Provides a dictionary of simple, deterministic agents for testing."""
    return {
        "Steering_Agent": ConstantAgent(yaw_angles=[10, 0]),
        "No_Steering_Agent": ConstantAgent(yaw_angles=[0, 0])
    }

# --- FIX: Moved from test_coliseum_basic.py to be shared ---
@pytest.fixture
def coliseum_instance(temp_yaml_file_for_coliseum, coliseum_agents):
    """
    Provides a fast Coliseum instance configured with FarmEval and no turbulence.
    This is the primary instance used for most edge case and API tests.
    """
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    env_factory = lambda: FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        yaml_path=temp_yaml_file_for_coliseum,
        turbtype="None",  # Optimized for speed
        Baseline_comp=True,
        reset_init=True,
        finite_episode=True
    )

    return Coliseum(
        env_factory=env_factory,
        agents=coliseum_agents,
        n_passthrough=0.1 # Very short episodes for fast tests
    )

# --- Fixture for creating fast environments on the fly ---
@pytest.fixture
def fast_farm_eval_factory(temp_yaml_file_for_coliseum):
    """
    Provides a function (a factory) that creates a fast FarmEval environment,
    optimized for testing by disabling turbulence generation.
    """
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    def _factory():
        return FarmEval(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            yaml_path=temp_yaml_file_for_coliseum,
            turbtype="None",      # Key optimization for speed
            reset_init=False,   # Avoid slow reset during object creation
            finite_episode=True
        )
    return _factory
