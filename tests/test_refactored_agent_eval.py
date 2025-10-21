import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import os
import tempfile
import yaml

from WindGym import FarmEval
from WindGym.AgentEval import eval_single_fast
from WindGym.Agents import ConstantAgent
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid

@pytest.fixture
def temp_yaml_file_factory():
    """Factory for creating temporary YAML files for tests."""
    created_files = []

    def _create_temp_yaml(content_str, name_suffix=""):
        if isinstance(content_str, dict):
            content_str = yaml.dump(content_str)

        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f"_{name_suffix}.yaml", encoding="utf-8"
        )
        tf.write(content_str)
        filepath = tf.name
        tf.close()
        created_files.append(filepath)
        return filepath

    yield _create_temp_yaml
    for f_path in created_files:
        if os.path.exists(f_path):
            os.remove(f_path)


@pytest.fixture
def basic_farm_eval_env(temp_yaml_file_factory):
    """Provides a basic FarmEval environment for testing AgentEval."""
    yaml_config = {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
            "xDist": 5,
            "yDist": 3,
            "nx": 2,
            "ny": 1,
        },
        "wind": {
            "ws_min": 8,
            "ws_max": 8,
            "TI_min": 0.07,
            "TI_max": 0.07,
            "wd_min": 270,
            "wd_max": 270,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True, "turb_wd": True, "turb_TI": True, "turb_power": True,
            "farm_ws": True, "farm_wd": True, "farm_TI": True, "farm_power": True,
        },
        "ws_mes": {
            "ws_current": True, "ws_rolling_mean": False, "ws_history_N": 1,
            "ws_history_length": 1, "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True, "wd_rolling_mean": False, "wd_history_N": 1,
            "wd_history_length": 1, "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True, "yaw_rolling_mean": False, "yaw_history_N": 1,
            "yaw_history_length": 1, "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True, "power_rolling_mean": False, "power_history_N": 1,
            "power_history_length": 1, "power_window_length": 1,
        },
    }
    yaml_filepath = temp_yaml_file_factory(yaml_config, "agent_eval_env_config")

    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = FarmEval(
        turbine=V80(),
        config=yaml_filepath,
        x_pos=x_pos,
        y_pos=y_pos,
        turbtype="None",
        seed=42,
        dt_sim=1,
        dt_env=10,
        Baseline_comp=True,
    )

    yield env
    env.close()


@pytest.fixture
def simple_constant_agent(basic_farm_eval_env):
    n_turb = basic_farm_eval_env.n_turb
    return ConstantAgent(yaw_angles=[0.0] * n_turb)


def test_eval_single_fast_returns_dataset(basic_farm_eval_env, simple_constant_agent):
    """Test that eval_single_fast returns a valid xarray.Dataset."""
    eval_t_sim = 20
    ds = eval_single_fast(
        env=basic_farm_eval_env,
        model=simple_constant_agent,
        ws=8,
        ti=0.07,
        wd=270,
        t_sim=eval_t_sim,
    )
    assert isinstance(ds, xr.Dataset), "eval_single_fast should return an xarray.Dataset"
    step_val = basic_farm_eval_env.sim_steps_per_env_step
    total_steps = eval_t_sim // step_val + 1
    assert len(ds.coords["time"]) == (total_steps * step_val + 1)


def test_eval_single_fast_with_user_vars(basic_farm_eval_env, simple_constant_agent):
    """Test that eval_single_fast correctly returns user-defined variables."""
    eval_t_sim = 20
    user_vars = ["nacelle_position"]
    ds = eval_single_fast(
        env=basic_farm_eval_env,
        model=simple_constant_agent,
        ws=8,
        ti=0.07,
        wd=270,
        t_sim=eval_t_sim,
        user_vars=user_vars,
    )
    assert "nacelle_position" in ds.variables
    assert ds["nacelle_position"].shape[0] == len(ds.coords["total_steps"])
