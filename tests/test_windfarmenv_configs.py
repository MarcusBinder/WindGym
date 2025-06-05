# tests/test_windfarmenv_configs.py

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
import yaml # For reading YAML in the test
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from gymnasium.utils.env_checker import check_env
from copy import deepcopy # For debugging RNG state if needed

# --- Base YAML Configuration String (MODIFIED for determinism) ---
BASE_YAML_CONFIG_STRING_FOR_TESTS = """
# WindGym Test Configuration
yaw_init: "Zeros"
noise: "{noise_setting}" # Placeholder for noise setting
BaseController: "Local"
ActionMethod: "yaw"
Track_power: False
farm:
  yaw_min: -30
  yaw_max: 30
  xDist: 5
  yDist: 3
  nx: 2
  ny: 1
wind:
  ws_min: 8
  ws_max: 10
  TI_min: 0.05
  TI_max: 0.10
  wd_min: 268
  wd_max: 272
act_pen:
  action_penalty: 0.0 # Set to 0.0 to simplify reward determinism for now
  action_penalty_type: "Change"
power_def:
  Power_reward: "Baseline"
  Power_avg: 5 # CRITICAL: Made this <= history_length for relevant measurements
  Power_scaling: 1.0
mes_level:
  turb_ws: True
  turb_wd: {include_turbine_wd_yaml} # Placeholder
  turb_TI: True
  turb_power: True
  farm_ws: True
  farm_wd: {include_farm_wd_yaml} # Placeholder
  farm_TI: True
  farm_power: True
ws_mes:
  ws_current: True
  ws_rolling_mean: True
  ws_history_N: 1
  ws_history_length: 5 # CRITICAL: Aligned with Power_avg
  ws_window_length: 2
wd_mes:
  wd_current: True
  wd_rolling_mean: True
  wd_history_N: 1
  wd_history_length: 5 # CRITICAL: Aligned with Power_avg
  wd_window_length: 2
yaw_mes:
  yaw_current: True
  yaw_rolling_mean: True
  yaw_history_N: 1
  yaw_history_length: 5 # CRITICAL: Aligned with Power_avg
  yaw_window_length: 2
power_mes:
  power_current: True
  power_rolling_mean: True
  power_history_N: 1
  power_history_length: 5 # CRITICAL: Aligned with Power_avg
  power_window_length: 2
"""

@pytest.fixture
def temp_yaml_file_path(request):
    config_string = request.param
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding='utf-8') as tmp_file:
        tmp_file.write(config_string)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

@pytest.mark.parametrize(
    "temp_yaml_file_path, description",
    [
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="true", include_farm_wd_yaml="true"), "NoNoise_AllWD"),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="Normal", include_turbine_wd_yaml="true", include_farm_wd_yaml="true"), "NormalNoise_AllWD"),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="false", include_farm_wd_yaml="true"), "NoNoise_FarmWDOnly"),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="true", include_farm_wd_yaml="false"), "NoNoise_TurbWDOnly"),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="false", include_farm_wd_yaml="false"), "NoNoise_NoWD"),
    ],
    indirect=["temp_yaml_file_path"]
)
def test_wind_farm_env_configurations(temp_yaml_file_path, description):
    print(f"Testing configuration: {description} with YAML: {temp_yaml_file_path}")
    env = None
    current_power_avg = 5 # Default, aligned with YAML
    with open(temp_yaml_file_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        current_power_avg = yaml_config.get("power_def", {}).get("Power_avg", 5)
        # Get history length for measurements to ensure fill_window is adequate
        # Assuming all history lengths are critical and set to the same value in the test YAML
        critical_history_length = yaml_config.get("ws_mes", {}).get("ws_history_length", 5)


    env_params = dict(
        turbine=V80(),
        yaml_path=temp_yaml_file_path,
        seed=42,
        dt_sim=1,
        dt_env=10, # dt_env must be multiple of dt_sim
        yaw_step=1.0,
        turbtype="None", # Simplifies by removing turbulence field generation/loading randomness
        Baseline_comp=True, # Keep True to test reward paths that might use it
        # fill_window needs to be >= maxlen of deques used for rewards (Power_avg)
        # AND >= max_hist for MesClass.max_hist() if full history is needed at obs.
        # For reward determinism, ensure deques are consistently filled.
        fill_window=max(current_power_avg, critical_history_length)
    )
    env = WindFarmEnv(**env_params)

    # The check_env utility will run its own series of resets and steps.
    check_env(env, skip_render_check=True) # This is the main check

    # Basic reset and step checks (mostly covered by check_env)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert obs.shape == env.observation_space.shape, f"Observation shape mismatch (Obs: {obs.shape}, Space: {env.observation_space.shape}) for {description}"
    assert isinstance(info, dict), "Info should be a dictionary"

    num_raw_features = env._get_num_raw_features()
    expected_obs_len = env.farm_measurements.observed_variables()
    assert num_raw_features <= expected_obs_len, f"Raw features count ({num_raw_features}) mismatch with farm_measurements config ({expected_obs_len}) for {description}"
    assert len(obs) == expected_obs_len, f"Observation length {len(obs)} doesn't match expected {expected_obs_len} for {description}"

    for i in range(3): # A few steps for basic functional check
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray), f"Observation should be a numpy array after step {i+1}"
        assert obs.shape == env.observation_space.shape, f"Observation shape mismatch after step {i+1}"
        assert isinstance(reward, float), f"Reward should be a float after step {i+1}"
        assert isinstance(terminated, bool), f"Terminated flag should be a bool after step {i+1}"
        assert isinstance(truncated, bool), f"Truncated flag should be a bool after step {i+1}"
        assert isinstance(info, dict), f"Info should be a dictionary after step {i+1}"

    env.close()

@pytest.mark.parametrize(
    "temp_yaml_file_path, expected_wd_in_obs_calc",
    [
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="true", include_farm_wd_yaml="false"), True),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="false", include_farm_wd_yaml="false"), False),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="true", include_farm_wd_yaml="true"), True),
        (BASE_YAML_CONFIG_STRING_FOR_TESTS.format(noise_setting="None", include_turbine_wd_yaml="false", include_farm_wd_yaml="true"), True),
    ],
    indirect=["temp_yaml_file_path"]
)
def test_wind_direction_in_observations(temp_yaml_file_path, expected_wd_in_obs_calc):
    env = None
    try:
        env = WindFarmEnv(turbine=V80(), yaml_path=temp_yaml_file_path, seed=43, turbtype="None")
        _ = env.reset()

        farm_mes_config = env.farm_measurements

        num_turb_wd_features_per_turbine = 0
        if env.mes_level["turb_wd"]:
            turb_mes_wd_settings = env.wd_mes
            if turb_mes_wd_settings["wd_current"]:
                num_turb_wd_features_per_turbine += 1
            if turb_mes_wd_settings["wd_rolling_mean"]:
                num_turb_wd_features_per_turbine += turb_mes_wd_settings["wd_history_N"]
        total_turb_wd_features = num_turb_wd_features_per_turbine * env.n_turb

        num_farm_wd_features = 0
        if env.mes_level["farm_wd"]:
            farm_mes_wd_settings = env.wd_mes
            if farm_mes_wd_settings["wd_current"]:
                num_farm_wd_features += 1
            if farm_mes_wd_settings["wd_rolling_mean"]:
                num_farm_wd_features += farm_mes_wd_settings["wd_history_N"]
        
        total_expected_wd_features_in_obs = total_turb_wd_features + num_farm_wd_features

        if expected_wd_in_obs_calc:
            assert total_expected_wd_features_in_obs > 0, f"Wind direction features expected to be present but calculated count is {total_expected_wd_features_in_obs}."
        else:
            assert total_expected_wd_features_in_obs == 0, f"Wind direction features expected to be absent but calculated count is {total_expected_wd_features_in_obs}."
    finally:
        if env:
            env.close()
