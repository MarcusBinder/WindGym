import pytest
import numpy as np
import os
import shutil
import yaml
from unittest.mock import patch  # Needed for dynamic patching

# Import necessary classes from WindGym
from WindGym.FarmEval import FarmEval
from WindGym.Agents import PyWakeAgent, NoisyPyWakeAgent
from WindGym.Measurement_Manager import (
    MeasurementManager,
    MeasurementType,
    EpisodicBiasNoiseModel,
)
from WindGym.Measurement_Manager import NoisyWindFarmEnv
from py_wake.examples.data.hornsrev1 import V80


# --- Test Parameters ---
TRUE_GLOBAL_WD = (
    270.0  # The true wind direction in the environment (e.g., from North, clockwise)
)
ADVERSARIAL_BIAS_DEGREES = 180.0  # The bias injected by the adversary (+180 degrees)

# The wind direction the agent will perceive: 270 + 180 = 450 -> 90 degrees
AGENT_PERCEIVED_WD = (TRUE_GLOBAL_WD + ADVERSARIAL_BIAS_DEGREES) % 360
if AGENT_PERCEIVED_WD < 0:
    AGENT_PERCEIVED_WD += 360


# --- Helper to get PyWake's ideal optimal yaw offset for a single turbine ---
# For a single turbine in uniform flow, the optimal yaw offset for maximum
# power production is generally 0 degrees relative to the incoming wind.
def get_pywake_ideal_optimal_yaw_offset():
    return 0.0


EXPECTED_OPTIMAL_YAW_OFFSET_FOR_TRUE_WIND = get_pywake_ideal_optimal_yaw_offset()
EXPECTED_OPTIMAL_YAW_OFFSET_FOR_PERCEIVED_WIND = get_pywake_ideal_optimal_yaw_offset()

# Number of steps to run in the loop to allow for dynamic settling
NUM_STEPS_TO_RUN = 5


# --- Fixtures ---


# Parameterize the environment's yaw limits directly in the YAML config fixture
@pytest.fixture(
    scope="module",
    params=[
        {"yaw_min": -45, "yaw_max": 45, "name": "small_yaw_range"},
        {"yaw_min": -180, "yaw_max": 180, "name": "large_yaw_range"},
    ],
)
def dummy_yaml_config(tmp_path_factory, request):
    """
    Creates a dummy YAML config file for the WindFarmEnv with parameterized yaw limits.
    Returns (config_path, yaw_min_from_config, yaw_max_from_config, test_case_name).
    """
    params = request.param
    yaw_min = params["yaw_min"]
    yaw_max = params["yaw_max"]
    test_case_name = params["name"]

    config_data = {
        "farm": {
            "yaw_min": yaw_min,  # Injected yaw_min
            "yaw_max": yaw_max,  # Injected yaw_max
        },
        "wind": {
            "ws_min": 8.0,
            "ws_max": 12.0,
            "TI_min": 0.05,
            "TI_max": 0.1,
            "wd_min": 0,
            "wd_max": 360,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "None"},
        "power_def": {"Power_scaling": 1.0, "Power_avg": 10, "Power_reward": "None"},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": True,
            "farm_wd": True,
            "farm_TI": False,
            "farm_power": False,  # Ensure farm_wd is True
            "ti_sample_count": 5,  # Small for quick tests
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": False,
            "ws_history_N": 1,
            "ws_history_length": 5,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": False,
            "wd_history_N": 1,
            "wd_history_length": 5,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 1,
            "yaw_history_length": 5,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 5,
            "power_window_length": 1,
        },
        "ActionMethod": "yaw",  # <--- Changed to "yaw"
        "BaseController": "None",
        "yaw_init": "Zeros",
        "Track_power": False,
    }

    tmp_dir = tmp_path_factory.mktemp(f"config_{test_case_name}")
    config_path = tmp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return str(config_path), yaw_min, yaw_max, test_case_name


@pytest.fixture(scope="function")
def test_env_instance(dummy_yaml_config):
    """
    Provides a configured FarmEval environment instance and its corresponding yaw limits.
    Returns (env, yaw_min_cfg, yaw_max_cfg, test_case_name, env_init_kwargs_dict).
    """
    config_path, yaw_min_cfg, yaw_max_cfg, test_case_name = dummy_yaml_config

    # Define all arguments for FarmEval.__init__ to pass explicitly
    env_init_kwargs_dict = {
        "turbine": V80(),
        "x_pos": [0.0],
        "y_pos": [0.0],
        "config": config_path,
        "dt_sim": 1,
        "dt_env": 1,
        "n_passthrough": 1,
        "burn_in_passthroughs": 0,
        "reset_init": False,
        "HTC_path": None,
        "finite_episode": False,
        "yaw_init": "Zeros",
        "TurbBox": "Default",
        "Baseline_comp": False,
        "render_mode": None,
        "turbtype": "None",  # Crucial: No turbulence for speed and predictability
        "yaw_step_sim": 180.0,  # Crucial: Large yaw step to reach target quickly in one go
        "yaw_step_env": None,
        "fill_window": True,
        "sample_site": None,
    }

    env = FarmEval(**env_init_kwargs_dict)
    yield env, yaw_min_cfg, yaw_max_cfg, test_case_name, env_init_kwargs_dict
    env.close()


# --- Test Function ---
def test_noisy_pywake_agent_turning_behavior(test_env_instance):
    """
    Tests how the NoisyPyWakeAgent (with bias compensation logic) commands
    turbine yaw, and how environment limits affect the actual turn.
    It verifies behavior over multiple steps and directly compares compensation logic.
    """
    (
        original_env_from_fixture,
        yaw_min_cfg,
        yaw_max_cfg,
        test_case_name,
        env_init_kwargs_dict,
    ) = test_env_instance

    print(f"\n======== STARTING TEST: {test_case_name} ========")
    print(f"True Global WD: {TRUE_GLOBAL_WD}°")
    print(f"Adversarial Bias: {ADVERSARIAL_BIAS_DEGREES}°")
    print(f"Expected Agent Perceived WD (no noise): {AGENT_PERCEIVED_WD}°")
    print(f"Yaw Min/Max (from config): {yaw_min_cfg}° / {yaw_max_cfg}°")
    print("-" * 50)

    # --- 1. SETUP: Create the base FarmEval environment instance ---
    base_farm_env = FarmEval(**env_init_kwargs_dict)

    # --- 2. SETUP: Create MeasurementManager and set the noise model on IT ---
    # The MeasurementManager should wrap the actual base environment that will be stepped.
    mm = MeasurementManager(env=base_farm_env)  # Pass the actual base_farm_env here

    bias_noise_model = EpisodicBiasNoiseModel(
        bias_ranges={
            MeasurementType.WIND_DIRECTION: (
                ADVERSARIAL_BIAS_DEGREES,
                ADVERSARIAL_BIAS_DEGREES,
            )
        }
    )
    mm.set_noise_model(bias_noise_model)
    print("MeasurementManager and EpisodicBiasNoiseModel set up.")

    # --- 3. SETUP: Create the NoisyWindFarmEnv, wrapping the base_farm_env and the configured mm ---
    noisy_env = NoisyWindFarmEnv(
        base_env_class=type(base_farm_env),
        measurement_manager=mm,
        **env_init_kwargs_dict,
    )
    # Ensure noisy_env actually uses the base_farm_env instance
    # This might overwrite the env reference inside the mm. Let's explicitly set it back if needed.
    # A more robust way might be to instantiate MeasurementManager *after* NoisyWindFarmEnv if possible,
    # or ensure mm is a property that rebuilds/updates its internal env reference.
    # For now, let's ensure the mm object holds a reference to the active base_env.
    mm.env = (
        noisy_env.base_env
    )  # <--- Crucial: Ensure MM's internal env reference is the live one

    # Set the TRUE_GLOBAL_WD for the underlying physics of the base environment.
    noisy_env.base_env.set_wind_vals(ws=10.0, ti=0.07, wd=TRUE_GLOBAL_WD)
    print(f"Base environment's true WD set to: {noisy_env.base_env.wd}°")

    # --- 4. SETUP: Create the NoisyPyWakeAgent (under test - now an "Oracle Compensator") ---
    noisy_pywake_agent = NoisyPyWakeAgent(
        measurement_manager=mm,  # Use the MM instance that has the noise model
        x_pos=noisy_env.base_env.x_pos,
        y_pos=noisy_env.base_env.y_pos,
        turbine=noisy_env.base_env.turbine,
        wind_dir=TRUE_GLOBAL_WD,  # Initial wind_dir, overwritten by agent's estimate later
    )
    print("NoisyPyWakeAgent (agent under test) initialized.")

    # --- 5. EXECUTION LOOP ---
    print("\nStarting environment reset...")
    current_obs, current_info = noisy_env.reset(
        seed=42
    )  # This also calls mm.reset_noise() and mm.apply_noise()
    print("Environment reset complete. Initial info:")
    print(
        f"  obs_true/farm/wd_current (clean): {current_info.get('obs_true/farm/wd_current', 'N/A'):.4f}°"
    )

    sensed_wd_from_obs_at_reset = current_info["obs_sensed/farm/wd_current"]
    print(
        f"  obs_sensed/farm/wd_current (perceived by agent): {sensed_wd_from_obs_at_reset:.4f}°"
    )
    print(f"  Expected Agent Perceived WD: {AGENT_PERCEIVED_WD}°")

    np.testing.assert_allclose(
        sensed_wd_from_obs_at_reset,
        AGENT_PERCEIVED_WD,
        atol=4.0,
        err_msg=f"[{test_case_name}] Agent's perceived WD ({sensed_wd_from_obs_at_reset}) at reset does not match expected biased WD ({AGENT_PERCEIVED_WD})",
    )
    print("ASSERTION PASSED: Initial sensed WD matches expected biased WD.")
    print("-" * 50)

    # Variables to store final state after the loop
    final_sensed_wd_from_obs = None
    final_turbine_abs_heading = None
    action_from_agent = None

    for step_idx in range(NUM_STEPS_TO_RUN):
        print(
            f"\n--- Loop Step {step_idx + 1}/{NUM_STEPS_TO_RUN} (Test Case: {test_case_name}) ---"
        )

        # Agent makes a prediction based on current observations
        action_from_agent, _ = noisy_pywake_agent.predict(current_obs)

        # Environment steps with agent's action
        current_obs, _, _, _, current_info = noisy_env.step(action_from_agent)

        final_sensed_wd_from_obs = current_info["obs_sensed/farm/wd_current"]
        final_turbine_abs_heading = current_info["yaw angles agent"][0]

        print(
            f"  Current OBS (true): {current_info.get('obs_true/farm/wd_current', 'N/A'):.4f}°"
        )
        print(f"  Current OBS (sensed by agent): {final_sensed_wd_from_obs:.4f}°")
        print(
            f"  Agent's internally estimated WS/WD: {noisy_pywake_agent.wsp[0]:.4f} m/s, {noisy_pywake_agent.wdir[0]:.4f}°"
        )
        print(f"  Agent Commanded Action (scaled): {action_from_agent[0]:.4f}")
        print(f"  Current Turbine Absolute Heading: {final_turbine_abs_heading:.4f}°")
        print(f"  Environment's true wind direction: {noisy_env.base_env.wd}°")

    print("\n" + "=" * 50)
    print(f"FINAL STATE for {test_case_name}:")
    print(f"  Final Sensed WD: {final_sensed_wd_from_obs:.4f}°")
    print(f"  Final Turbine Absolute Heading: {final_turbine_abs_heading:.4f}°")
    print(f"  Agent's last internally estimated WD: {noisy_pywake_agent.wdir[0]:.4f}°")
    print("=" * 50)

    # --- 7. ASSERTIONS AFTER LOOP (Using values from the final step) ---

    # Normalize final sensed and turbine heading for robust comparison
    final_sensed_wd_from_obs_normalized = final_sensed_wd_from_obs % 360
    if final_sensed_wd_from_obs_normalized < 0:
        final_sensed_wd_from_obs_normalized += 360
    final_turbine_abs_heading_normalized = final_turbine_abs_heading % 360
    if final_turbine_abs_heading_normalized < 0:
        final_turbine_abs_heading_normalized += 360

    # --- 7.1. Assert Compensation Logic produces expected offset ---
    calculated_wd_error_final = final_sensed_wd_from_obs - noisy_env.base_env.wd

    # For ActionMethod="yaw", optimal_yaws from optimize() is 0.0 (relative to true wind).
    # yaw_goal = self.optimized_yaws - wd_error
    # This means the agent's target yaw angle (relative to wind) becomes 0 - (-180) = 180.
    expected_agent_target_yaw_angle = (
        noisy_pywake_agent.optimized_yaws - calculated_wd_error_final
    )  # e.g., 0 - (-180) = 180

    print("\n--- ASSERTION BREAKDOWN (Final Check) ---")
    print(f"  Calculated WD Error (Sensed - True): {calculated_wd_error_final:.4f}°")
    print(
        f"  Agent's Target Yaw Offset (from code logic): {expected_agent_target_yaw_angle:.4f}°"
    )
    print(
        f"  Expected Agent Target Yaw Offset (TRUE_WD - PERCEIVED_WD): {TRUE_GLOBAL_WD - AGENT_PERCEIVED_WD}°"
    )

    np.testing.assert_allclose(
        expected_agent_target_yaw_angle,
        (TRUE_GLOBAL_WD - AGENT_PERCEIVED_WD),
        atol=2.0,
        err_msg=f"[{test_case_name}] Agent's internal target yaw does NOT match expected compensation.",
    )
    print("ASSERTION PASSED: Compensation logic produces expected offset.")

    # --- 7.2. Assert Agent's Unscaled Action Leads to Expected Change ---
    # The action_from_agent is scaled ([-1, 1]) where 1 means yaw_step_env degrees.
    # We need to unscale this action to understand the commanded *change* in yaw.
    commanded_yaw_change_from_agent = (
        action_from_agent[0] * noisy_env.base_env.yaw_step_env
    )

    # The agent calculates `yaw_goal = optimal_yaws - wd_error`.
    # It wants to move from `base_env.current_yaw` towards `yaw_goal`.
    # The actual change it *tries* to make is clamped by `yaw_step_env`.
    expected_raw_change_needed = (
        expected_agent_target_yaw_angle - noisy_env.base_env.current_yaw[0]
    )
    expected_commanded_change = np.sign(expected_raw_change_needed) * min(
        abs(expected_raw_change_needed), noisy_env.base_env.yaw_step_env
    )

    print(
        f"  Agent's Commanded Yaw Change (unscaled action * yaw_step_env): {commanded_yaw_change_from_agent:.4f}°"
    )
    print(
        f"  Expected Commanded Change (clamped by yaw_step_env): {expected_commanded_change:.4f}°"
    )

    # Assert that the commanded change matches what we expect from the agent's logic
    np.testing.assert_allclose(
        commanded_yaw_change_from_agent,
        expected_commanded_change,
        atol=2.0,
        err_msg=f"[{test_case_name}] Agent's unscaled action does NOT command the expected yaw change.",
    )
    print(
        "ASSERTION PASSED: Agent's unscaled action correctly commands expected yaw change."
    )

    # --- 7.3. Assert Final Turbine Heading (After Environment Clipping) ---
    # This now reflects the *actual* change applied by the environment
    # We directly assert the final turbine heading against the high-level expectation.

    # Calculate the *true* expected final heading based on test parameters and clipping.
    # The agent's goal is to offset by (True WD - Perceived WD), which is 180.
    # It starts at 0, and has a large yaw_step_env (180), so it should reach this in one step.
    # Then it's clipped by the env's yaw_min/max.
    target_offset_from_compensation = (
        TRUE_GLOBAL_WD - AGENT_PERCEIVED_WD
    )  # Should be 180.0
    expected_final_heading_based_on_test_setup = np.clip(
        target_offset_from_compensation, yaw_min_cfg, yaw_max_cfg
    )

    print(
        f"  Expected Final Absolute Heading (from test goal and clipping): {expected_final_heading_based_on_test_setup:.4f}°"
    )

    np.testing.assert_allclose(
        final_turbine_abs_heading_normalized,
        expected_final_heading_based_on_test_setup,
        atol=2.0,
        err_msg=f"[{test_case_name}] Final turbine heading does NOT match expected value based on test goal.",
    )
    print(
        "ASSERTION PASSED: Final turbine heading matches expected value based on test goal."
    )

    # --- 7.4. Intuitive Assertions for "Turned Around" / "Clipped" Behavior ---
    # These become redundant with the refined 7.3 but are good for explicit clarity.
    if "small_yaw_range" in test_case_name:
        np.testing.assert_allclose(
            final_turbine_abs_heading_normalized,
            45.0,
            atol=0.1,
            err_msg=f"[{test_case_name}] Small yaw range: Expected clipped to +45 deg offset, got {final_turbine_abs_heading_normalized}",
        )
        print(
            f"[{test_case_name}] Turbine attempted to turn towards {expected_agent_target_yaw_angle}° but was CLIPPED to {final_turbine_abs_heading_normalized}° absolute."
        )

    elif "large_yaw_range" in test_case_name:
        np.testing.assert_allclose(
            final_turbine_abs_heading_normalized,
            180.0,
            atol=0.1,
            err_msg=f"[{test_case_name}] Large yaw range: Expected 180 deg offset, got {final_turbine_abs_heading_normalized}"
            " (Error is likely from DWM simulation dynamics or floating point precision that accumulates)",
        )
        print(
            f"[{test_case_name}] Turbine successfully turned to {final_turbine_abs_heading_normalized}° absolute."
        )

    # --- 8. CLEANUP ---
    noisy_env.close()
    base_farm_env.close()  # Close the base_farm_env as well
    print(f"======== END OF TEST: {test_case_name} ========\n")
