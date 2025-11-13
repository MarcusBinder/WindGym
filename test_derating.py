#!/usr/bin/env python3
"""
Test script for derating functionality in WindGym.
Tests three scenarios: yaw only, derate only, and both.
"""

import numpy as np
import yaml
from pathlib import Path
from py_wake.examples.data.iea37 import IEA37_WindTurbines
from WindGym import WindFarmEnv


def create_test_config(enable_yaw=True, enable_derate=False):
    """Create a test configuration."""
    config = {
        "yaw_init": "Random",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "wind",
        "Track_power": False,
        "enable_yaw": enable_yaw,
        "enable_derate": enable_derate,
        "DerateMethod": "wind",
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
            "derate_min": 0.0,
            "derate_max": 1.0,
            "xDist": 5,
            "yDist": 5,
            "nx": 2,
            "ny": 2,
        },
        "wind": {
            "ws_min": 8,
            "ws_max": 12,
            "TI_min": 0.05,
            "TI_max": 0.10,
            "wd_min": 260,
            "wd_max": 280,
        },
        "act_pen": {
            "action_penalty": 0.0,
            "action_penalty_type": "Change",
        },
        "power_def": {
            "Power_reward": "Baseline",
            "Power_avg": 10,
            "Power_scaling": 1.0,
        },
        "mes_level": {
            "turb_ws": True,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 1,
            "ws_history_length": 25,
            "ws_window_length": 25,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": False,
            "wd_history_N": 1,
            "wd_history_length": 20,
            "wd_window_length": 20,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 1,
            "yaw_history_length": 10,
            "yaw_window_length": 10,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 10,
            "power_window_length": 10,
        },
    }
    return config


def test_scenario(name, enable_yaw, enable_derate, backend="pywake"):
    """Test a specific action configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Backend: {backend}")
    print(f"enable_yaw: {enable_yaw}, enable_derate: {enable_derate}")
    print(f"{'='*60}")

    # Create config
    config = create_test_config(enable_yaw=enable_yaw, enable_derate=enable_derate)

    # Create turbine
    turbine = IEA37_WindTurbines()

    # Create environment
    try:
        env = WindFarmEnv(
            turbine=turbine,
            x_pos=np.array([0, 1000]),
            y_pos=np.array([0, 0]),
            config=config,
            backend=backend,
            seed=42,
        )

        print(f"✓ Environment created successfully")
        print(f"  Action space shape: {env.action_space.shape}")
        print(f"  Expected shape: ({2 * env.act_var},)")
        print(f"  act_var: {env.act_var}")

        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")

        # Test action
        n_actions = env.action_space.shape[0]
        action = np.zeros(n_actions)

        if enable_yaw and enable_derate:
            # Set yaw to max and derate to 50%
            action[:2] = 1.0  # Max yaw
            action[2:] = 0.0  # 50% power (maps to 0.5 after scaling)
            print(f"  Action: yaw=[1.0, 1.0], derate=[0.0, 0.0] (50% power)")
        elif enable_yaw:
            action[:] = 1.0  # Max yaw
            print(f"  Action: yaw=[1.0, 1.0]")
        elif enable_derate:
            action[:] = 0.0  # 50% power
            print(f"  Action: derate=[0.0, 0.0] (50% power)")

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step executed successfully")
        print(f"  Reward: {reward:.2f}")
        print(f"  Yaw angles: {env.fs.windTurbines.yaw}")
        print(f"  Derating: {env.fs.windTurbines.derate}")
        print(f"  Power: {env.fs.windTurbines.power()}")

        # Take another step to verify incremental changes work
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Second step executed successfully")
        print(f"  Yaw angles: {env.fs.windTurbines.yaw}")
        print(f"  Derating: {env.fs.windTurbines.derate}")
        print(f"  Power: {env.fs.windTurbines.power()}")

        print(f"\n✓✓✓ {name} - PASSED ✓✓✓")
        return True

    except Exception as e:
        print(f"\n✗✗✗ {name} - FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("WindGym Derating Feature Test Suite")
    print("="*60)

    results = {}

    # Test with PyWake backend
    backend = "pywake"

    # Test 1: Yaw only (backward compatibility)
    results["Yaw Only"] = test_scenario(
        "Yaw Only (Backward Compatibility)",
        enable_yaw=True,
        enable_derate=False,
        backend=backend,
    )

    # Test 2: Derate only
    results["Derate Only"] = test_scenario(
        "Derate Only",
        enable_yaw=False,
        enable_derate=True,
        backend=backend,
    )

    # Test 3: Both yaw and derate
    results["Both Yaw and Derate"] = test_scenario(
        "Both Yaw and Derate",
        enable_yaw=True,
        enable_derate=True,
        backend=backend,
    )

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
