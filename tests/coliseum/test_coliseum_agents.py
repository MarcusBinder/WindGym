# test_coliseum_agents.py

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os
from unittest.mock import patch

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.FarmEval import FarmEval
from WindGym.Agents import PyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

# --- Test Class for Advanced Coliseum Features ---


class TestColiseumAdvanced:
    """Tests advanced and specific logic within the Coliseum evaluator."""

    def test_create_env_factory_with_site(self, temp_yaml_file):
        """Covers the static helper method for creating a site-based factory."""
        mock_site = Hornsrev1Site()
        mock_kwargs = {
            "x_pos": [0],
            "y_pos": [0],
            "turbine": V80(),
            "yaml_path": temp_yaml_file,
            "n_passthrough": 0.05,
            "burn_in_passthroughs": 0.01,
            "turbtype": "None",
        }

        factory = Coliseum.create_env_factory_with_site(
            env_class=FarmEval, site=mock_site, **mock_kwargs
        )

        env = factory()

        assert isinstance(env, FarmEval)
        assert env.sample_site is mock_site
        env.close()

    def test_pywake_agent_update_wind_is_called(self, temp_yaml_file):
        """
        Tests that for agents with an `update_wind` method (like PyWakeAgent),
        the Coliseum calls it with the correct wind conditions during grid evaluation.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

        # 1. Create a real PyWakeAgent instance
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        # 2. Create the Coliseum instance with this agent
        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                yaml_path=temp_yaml_file,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
            )

        coliseum = Coliseum(
            env_factory, agents={"PyWake": pywake_agent}, n_passthrough=0.1
        )

        # 3. "Spy" on the agent's update_wind method
        with patch.object(
            pywake_agent, "update_wind", wraps=pywake_agent.update_wind
        ) as mock_update_method:
            # 4. Run a single-point grid evaluation
            ws_test, wd_test, ti_test = 10, 270, 0.08

            coliseum.run_wind_grid_evaluation(
                ws_min=ws_test,
                ws_max=ws_test,
                ws_step=1,
                wd_min=wd_test,
                wd_max=wd_test,
                wd_step=1,
                ti_min=ti_test,
                ti_max=ti_test,
                ti_points=1,
            )

            # 5. Assert that the spy was called correctly
            mock_update_method.assert_called_once()

            # FIX: Use keyword arguments in the assertion to match the actual call
            mock_update_method.assert_called_with(
                wind_speed=ws_test, wind_direction=wd_test, TI=ti_test
            )


# --- Integration Tests for PyWakeAgent ---


@pytest.fixture
def temp_yaml_file_integration():
    """Provides a path to a temporary, valid YAML file for the duration of a test."""
    yaml_config = """
    yaw_init: "Zeros"
    noise: "None"
    BaseController: "Local"
    ActionMethod: "yaw"
    farm: {yaw_min: -30, yaw_max: 30}
    wind: {ws_min: 10, ws_max: 10, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
    act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
    power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
    mes_level: {turb_ws: True, turb_wd: True, turb_TI: True, turb_power: False, farm_ws: False, farm_wd: False, farm_TI: False, farm_power: False}
    ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 50, ws_window_length: 1}
    wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
    yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
    power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_file:
        tmp_file.write(yaml_config)
        filepath = tmp_file.name

    yield filepath

    os.remove(filepath)


class TestColiseumIntegration:
    """
    Tests the Coliseum framework with real agents and environments,
    focusing on end-to-end functionality without mocks.
    """

    def test_create_env_factory_with_site(self, temp_yaml_file_integration):
        """Covers the static helper method for creating a site-based factory."""
        mock_site = Hornsrev1Site()
        mock_kwargs = {
            "x_pos": [0],
            "y_pos": [0],
            "turbine": V80(),
            "yaml_path": temp_yaml_file_integration,
            "n_passthrough": 0.05,
            "burn_in_passthroughs": 0.01,
            "turbtype": "None",
        }

        factory = Coliseum.create_env_factory_with_site(
            env_class=FarmEval, site=mock_site, **mock_kwargs
        )

        env = factory()

        assert isinstance(env, FarmEval)
        assert env.sample_site is mock_site
        env.close()

    def test_pywake_agent_in_time_series_evaluation(self, temp_yaml_file_integration):
        """
        Tests that a PyWakeAgent can run a full time-series evaluation and that its
        internal state is correctly updated by the Coliseum framework.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        # The env_factory will create an env with ws=10, wd=270 from the YAML
        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                yaml_path=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.001,
            )

        coliseum = Coliseum(
            env_factory, agents={"PyWake": pywake_agent}, n_passthrough=0.1
        )

        # Run a single episode. This will trigger the agent's update_wind method.
        coliseum.run_time_series_evaluation(num_episodes=1, seed=42)

        # After the run, the agent's internal state should match the environment's.
        # This is a direct test that the synchronization happened.
        assert pywake_agent.wsp == [10]
        assert pywake_agent.wdir == [270]
        assert pywake_agent.TI == 0.07

        # Also, check that the agent has calculated a valid optimization result.
        assert pywake_agent.optimized is True
        assert pywake_agent.optimized_yaws is not None
        # For this specific case (2 turbines at 270 deg), the optimal yaw for the upstream
        # turbine should be non-zero to steer the wake.
        assert pywake_agent.optimized_yaws[0] != 0

    def test_pywake_agent_synchronization(self, temp_yaml_file_integration):
        """
        Tests that a PyWakeAgent's internal state is correctly updated by the Coliseum
        framework before a time-series episode run.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                yaml_path=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
            )

        coliseum = Coliseum(
            env_factory, agents={"PyWake": pywake_agent}, n_passthrough=0.1
        )

        coliseum.run_time_series_evaluation(num_episodes=1, seed=42)

        assert pywake_agent.wsp == [10]
        assert pywake_agent.wdir == [270]
        assert pywake_agent.TI == 0.07
        assert pywake_agent.optimized is True

    def test_pywake_agent_in_grid_evaluation(self, temp_yaml_file_integration):
        """
        Tests that the PyWakeAgent's state is correctly updated for each point
        in a grid evaluation.
        """

        class SpyPyWakeAgent(PyWakeAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.update_calls = []

            def update_wind(self, wind_speed, wind_direction, TI):
                self.update_calls.append(
                    {"ws": wind_speed, "wd": wind_direction, "ti": TI}
                )
                super().update_wind(wind_speed, wind_direction, TI)

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        spy_agent = SpyPyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                yaml_path=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.01,
            )

        coliseum = Coliseum(
            env_factory,
            agents={"SpyWake": spy_agent},
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        ws_grid = [9, 11]
        wd_grid = [265, 275]
        ti_val = 0.07

        coliseum.run_wind_grid_evaluation(
            ws_min=min(ws_grid),
            ws_max=max(ws_grid),
            ws_step=2,
            wd_min=min(wd_grid),
            wd_max=max(wd_grid),
            wd_step=10,
            ti_min=ti_val,
            ti_max=ti_val,
            ti_points=1,
        )

        assert len(spy_agent.update_calls) == 4

        expected_calls = [
            {"ws": 9.0, "wd": 265.0, "ti": ti_val},
            {"ws": 11.0, "wd": 265.0, "ti": ti_val},
            {"ws": 9.0, "wd": 275.0, "ti": ti_val},
            {"ws": 11.0, "wd": 275.0, "ti": ti_val},
        ]
        assert spy_agent.update_calls == expected_calls
