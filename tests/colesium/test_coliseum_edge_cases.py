# tests/conf2/test_coliseum_edge_cases.py

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.Agents import ConstantAgent


class TestColiseumEdgeCases:
    """
    This test suite targets edge cases, different parameter combinations,
    and uncovered methods in the Coliseum class using optimized fixtures.
    """

    def test_init_with_agent_list_and_default_labels(self, fast_farm_eval_factory):
        """
        Tests if Coliseum correctly assigns default labels when agents are passed as a list.
        Uses the fast factory to avoid slow environment setup.
        """
        agents_list = [ConstantAgent([10, 0]), ConstantAgent([0, 0])]
        coliseum = Coliseum(fast_farm_eval_factory, agents=agents_list)

        assert list(coliseum.agents.keys()) == ["Agent_0", "Agent_1"]
        assert coliseum.agent_names == ["Agent_0", "Agent_1"]

    def test_run_time_series_without_detailed_history(self, coliseum_instance):
        """Tests time series evaluation with save_detailed_history=False."""
        summary_df = coliseum_instance.run_time_series_evaluation(
            num_episodes=2, save_detailed_history=False
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert "episode" in summary_df.columns
        assert (
            not coliseum_instance.time_series_results
        ), "time_series_results should be empty."

    def test_run_grid_evaluation_with_netcdf_save(self, coliseum_instance, tmp_path):
        """Tests that the grid evaluation can save its results to a NetCDF file."""
        save_path = tmp_path / "grid_results.nc"

        coliseum_instance.run_wind_grid_evaluation(
            ws_min=8,
            ws_max=8,
            ws_step=1,
            wd_min=270,
            wd_max=270,
            wd_step=1,
            ti_points=1,
            save_netcdf=str(save_path),
        )

        assert os.path.exists(save_path), "NetCDF file was not created."
        loaded_ds = xr.open_dataset(save_path)
        assert "Steering_Agent_mean_reward" in loaded_ds.data_vars

    def test_run_grid_evaluation_with_default_bounds(self, coliseum_instance):
        """Tests that grid evaluation correctly uses the environment's default wind bounds."""
        dataset = coliseum_instance.run_wind_grid_evaluation(
            ws_step=4, wd_step=10, ti_points=1
        )

        # From conftest.py: ws is [8, 12] and wd is [265, 275].
        np.testing.assert_array_equal(dataset.coords["ws"], [8, 12])
        np.testing.assert_array_equal(dataset.coords["wd"], [265, 275])

    def test_plot_summary_comparison(self, coliseum_instance, monkeypatch):
        """Tests that the summary plotting function runs without error."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        coliseum_instance.run_time_series_evaluation(num_episodes=2)
        coliseum_instance.plot_summary_comparison(save_path="summary.png")

        mock_savefig.assert_called_once_with(
            "summary.png", dpi=150, bbox_inches="tight"
        )

    def test_get_summary_statistics(self, coliseum_instance):
        """Tests the get_summary_statistics method for correctness."""
        empty_stats = coliseum_instance.get_summary_statistics()
        assert empty_stats.empty, "Statistics should be empty before an evaluation."

        coliseum_instance.run_time_series_evaluation(num_episodes=5)
        stats_df = coliseum_instance.get_summary_statistics()

        assert isinstance(stats_df, pd.DataFrame)
        assert not stats_df.empty
        assert len(stats_df) == len(coliseum_instance.agent_names)

    def test_warning_on_time_series_without_site(self, capsys, fast_farm_eval_factory):
        """
        Tests that a warning is printed when running time series evaluation
        without a PyWake site configured for realistic wind sampling.
        """
        env_factory = fast_farm_eval_factory
        env = env_factory()
        env.sample_site = None  # Explicitly ensure no site is configured

        agent = ConstantAgent([0, 0])
        coliseum = Coliseum(lambda: env, agents={"NoOp": agent})

        coliseum.run_time_series_evaluation(num_episodes=1)

        captured = capsys.readouterr()
        assert "Warning: No sample_site configured" in captured.out

    def test_init_raises_error_for_mismatched_labels(self, fast_farm_eval_factory):
        """
        Covers: `raise ValueError("Number of agent_labels must match...")`

        Tests that a ValueError is raised if the number of agents in a list
        does not match the number of provided labels.
        """
        agents_list = [ConstantAgent([0, 0])]  # 1 agent
        agent_labels = ["Label_A", "Label_B"]  # 2 labels

        with pytest.raises(
            ValueError, match="Number of agent_labels must match number of agents"
        ):
            Coliseum(
                fast_farm_eval_factory, agents=agents_list, agent_labels=agent_labels
            )

    def test_init_with_custom_agent_labels(self, fast_farm_eval_factory):
        """
        Covers: `self.agent_names = agent_labels`

        Tests that custom labels are correctly assigned when provided with a list of agents.
        This confirms the successful path for the check above.
        """
        agents_list = [ConstantAgent([10, 0]), ConstantAgent([0, 0])]
        custom_labels = ["AgentAlpha", "AgentBeta"]

        coliseum = Coliseum(
            fast_farm_eval_factory, agents=agents_list, agent_labels=custom_labels
        )

        assert coliseum.agent_names == custom_labels
        assert list(coliseum.agents.keys()) == custom_labels

    def test_init_raises_error_for_invalid_agents_type(self, fast_farm_eval_factory):
        """
        Covers: `raise ValueError("agents must be a dictionary or list")`

        Tests that a ValueError is raised if the 'agents' argument is not a list or dict.
        """
        with pytest.raises(ValueError, match="agents must be a dictionary or list"):
            Coliseum(fast_farm_eval_factory, agents="this is not a valid type")

    @pytest.mark.parametrize("empty_agents", [{}, []])
    def test_init_raises_error_for_empty_agents(
        self, fast_farm_eval_factory, empty_agents
    ):
        """
        Covers: `raise ValueError("At least one agent must be provided")`

        Tests that a ValueError is raised if the 'agents' list or dict is empty.
        """
        with pytest.raises(ValueError, match="At least one agent must be provided"):
            Coliseum(fast_farm_eval_factory, agents=empty_agents)

    def test_update_wind_warning_in_timeseries_eval(self, capsys):
        """
        Covers the warning print in `_run_single_episode_with_history`.

        Tests that a warning is printed if an agent has an `update_wind` method
        but the environment is missing the required attributes (e.g., 'ti').
        """

        class AgentWithUpdateWind:
            def predict(self, obs, deterministic):
                return np.array([0]), None

            def update_wind(self, wind_speed, wind_direction, TI):
                pass

        # FIX: Add 'close' and 'sample_site' to the spec list for the mock.
        spec_list = [
            "ws",
            "wd",
            "reset",
            "step",
            "n_passthrough",
            "close",
            "sample_site",
        ]
        mock_env = MagicMock(spec=spec_list)

        # Define the attributes on the mock object itself
        mock_env.ws = 10.0
        mock_env.wd = 270.0
        mock_env.sample_site = None
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (
            np.zeros(10),
            0,
            True,
            False,
            {},
        )  # Ensure episode terminates

        def dummy_factory():
            return mock_env

        coliseum = Coliseum(dummy_factory, agents={"DummyAgent": AgentWithUpdateWind()})

        coliseum.run_time_series_evaluation(num_episodes=1)

        captured = capsys.readouterr()
        assert (
            "Warning: Agent has 'update_wind' method, but env is missing"
            in captured.out
        )

    def test_update_wind_warning_in_grid_eval(self, capsys):
        """
        Covers the warning print in `_run_single_episode_summary`.

        Tests the same warning condition but ensures it's also triggered
        via the wind grid evaluation path.
        """

        class AgentWithUpdateWind:
            def predict(self, obs, deterministic):
                return np.array([0]), None

            def update_wind(self, wind_speed, wind_direction, TI):
                pass

        # The spec now correctly lists all attributes and methods that will be called.
        spec_list = [
            "ws",
            "wd",
            "reset",
            "step",
            "set_wind_vals",
            "n_passthrough",
            "close",
            "wd_min",
            "wd_max",
            "ws_min",
            "ws_max",
            "TI_min",
            "TI_max",
        ]
        mock_env = MagicMock(spec=spec_list)

        # FIX: Assign concrete numerical values to the mock's boundary attributes.
        # This prevents the TypeError during string formatting.
        mock_env.wd_min = 260
        mock_env.wd_max = 280
        mock_env.ws_min = 8
        mock_env.ws_max = 12
        mock_env.TI_min = 0.05
        mock_env.TI_max = 0.15

        # Define return values for the mock's methods
        mock_env.set_wind_vals.return_value = None
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 0, True, False, {})

        def dummy_factory():
            return mock_env

        coliseum = Coliseum(dummy_factory, agents={"DummyAgent": AgentWithUpdateWind()})

        # Run grid evaluation, which will trigger the warning logic.
        coliseum.run_wind_grid_evaluation(
            ws_min=10,
            ws_max=10,
            ws_step=1,
            wd_min=270,
            wd_max=270,
            wd_step=1,
            ti_points=1,
        )

        captured = capsys.readouterr()
        assert (
            "Warning: Agent has 'update_wind' method, but env is missing"
            in captured.out
        )

    def test_plot_time_series_prints_warning_if_no_data(
        self, coliseum_instance, capsys
    ):
        """
        Covers the check for non-existent time series data before plotting.

        Tests that a warning is printed if plot_time_series_comparison is called
        before run_time_series_evaluation has been executed.
        """
        # On a fresh Coliseum instance from the fixture, self.time_series_results is None.
        # We directly call the plotting function without generating data first.
        coliseum_instance.plot_time_series_comparison()

        # Capture the output and check for the specific warning message.
        captured = capsys.readouterr()
        expected_warning = (
            "No time series data available. Run time series evaluation first."
        )
        assert expected_warning in captured.out

    def test_plot_summary_prints_warning_if_no_data(self, coliseum_instance, capsys):
        """
        Covers the check for non-existent results data before plotting the summary.

        Tests that a warning is printed if plot_summary_comparison is called
        before an evaluation has been run.
        """
        # A fresh Coliseum instance from the fixture has self.results = None
        # We call the plotting function immediately without running an evaluation
        coliseum_instance.plot_summary_comparison()

        # We then check that the expected warning was printed to the console
        captured = capsys.readouterr()
        expected_warning = "No results to plot. Run an evaluation first."
        assert expected_warning in captured.out
