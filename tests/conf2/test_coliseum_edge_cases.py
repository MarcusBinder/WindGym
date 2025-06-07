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
            num_episodes=2,
            save_detailed_history=False
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert "episode" in summary_df.columns
        assert not coliseum_instance.time_series_results, "time_series_results should be empty."

    def test_run_grid_evaluation_with_netcdf_save(self, coliseum_instance, tmp_path):
        """Tests that the grid evaluation can save its results to a NetCDF file."""
        save_path = tmp_path / "grid_results.nc"
        
        coliseum_instance.run_wind_grid_evaluation(
            ws_min=8, ws_max=8, ws_step=1,
            wd_min=270, wd_max=270, wd_step=1,
            ti_points=1,
            save_netcdf=str(save_path)
        )
        
        assert os.path.exists(save_path), "NetCDF file was not created."
        loaded_ds = xr.open_dataset(save_path)
        assert "Steering_Agent_mean_reward" in loaded_ds.data_vars

    def test_run_grid_evaluation_with_default_bounds(self, coliseum_instance):
        """Tests that grid evaluation correctly uses the environment's default wind bounds."""
        dataset = coliseum_instance.run_wind_grid_evaluation(ws_step=4, wd_step=10, ti_points=1)
        
        # From conftest.py: ws is [8, 12] and wd is [265, 275].
        np.testing.assert_array_equal(dataset.coords['ws'], [8, 12])
        np.testing.assert_array_equal(dataset.coords['wd'], [265, 275])

    def test_plot_summary_comparison(self, coliseum_instance, monkeypatch):
        """Tests that the summary plotting function runs without error."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)
        
        coliseum_instance.run_time_series_evaluation(num_episodes=2)
        coliseum_instance.plot_summary_comparison(save_path="summary.png")
        
        mock_savefig.assert_called_once_with("summary.png", dpi=150, bbox_inches='tight')

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
        env.sample_site = None # Explicitly ensure no site is configured
        
        agent = ConstantAgent([0, 0])
        coliseum = Coliseum(lambda: env, agents={"NoOp": agent})

        coliseum.run_time_series_evaluation(num_episodes=1)
        
        captured = capsys.readouterr()
        assert "Warning: No sample_site configured" in captured.out
