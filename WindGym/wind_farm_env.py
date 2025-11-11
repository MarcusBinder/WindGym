from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os
import gc
import socket
import shutil
import math
from pathlib import Path


# Dynamiks imports
from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion
from dynamiks.wind_turbines import PyWakeWindTurbines
from dynamiks.views import XYView

from IPython import display

# WindGym imports
from . import utils
from .core.mes_class import FarmMes
from .core.reward_calculator import RewardCalculator
from .core.wind_manager import WindManager
from .core.turbulence_manager import TurbulenceManager
from .core.renderer import WindFarmRenderer
from .core.baseline_manager import BaselineManager
from .core.probe_manager import ProbeManager
from .config_schema import EnvConfig, SimulationConfig, ScalingConfig

from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from collections import deque, defaultdict
import yaml
from dynamiks.wind_turbines.hawc2_windturbine import HAWC2WindTurbines
from dynamiks.dwm.particle_motion_models import CutOffFrq

# For live plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from WindGym.core.wind_probe import WindProbe


CutOffFrqLio2021 = CutOffFrq(4)

"""
This is the base for the wind farm environment. This is where the magic happens.
For now it only supports the PyWakeWindTurbines, but it should be easy to expand to other types of turbines.
"""


# TODO make it so that the turbines can be other then a square grid
# TODO thrust coefficient control
# TODO for now I have just hardcoded this scaling value (1 and 25 for the wind_speed min and max). This is beacuse the wind speed is chosen from the normal distribution, but becasue of the wakes and the turbulence, we canhave cases where we go above or below these values.


class WindFarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        config=None,
        backend: str = "dynamiks",
        sim_config: Optional[SimulationConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        # Turbulence parameters
        TurbBox="Default",
        turbtype="Random",
        # Optional overrides
        Baseline_comp=False,
        yaw_init=None,
        # Simulation control
        render_mode=None,
        seed=None,
        yaw_step_sim=1,
        yaw_step_env=None,
        fill_window=True,
        sample_site=None,
        HTC_path=None,
        reset_init=True,
        cleanup_on_time_limit: bool = True,
        wd_function=None,
        # Deprecated parameters for backward compatibility
        n_passthrough=None,
        ws_scaling_min=None,
        ws_scaling_max=None,
        wd_scaling_min=None,
        wd_scaling_max=None,
        ti_scaling_min=None,
        ti_scaling_max=None,
        yaw_scaling_min=None,
        yaw_scaling_max=None,
        dt_sim=None,
        dt_env=None,
        burn_in_passthroughs=None,
        max_turb_move=None,
        **kwargs,
    ):
        """
        WindGym environment for wind farm control using reinforcement learning.

        Args:
            turbine: PyWake wind turbine model to use
            x_pos: Array of turbine x-coordinates
            y_pos: Array of turbine y-coordinates
            config: Environment configuration (required). Can be:
                - Path to YAML file
                - YAML string
                - Dictionary
                - EnvConfig object
            backend: Simulation backend ("dynamiks" or "pywake")
            sim_config: Simulation configuration (dt_sim, dt_env, n_passthrough, etc.)
                If None, uses defaults or values from deprecated parameters
            scaling_config: Observation/action space scaling configuration
                If None, uses defaults or values from deprecated parameters
            TurbBox: Path to turbulence box files or "Default"
            turbtype: Turbulence type ("Random", "MannLoad", "MannGenerate", etc.)
            Baseline_comp: Whether to run baseline controller comparison
            yaw_init: Override for yaw initialization method
            render_mode: Rendering mode (None, "human", or "rgb_array")
            seed: Random seed for reproducibility
            yaw_step_sim: Degrees the yaw can change per simulation step
            yaw_step_env: Degrees the yaw can change per environment step
            fill_window: Whether to fill measurement history at reset
            sample_site: PyWake site object for wind sampling
            HTC_path: Path to HAWC2 high-fidelity turbine model
            reset_init: Whether to call reset() during initialization
            cleanup_on_time_limit: Whether to cleanup HAWC2 files on episode end
            wd_function: Custom function for wind direction (takes timestep, returns WD)

        Deprecated (use sim_config instead):
            n_passthrough, dt_sim, dt_env, burn_in_passthroughs, max_turb_move

        Deprecated (use scaling_config instead):
            ws_scaling_min, ws_scaling_max, wd_scaling_min, wd_scaling_max,
            ti_scaling_min, ti_scaling_max, yaw_scaling_min, yaw_scaling_max
        """
        # Store all initialization parameters for debugging
        self.kwargs = locals()
        del self.kwargs["self"]

        # Validate backend
        self.backend = backend.lower().strip()
        if self.backend not in {"dynamiks", "pywake"}:
            raise ValueError("backend must be 'dynamiks' or 'pywake'")

        # Validate positions
        if len(x_pos) != len(y_pos):
            raise ValueError("x_pos and y_pos must be the same length")

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_turb = len(x_pos)
        self.turbine = turbine

        # --- Handle configuration objects ---
        # Create simulation config from either sim_config or deprecated parameters
        if sim_config is None:
            sim_config = SimulationConfig(
                dt_sim=dt_sim if dt_sim is not None else 1.0,
                dt_env=dt_env if dt_env is not None else 1.0,
                n_passthrough=n_passthrough if n_passthrough is not None else 5,
                burn_in_passthroughs=(
                    burn_in_passthroughs if burn_in_passthroughs is not None else 2
                ),
                max_turb_move=max_turb_move if max_turb_move is not None else 2.0,
            )
        self.sim_config = sim_config

        # Create scaling config from either scaling_config or deprecated parameters
        if scaling_config is None:
            scaling_config = ScalingConfig(
                ws_min=ws_scaling_min if ws_scaling_min is not None else 0.0,
                ws_max=ws_scaling_max if ws_scaling_max is not None else 30.0,
                wd_min=wd_scaling_min if wd_scaling_min is not None else 0.0,
                wd_max=wd_scaling_max if wd_scaling_max is not None else 360.0,
                ti_min=ti_scaling_min if ti_scaling_min is not None else 0.0,
                ti_max=ti_scaling_max if ti_scaling_max is not None else 1.0,
                yaw_min=yaw_scaling_min if yaw_scaling_min is not None else -45.0,
                yaw_max=yaw_scaling_max if yaw_scaling_max is not None else 45.0,
            )
        self.scaling_config = scaling_config

        # Extract commonly used values for backward compatibility
        self.dt = self.sim_config.dt_sim
        self.dt_sim = self.sim_config.dt_sim
        self.dt_env = self.sim_config.dt_env
        self.n_passthrough = self.sim_config.n_passthrough
        self.burn_in_passthroughs = self.sim_config.burn_in_passthroughs
        self.max_turb_move = self.sim_config.max_turb_move
        self.sim_steps_per_env_step = int(self.dt_env / self.dt_sim)

        # Validate pywake backend constraint
        if self.backend == "pywake" and self.dt_env != self.dt_sim:
            raise ValueError(
                "When using pywake as backend, dt_env must be equal to dt_sim"
            )

        # Store scaling values for backward compatibility
        self.ws_scaling_min = self.scaling_config.ws_min
        self.ws_scaling_max = self.scaling_config.ws_max
        self.wd_scaling_min = self.scaling_config.wd_min
        self.wd_scaling_max = self.scaling_config.wd_max
        self.ti_scaling_min = self.scaling_config.ti_min
        self.ti_scaling_max = self.scaling_config.ti_max
        self.yaw_scaling_min = self.scaling_config.yaw_min
        self.yaw_scaling_max = self.scaling_config.yaw_max

        # --- Load and validate environment configuration ---
        self.env_config = self._normalize_config_input(config)
        self._apply_config(self.env_config)

        # --- Initialize other parameters ---
        self.wd_function = wd_function
        self.wts = None
        self.wts_baseline = None
        self.cleanup_on_time_limit = cleanup_on_time_limit
        self.power_setpoint = 0.0
        self.act_var = 1  # Number of actions per turbine (currently just yaw)
        self.HTC_path = HTC_path
        self.fill_window = fill_window
        self.delay = self.dt_env
        self.sample_site = sample_site
        self.yaw_start = 15.0
        self.maxturbpower = max(turbine.power(np.arange(10, 25, 1)))
        self.baseline_wakes = True
        self.d_particle = 0.2
        self.n_particles = None
        self.temporal_filter = CutOffFrqLio2021
        self.turbtype = turbtype
        self.yaw_step_sim = yaw_step_sim
        self.seed = seed
        self.TurbBox = TurbBox
        self.time_max = 0
        self.timestep = 0
        self.yaw_initial = [0]
        self.n_probes_per_turb = None

        # Calculate yaw_step_env
        if yaw_step_env is None:
            self.yaw_step_env = yaw_step_sim * self.sim_steps_per_env_step
        else:
            self.yaw_step_env = yaw_step_env

        # --- Initialize yaw initializer ---
        yaw_init_method = yaw_init if yaw_init is not None else self.yaw_init
        self._yaw_init = self._create_yaw_initializer(yaw_init_method)

        # --- Initialize manager components ---
        self._init_managers(turbine, sample_site, Baseline_comp, HTC_path, render_mode)

        # --- Initialize measurements ---
        self._init_farm_mes()

        # Calculate history length
        self.hist_max = max(self.power_avg, self.farm_measurements.max_hist())

        # Determine steps on reset
        if self.fill_window is True:
            self.steps_on_reset = self.hist_max
        elif isinstance(self.fill_window, int) and self.fill_window >= 1:
            self.steps_on_reset = min(self.fill_window, self.hist_max)
        elif self.fill_window is False:
            self.steps_on_reset = 1
        else:
            raise ValueError("fill_window must be True, False, or a positive integer")

        # --- Setup spaces ---
        self.D = turbine.diameter()
        self.obs_var = self.farm_measurements.observed_variables()
        self._init_spaces()

        # --- Initialize environment ---
        if reset_init:
            self.reset(seed=seed)

        # --- Setup rendering ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode is not None:
            self.init_render()

    def _init_managers(
        self,
        turbine,
        sample_site,
        Baseline_comp: bool,
        HTC_path: Optional[str],
        render_mode: Optional[str],
    ) -> None:
        """
        Initialize all manager components in a separate method for clarity.

        Args:
            turbine: Wind turbine model
            sample_site: PyWake site for wind sampling
            Baseline_comp: Whether to use baseline comparison
            HTC_path: Path to HAWC2 high-fidelity model
            render_mode: Rendering mode
        """
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            power_reward_type=self.power_reward,
            track_power=self.Track_power,
            power_scaling=self.Power_scaling,
            action_penalty=self.action_penalty,
            action_penalty_type=self.action_penalty_type,
            power_window_size=self.power_avg,
        )

        # Initialize wind manager
        self.wind_manager = WindManager(
            ws_min=self.ws_inflow_min,
            ws_max=self.ws_inflow_max,
            wd_min=self.wd_inflow_min,
            wd_max=self.wd_inflow_max,
            ti_min=self.TI_inflow_min,
            ti_max=self.TI_inflow_max,
            sample_site=sample_site,
        )

        # Initialize turbulence manager
        self.turbulence_manager = TurbulenceManager(
            turbulence_type=self.turbtype,
            turbulence_box_path=self.TurbBox,
            max_turb_move=self.max_turb_move,
        )
        self.TF_files = self.turbulence_manager.turbulence_files

        # Initialize renderer
        self.renderer = WindFarmRenderer(render_mode=render_mode)

        # Determine if baseline is needed
        if self.power_reward == "Baseline" or Baseline_comp:
            self.Baseline_comp = True
        else:
            self.Baseline_comp = False

        # Initialize baseline manager if needed
        self.baseline_manager = None
        if self.Baseline_comp:
            self.baseline_manager = BaselineManager(
                baseline_controller_type=self.BaseController,
                x_pos=self.x_pos,
                y_pos=self.y_pos,
                turbine=turbine,
                yaw_max=self.yaw_max,
                yaw_min=self.yaw_min,
                yaw_step_env=self.yaw_step_env,
                yaw_step_sim=self.yaw_step_sim,
                htc_path=HTC_path,
            )

    def _create_yaw_initializer(self, method: str):
        """
        Factory method for creating yaw initialization functions.

        Args:
            method: Initialization method ("Random", "Defined", or default to zeros)

        Returns:
            Callable that initializes yaw angles
        """
        if method == "Random":
            return lambda **kwargs: utils.randoms_uniform(
                self.np_random, kwargs["min_val"], kwargs["max_val"], kwargs["n"]
            )
        elif method == "Defined":
            return lambda **kwargs: utils.defined_yaw(kwargs["yaws"], self.n_turb)
        else:
            return lambda **kwargs: utils.return_zeros(kwargs["n"])

    def _init_wts(self):
        """
        Initialize the wind turbines.
        If the HTC path is given, then use hawc2 turbines, else use pywake turbines.
        Also is we have a baseline, then set that up also
        """
        if self.wts is not None:
            self.wts = None
        if self.wts_baseline is not None:
            self.wts_baseline = None

        if self.HTC_path is not None:  # pragma: no cover
            # TODO HTC stuff is not covered by the tests atm
            # If we have a high fidelity turbine model, then we need to load it in

            # We need to make a unique string, such that the results file doenst get overwritten
            node_string = socket.gethostname().split(".")[0]
            name_string = f"{node_string}_{self.wd:.2f}_{self.ws:.2f}_{self.ti:.2f}_{self.np_random.integers(low=0, high=45000)}"
            name_string = name_string.replace(".", "p")

            self.wts = HAWC2WindTurbines(
                x=self.x_pos,
                y=self.y_pos,
                htc_lst=[self.HTC_path],
                case_name=name_string,  # subfolder name in the htc, res and log folders
                suppress_output=True,  # don't show hawc2 output in console
            )
            # Add the yaw sensor, but because the only keyword does not work with h2lib, we add another layer that then only returns the first values of them.
            self.wts.add_sensor(
                name="yaw_getter",
                getter="constraint bearing2 yaw_rot 1 only 1;",  #
                expose=False,
                ext_lst=["angle", "speed"],
            )
            self.wts.add_sensor(
                "yaw",
                getter=lambda wt: np.rad2deg(wt.sensors.yaw_getter[:, 0]),
                setter=lambda wt, value: wt.h2.set_variable_sensor_value(
                    1, np.deg2rad(value).tolist()
                ),
                expose=True,
            )
        else:  # If we have no HTC path, use the pywake turbine
            self.wts = PyWakeWindTurbines(
                x=self.x_pos,
                y=self.y_pos,  # x and y position of two wind turbines
                windTurbine=self.turbine,
            )

        # Initialize baseline turbines if needed
        if self.Baseline_comp and self.baseline_manager is not None:
            # Pass name_string if we have it (only created for HAWC2 turbines)
            baseline_name = name_string if self.HTC_path is not None else None
            self.wts_baseline = self.baseline_manager.initialize_baseline_turbines(
                name_string=baseline_name
            )
        else:
            self.wts_baseline = None


    def _normalize_config_input(self, config) -> EnvConfig:
        """
        Normalizes the config input to an EnvConfig object with validation.

        Args:
            config: Configuration input (dict, YAML string, file path, or EnvConfig)

        Returns:
            Validated EnvConfig object

        Raises:
            ValueError: If config is None or invalid
            TypeError: If config is not a supported type
        """
        if config is None:
            raise ValueError(
                "A configuration must be provided via the `config` argument."
            )

        # If already an EnvConfig, return it
        if isinstance(config, EnvConfig):
            self.yaml_path = None
            return config

        # Load config dict from various sources
        config_dict = None

        if isinstance(config, dict):
            self.yaml_path = None
            config_dict = config
        elif isinstance(config, (str, Path)):
            config_str = str(config)
            if os.path.exists(config_str):
                # Load from file
                with open(config_str, "r") as f:
                    self.yaml_path = config_str
                    config_dict = yaml.safe_load(f) or {}
            else:
                # Parse as YAML string
                self.yaml_path = None
                config_dict = yaml.safe_load(str(config)) or {}
        else:
            raise TypeError(
                f"`config` must be a dict, YAML string, path to YAML file, or EnvConfig object. "
                f"Got {type(config)}"
            )

        # Parse dict into validated EnvConfig object
        try:
            return EnvConfig.from_dict(config_dict)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(
                f"Invalid configuration: {e}\n"
                f"Please check your configuration file/dict against the schema requirements."
            ) from e

    def _apply_config(self, config: EnvConfig) -> None:
        """
        Applies the validated EnvConfig to instance attributes.

        Args:
            config: Validated EnvConfig object
        """
        # Top-level fields
        self.yaw_init = config.yaw_init
        self.BaseController = config.BaseController
        self.ActionMethod = config.ActionMethod
        self.Track_power = config.Track_power

        # Farm configuration
        self.yaw_min = config.farm.yaw_min
        self.yaw_max = config.farm.yaw_max

        # Wind configuration
        self.ws_inflow_min = config.wind.ws_min
        self.ws_inflow_max = config.wind.ws_max
        self.TI_inflow_min = config.wind.TI_min
        self.TI_inflow_max = config.wind.TI_max
        self.wd_inflow_min = config.wind.wd_min
        self.wd_inflow_max = config.wind.wd_max

        # Action penalty configuration
        self.action_penalty = config.act_pen.action_penalty
        self.action_penalty_type = config.act_pen.action_penalty_type

        # Power reward configuration
        self.Power_scaling = config.power_def.Power_scaling
        self.power_avg = config.power_def.Power_avg
        self.power_reward = config.power_def.Power_reward

        # Measurement level configuration
        self.ti_sample_count = config.mes_level.ti_sample_count

        # Store measurement level dict for compatibility
        self.mes_level = {
            "turb_ws": config.mes_level.turb_ws,
            "turb_wd": config.mes_level.turb_wd,
            "turb_TI": config.mes_level.turb_TI,
            "turb_power": config.mes_level.turb_power,
            "farm_ws": config.mes_level.farm_ws,
            "farm_wd": config.mes_level.farm_wd,
            "farm_TI": config.mes_level.farm_TI,
            "farm_power": config.mes_level.farm_power,
            "ti_sample_count": config.mes_level.ti_sample_count,
        }

        # Store measurement detail dicts for compatibility
        self.ws_mes = {
            "ws_current": config.ws_mes.current,
            "ws_rolling_mean": config.ws_mes.rolling_mean,
            "ws_history_N": config.ws_mes.history_N,
            "ws_history_length": config.ws_mes.history_length,
            "ws_window_length": config.ws_mes.window_length,
        }

        self.wd_mes = {
            "wd_current": config.wd_mes.current,
            "wd_rolling_mean": config.wd_mes.rolling_mean,
            "wd_history_N": config.wd_mes.history_N,
            "wd_history_length": config.wd_mes.history_length,
            "wd_window_length": config.wd_mes.window_length,
        }

        self.yaw_mes = {
            "yaw_current": config.yaw_mes.current,
            "yaw_rolling_mean": config.yaw_mes.rolling_mean,
            "yaw_history_N": config.yaw_mes.history_N,
            "yaw_history_length": config.yaw_mes.history_length,
            "yaw_window_length": config.yaw_mes.window_length,
        }

        self.power_mes = {
            "power_current": config.power_mes.current,
            "power_rolling_mean": config.power_mes.rolling_mean,
            "power_history_N": config.power_mes.history_N,
            "power_history_length": config.power_mes.history_length,
            "power_window_length": config.power_mes.window_length,
        }

        # Store original config sections for compatibility
        self.act_pen = {
            "action_penalty": config.act_pen.action_penalty,
            "action_penalty_type": config.act_pen.action_penalty_type,
        }

        self.power_def = {
            "Power_reward": config.power_def.Power_reward,
            "Power_avg": config.power_def.Power_avg,
            "Power_scaling": config.power_def.Power_scaling,
        }

        # Convert probe configs to dicts for ProbeManager
        probes_config = [
            {
                "name": p.name,
                "turbine_index": p.turbine_index,
                "relative_position": p.relative_position,
                "include_wakes": p.include_wakes,
                "probe_type": p.probe_type,
            }
            for p in config.probes
        ]

        # Initialize probe manager
        self.probe_manager = ProbeManager(probes_config=probes_config)

        # Keep references for backward compatibility
        self.probes_config = probes_config
        self.probes = self.probe_manager.probes
        self.turbine_probes = self.probe_manager.turbine_probes

        # Set n_probes_per_turb now that probe_manager is initialized
        self.n_probes_per_turb = self.probe_manager.count_probes_per_turbine()

    def _init_farm_mes(self) -> None:
        """
        This function initializes the farm measurements class.
        This id done partly due to modularity, but also because we can delete it from memory later, as I suspect this might be the source of the memory leak
        """
        # Initializing the measurements class with the specified values.
        # TODO if history_length is 1, then we dont need to save the history, and we can just use the current values.
        # TODO is history_N is 1 or larger, then it is kinda implied that the rolling_mean is true.. Therefore we can change the if self.rolling_mean: check in the Mes() class, to be a if self.history_N >= 1 check... or something like that
        self.farm_measurements = FarmMes(
            n_turbines=self.n_turb,
            n_probes_per_turb=self.n_probes_per_turb,
            turb_ws=self.mes_level["turb_ws"],
            turb_wd=self.mes_level["turb_wd"],
            turb_TI=self.mes_level["turb_TI"],
            turb_power=self.mes_level["turb_power"],
            farm_ws=self.mes_level["farm_ws"],
            farm_wd=self.mes_level["farm_wd"],
            farm_TI=self.mes_level["farm_TI"],
            farm_power=self.mes_level["farm_power"],
            ws_current=self.ws_mes["ws_current"],
            ws_rolling_mean=self.ws_mes["ws_rolling_mean"],
            ws_history_N=self.ws_mes["ws_history_N"],
            ws_history_length=self.ws_mes["ws_history_length"],
            ws_window_length=self.ws_mes["ws_window_length"],
            wd_current=self.wd_mes["wd_current"],
            wd_rolling_mean=self.wd_mes["wd_rolling_mean"],
            wd_history_N=self.wd_mes["wd_history_N"],
            wd_history_length=self.wd_mes["wd_history_length"],
            wd_window_length=self.wd_mes["wd_window_length"],
            yaw_current=self.yaw_mes["yaw_current"],
            yaw_rolling_mean=self.yaw_mes["yaw_rolling_mean"],
            yaw_history_N=self.yaw_mes["yaw_history_N"],
            yaw_history_length=self.yaw_mes["yaw_history_length"],
            yaw_window_length=self.yaw_mes["yaw_window_length"],
            power_current=self.power_mes["power_current"],
            power_rolling_mean=self.power_mes["power_rolling_mean"],
            power_history_N=self.power_mes["power_history_N"],
            power_history_length=self.power_mes["power_history_length"],
            power_window_length=self.power_mes["power_window_length"],
            ws_min=self.ws_scaling_min,
            ws_max=self.ws_scaling_max,
            # Max and min values for wind direction measurements   NOTE i have added 5 for some slack in the measurements. so the scaling is better.
            wd_min=self.wd_scaling_min,
            wd_max=self.wd_scaling_max,
            yaw_min=self.yaw_scaling_min,
            yaw_max=self.yaw_scaling_max,
            TI_min=self.ti_scaling_min,
            TI_max=self.ti_scaling_max,
            power_max=self.maxturbpower,
            ti_sample_count=self.ti_sample_count,
        )

        # Deques that holds the power output of the farm and the baseline farm. This is used for the power reward
        self.farm_pow_deq = deque(maxlen=self.power_avg)
        self.base_pow_deq = deque(maxlen=self.power_avg)
        self.power_len = self.power_avg

        for i, tm in enumerate(self.farm_measurements.turb_mes):
            probes = self.turbine_probes.get(i, [])
            tm.probes = probes
            tm.n_probes = len(probes)
            tm.probe_min = self.ws_scaling_min
            tm.probe_max = self.ws_scaling_max

    def _init_spaces(self):
        """
        This function initializes the observation and action spaces.
        This is done in a seperate function, so we can replace it in the multi agent version of the environment
        """
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=((self.obs_var),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=((self.n_turb * self.act_var),), dtype=np.float32
        )

    def init_render(self):
        """Initialize rendering - delegates to renderer."""
        self.renderer.init_render(self.fs, self.turbine)

    def _take_measurements(self) -> None:
        """
        Does the measurement and saves it to the self.
        """
        # Get the observation of the environment
        self.current_ws = np.linalg.norm(
            self.fs.windTurbines.rotor_avg_windspeed, axis=1
        )

        u_speed = self.fs.windTurbines.rotor_avg_windspeed[:, 0]
        v_speed = self.fs.windTurbines.rotor_avg_windspeed[:, 1]

        self.current_wd = np.rad2deg(np.arctan2(v_speed, u_speed)) + self.wd

        self.current_yaw = self.fs.windTurbines.yaw
        self.current_powers = self.fs.windTurbines.power()  # The Power pr turbine

    def _update_measurements(self) -> None:
        """
        This function adds the current observations to the farm_measurements class
        """

        # Add a deprecation warning to this:
        raise DeprecationWarning(
            "This function is deprecated. Use _take_measurements instead, and then put them into the mes class yourself"
        )

        self._take_measurements()

        self.farm_measurements.add_measurements(
            self.current_ws, self.current_wd, self.current_yaw, self.powers
        )

    def _get_obs(self) -> np.ndarray:
        """
        Gets the sensordata from the farm_measurements class, and scales it to be between -1 and 1
        If you want to implement your own handling of the observations, then you can do that here by overwriting this function
        """

        values = self.farm_measurements.get_measurements(scaled=True)
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """
        Return info dictionary.
        If we have a baseline comparison, then we also return the baseline values.
        """
        return_dict = {
            "yaw angles agent": self.current_yaw,
            "yaw angles measured": self.farm_measurements.get_yaw_turb(),
            "Wind speed Global": self.ws,
            "Wind speed at turbines": self.current_ws,
            "Wind speed at turbines measured": self.farm_measurements.get_ws_turb(),
            "Wind speed at farm measured": self.farm_measurements.get_ws_farm(),
            "Wind direction Global": self.wd,
            "Wind direction at turbines": self.current_wd,
            "Wind direction at turbines measured": self.farm_measurements.get_wd_turb(),
            "Wind direction at farm measured": self.farm_measurements.get_wd_farm(),
            "Turbulence intensity": self.ti,
            "Power agent": self.fs.windTurbines.power().sum(),
            "Power agent nowake": self.fs.windTurbines.power(include_wakes=False).sum(),
            "Power pr turbine agent": self.fs.windTurbines.power(),
            "Turbine x positions": self.fs.windTurbines.positions_xyz[0],
            "Turbine y positions": self.fs.windTurbines.positions_xyz[1],
            "Turbulence intensity at turbines": self.farm_measurements.get_TI_turb(),
        }

        if self.Baseline_comp:
            return_dict["yaw angles base"] = self.fs_baseline.windTurbines.yaw
            return_dict["Power baseline"] = self.fs_baseline.windTurbines.power().sum()
            return_dict["Power pr turbine baseline"] = (
                self.fs_baseline.windTurbines.power()
            )
            # return_dict["Wind speed at turbines baseline"] = self.fs_baseline.windTurbines.rotor_avg_windspeed[:,0] #Just the largest component
            return_dict["Wind speed at turbines baseline"] = (
                self.fs_baseline.windTurbines.rotor_avg_windspeed[:, 0]
            )  # Just the largest component
        return return_dict


    def _set_windconditions(self) -> None:
        """
        Sets the global windconditions for the environment
        """
        wind_cond = self.wind_manager.sample_conditions()
        self.ws, self.wd, self.ti = wind_cond.unpack()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. This is called at the start of every episode.
        - The wind conditions are sampled, and the site is set.
        - The flow simulation is run for the time it takes for the flow to develop.
        - The measurements are filled up with the initial values.

        """
        # Seed the RNG used by this Env (sets self.np_random)
        super().reset(seed=seed)
        self.timestep = 0

        # Set random generators for managers
        self.wind_manager.np_random = self.np_random

        # 1) Global wind conditions + sites
        # wind_cond = self.wind_manager.sample_conditions()
        # self.ws, self.wd, self.ti = wind_cond.unpack()
        self._set_windconditions()

        # 2) Fresh measurement buffers
        self._init_farm_mes()
        if hasattr(self, "farm_measurements") and self.farm_measurements is not None:
            self.farm_measurements.np_random = self.np_random
        else:
            print("WARNING: farm_measurements was not initialized before reset.")

        # Rated power at current ws (for reward scaling)
        self.rated_power = self.turbine.power(self.ws)

        # 3) Turbines + main flow sim
        self._init_wts()

        # Set random generator for turbulence manager
        self.turbulence_manager.np_random = self.np_random

        # First need to calculate time parameters using turbulence manager
        turb_pos = np.stack([self.x_pos, self.y_pos]).T
        self.t_developed, self.time_max = (
            self.turbulence_manager._calculate_time_parameters(
                turbine_positions=turb_pos,
                rotor_diameter=self.D,
                ws=self.ws,
                n_passthrough=self.n_passthrough,
                burn_in_passthroughs=self.burn_in_passthroughs,
            )
        )

        if self.backend == "dynamiks":
            # --- ORIGINAL dynamic backend ---
            # Generate wind direction list for the episode

            # Generate wind direction list
            wd_list = self.wind_manager.make_wind_direction_list(
                base_wd=self.wd,
                time_max=self.time_max,
                dt_sim=self.dt_sim,
                t_developed=self.t_developed,
                steps_on_reset=self.steps_on_reset,
                wd_function=self.wd_function,
            )

            # Create sites and turbulence fields
            (
                self.site,
                self.site_base,
                _,
                _,
                self.addedTurbulenceModel,
            ) = self.turbulence_manager.create_sites(
                ws=self.ws,
                wd=self.wd,
                ti=self.ti,
                wd_list=wd_list,
                dt_sim=self.dt_sim,
                turbine_positions=turb_pos,
                rotor_diameter=self.D,
                n_passthrough=self.n_passthrough,
                burn_in_passthroughs=self.burn_in_passthroughs,
                create_baseline=self.Baseline_comp,
            )

            self.fs = DWMFlowSimulation(
                site=self.site,
                windTurbines=self.wts,
                wind_direction=self.wd,
                particleDeficitGenerator=jDWMAinslieGenerator(),
                dt=self.dt,
                n_particles=self.n_particles,
                d_particle=self.d_particle,
                particleMotionModel=HillVortexParticleMotion(
                    temporal_filter=self.temporal_filter
                ),
                addedTurbulenceModel=self.addedTurbulenceModel,
            )
        else:
            # --- STEADY pywake_steady backend ---
            if self.HTC_path is not None:
                raise NotImplementedError(
                    "pywake_steady backend does not support HAWC2WindTurbines."
                )
            from .backend.pywake_adapter import (
                PyWakeFlowSimulationAdapter,
            )  # or adjust import path

            self.fs = PyWakeFlowSimulationAdapter(
                x=np.asarray(self.x_pos, float),
                y=np.asarray(self.y_pos, float),
                windTurbine=self.turbine,  # py_wake WindTurbines definition
                ws=self.ws,
                wd=self.wd,
                ti=self.ti,
                dt=self.dt,
            )

        # Initial yaw set (bounded by yaw_start)
        self.fs.windTurbines.yaw = self._yaw_init(
            min_val=-self.yaw_start,
            max_val=self.yaw_start,
            n=self.n_turb,
            yaws=self.yaw_initial,
        )

        # Must init probes after fs
        self.probe_manager.initialize_probes(self.fs, self.fs.windTurbines.yaw)
        # Update references to point to probe_manager's collections
        self.probes = self.probe_manager.probes
        self.turbine_probes = self.probe_manager.turbine_probes

        # 3b) Baseline flow sim (optional)
        if self.Baseline_comp:
            if self.backend == "dynamiks":
                self.fs_baseline = DWMFlowSimulation(
                    site=self.site_base,
                    windTurbines=self.wts_baseline,
                    wind_direction=self.wd,
                    particleDeficitGenerator=jDWMAinslieGenerator(),
                    dt=self.dt,
                    n_particles=self.n_particles,
                    d_particle=self.d_particle,
                    particleMotionModel=HillVortexParticleMotion(
                        temporal_filter=self.temporal_filter
                    ),
                    addedTurbulenceModel=self.addedTurbulenceModel,
                )
            else:
                if self.HTC_path is not None:
                    raise NotImplementedError(
                        "pywake_steady baseline does not support HAWC2WindTurbines."
                    )
                from .backend.pywake_adapter import PyWakeFlowSimulationAdapter

                self.fs_baseline = PyWakeFlowSimulationAdapter(
                    x=np.asarray(self.x_pos, float),
                    y=np.asarray(self.y_pos, float),
                    windTurbine=self.turbine,
                    ws=self.ws,
                    wd=self.wd,
                    ti=self.ti,
                    dt=self.dt,
                )

            # Start baseline with same yaw as agent at reset
            self.fs_baseline.windTurbines.yaw = self.fs.windTurbines.yaw

        # 3c) Run the flow for the time it takes to develop
        if self.backend == "dynamiks":
            self.fs.run(self.t_developed)
            if self.Baseline_comp:
                self.fs_baseline.run(self.t_developed)
        else:
            # Steady-state: nothing to "develop", but keep API consistent
            self.fs.run(0)
            if self.Baseline_comp:
                self.fs_baseline.run(0)

        if self.Baseline_comp and self.baseline_manager is not None:
            # Update baseline manager wind conditions
            self.baseline_manager.update_wind_conditions(
                ws=self.ws, wd=self.wd, ti=self.ti
            )

        # 4) Fill measurement history window (and power deques)
        #    Uses the unified inner loop; no action applied during reset.
        for _ in range(self.steps_on_reset):
            out = self._advance_and_measure(
                self.sim_steps_per_env_step,
                apply_agent_action=False,
                action=None,
                include_baseline=self.Baseline_comp,
            )

            # Push means into measurement buffers
            self.farm_measurements.add_measurements(
                out["mean_windspeed"],
                out["mean_winddir"],
                out["mean_yaw"],
                out["mean_power"],
            )
            # Power history (farm-level)
            self.farm_pow_deq.append(out["mean_power"].sum())
            if self.Baseline_comp:
                self.base_pow_deq.append(out["baseline_power_mean"].sum())

        # 5) Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Init render can now be called as fs needs to be created first
        if self.render_mode in ["human", "rgb_array"]:
            self.init_render()

        return observation, info


    def _advance_and_measure(
        self,
        n_sim_steps: int,
        ignore_steps: int = 0,
        *,
        apply_agent_action: bool = False,
        action: np.ndarray | None = None,
        include_baseline: bool = False,
    ):
        """
        Advance the simulation n_sim_steps times.
        Optionally skip x ammount of measurements for what is meaned over.
        Optionally apply the agent action each sim step (yaw or wind method).
        Optionally step baseline using its controller.

        Returns:
            dict with keys:
            - time_array: (n_sim_steps,)
            - windspeeds, winddirs, yaws, powers: (n_sim_steps, n_turb)
            - baseline_powers, yaws_baseline, windspeeds_baseline (if include_baseline): same shapes
            - mean_windspeed, mean_winddir, mean_yaw, mean_power: (n_turb,)
            - baseline_power_mean (if include_baseline): scalar (farm sum) or (n_turb,) – here we return (n_turb,)
        """
        T = n_sim_steps
        n = self.n_turb
        time_array = np.zeros(T, dtype=np.float32)
        windspeeds = np.zeros((T, n), dtype=np.float32)
        winddirs = np.zeros((T, n), dtype=np.float32)
        yaws = np.zeros((T, n), dtype=np.float32)
        powers = np.zeros((T, n), dtype=np.float32)

        # Make sure that ignore_steps is either none, or less than T
        if ignore_steps >= T:
            raise ValueError("ignore_steps must be less than n_sim_steps")
        elif ignore_steps < 0:
            raise ValueError("ignore_steps must be non-negative")

        if include_baseline:
            baseline_powers = np.zeros((T, n), dtype=np.float32)
            yaws_baseline = np.zeros((T, n), dtype=np.float32)
            windspeeds_baseline = np.zeros((T, n), dtype=np.float32)

        # If in "yaw" mode, we have an action budget that spans the env step
        if apply_agent_action and self.ActionMethod == "yaw":
            self.action_remaining = (
                action * self.yaw_step_env
            )  # total budget for this env step

        for j in range(T):
            # 1) Agent yaw update (if any)
            if apply_agent_action:
                self._adjust_yaws(action)

            # 2) Step agent flow
            wd_old = self.fs.wind_direction
            self.fs.step()

            wd_new = self.fs.wind_direction
            delta_wd = wd_new - wd_old
            self.fs.windTurbines.yaw += delta_wd

            # 3) Baseline, only if requested
            if include_baseline:
                if apply_agent_action:
                    new_baseline_yaws = self.baseline_manager.compute_baseline_action(
                        fs=self.fs_baseline, yaw_step=self.yaw_step_sim
                    )
                    self.fs_baseline.windTurbines.yaw = new_baseline_yaws
                wd_old_baseline = self.fs_baseline.wind_direction
                self.fs_baseline.step()

                wd_new_baseline = self.fs_baseline.wind_direction
                delta_wd_baseline = wd_new_baseline - wd_old_baseline
                self.fs_baseline.windTurbines.yaw += delta_wd_baseline

            # 4) Measurements at this sim step
            self._take_measurements()

            # HF TI buffering (if requested)
            if self.farm_measurements.turb_TI or self.farm_measurements.farm_TI:
                for i in range(self.n_turb):
                    self.farm_measurements.turb_mes[i].add_hf_ws(self.current_ws[i])
                if self.farm_measurements.farm_TI:
                    self.farm_measurements.farm_mes.add_hf_ws(np.mean(self.current_ws))

            # 5) Store arrays
            windspeeds[j] = self.current_ws
            winddirs[j] = self.current_wd
            yaws[j] = self.current_yaw
            powers[j] = self.current_powers
            time_array[j] = self.fs.time

            self.probe_manager.update_probe_positions(self.fs, yaws[j])

            if include_baseline:
                baseline_powers[j] = self.fs_baseline.windTurbines.power(
                    include_wakes=self.baseline_wakes
                )
                yaws_baseline[j] = self.fs_baseline.windTurbines.yaw
                windspeeds_baseline[j] = np.linalg.norm(
                    self.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
                )
                # update probe positions to follow turbine

        # 6) Aggregate to per-env-step means
        mean_windspeed = np.mean(windspeeds[ignore_steps:, :], axis=0)
        mean_winddir = np.mean(winddirs[ignore_steps:, :], axis=0)
        mean_yaw = np.mean(yaws[ignore_steps:, :], axis=0)
        mean_power = np.mean(powers[ignore_steps:, :], axis=0)  # per-turbine

        result = dict(
            time_array=time_array,
            windspeeds=windspeeds,
            winddirs=winddirs,
            yaws=yaws,
            powers=powers,
            mean_windspeed=mean_windspeed,
            mean_winddir=mean_winddir,
            mean_yaw=mean_yaw,
            mean_power=mean_power,
        )
        if include_baseline:
            result.update(
                baseline_powers=baseline_powers,
                yaws_baseline=yaws_baseline,
                windspeeds_baseline=windspeeds_baseline,
                baseline_power_mean=np.mean(baseline_powers, axis=0),  # per-turbine
            )
        return result

    def _adjust_yaws(self, action):
        """
        Heavily inspired from https://github.com/AlgTUDelft/wind-farm-env
        This function adjusts the yaw angles of the turbines, based on the actions given, but we now have differnt methods for the actions
        """

        if self.ActionMethod == "yaw":
            # The new yaw angles are the old yaw angles + the action, scaled with the yaw_step
            # 0 action means no change
            # the new yaw angles are the old yaw angles + the action, scaled with the yaw_step

            # This is how much the yaw can change pr sim step
            yaw_change = np.clip(
                self.action_remaining,
                -self.yaw_step_sim,
                self.yaw_step_sim,
                dtype=np.float32,
            )

            self.fs.windTurbines.yaw += yaw_change
            # clip the yaw angles to be between -30 and 30
            self.fs.windTurbines.yaw = np.clip(
                self.fs.windTurbines.yaw, self.yaw_min, self.yaw_max
            )

            self.action_remaining -= yaw_change

        elif self.ActionMethod == "wind":
            # The new yaw angles are the action, scaled to be between the min and max yaw angles
            # 0 action means to move to 0 yaw angle, and 1 action means to move to the max yaw angle
            new_yaws = (action + 1.0) / 2.0 * (
                self.yaw_max - self.yaw_min
            ) + self.yaw_min

            if (
                self.HTC_path is None
            ):  # This clip is only usefull for the pywake turbine model, as the hawc2 model has inertia anyways
                # The bounds for the yaw angles are:
                yaw_max = self.fs.windTurbines.yaw + self.yaw_step_sim
                yaw_min = self.fs.windTurbines.yaw - self.yaw_step_sim

                # The new yaw angles are the new yaw angles, but clipped to be between the yaw_max and yaw_min
                self.fs.windTurbines.yaw = np.clip(
                    np.clip(new_yaws, yaw_min, yaw_max), self.yaw_min, self.yaw_max
                )

            else:
                # The new yaw angles are the new yaw angles, but clipped to be between the yaw_min and yaw_max
                self.fs.windTurbines.yaw = np.clip(new_yaws, self.yaw_min, self.yaw_max)

        elif self.ActionMethod == "absolute":
            raise NotImplementedError("The absolute method is not implemented yet")

        else:
            raise ValueError("The ActionMethod must be yaw, wind or absolute")


    def step(self, action):
        """
        The step function
        1. Adjust the yaw angles of the turbines
        2. Take a step in the flow simulation
        3. Update the measurements
        4. Calculate the reward
        5. Return the observation, reward, terminated, truncated and info

        """

        # Save the old yaw angles, so we can calculate the change in yaw angles
        self.old_yaws = copy.copy(self.fs.windTurbines.yaw)

        # This is the ammount of steps we need to do, to ensure we have the correct delay
        steps_with_delay = (
            self.sim_steps_per_env_step
            + ((self.delay - self.dt_env) // self.dt_env) * self.sim_steps_per_env_step
        )
        ignore_steps = steps_with_delay - self.sim_steps_per_env_step

        out = self._advance_and_measure(
            steps_with_delay,
            ignore_steps=ignore_steps,
            apply_agent_action=True,
            action=action,
            include_baseline=self.Baseline_comp,
        )

        # add to measurements/history
        self.farm_measurements.add_measurements(
            out["mean_windspeed"],
            out["mean_winddir"],
            out["mean_yaw"],
            out["mean_power"],
        )
        self.farm_pow_deq.append(out["mean_power"].sum())
        if self.Baseline_comp:
            self.base_pow_deq.append(out["baseline_power_mean"].sum())

        if np.any(np.isnan(self.farm_pow_deq)):
            raise Exception("NaN Power")

        # Build observation / info
        observation = self._get_obs()
        info = self._get_info()
        info["time_array"] = out["time_array"]
        info["windspeeds"] = out["windspeeds"]
        info["yaws"] = out["yaws"]
        info["powers"] = out["powers"]
        if self.Baseline_comp:
            info["baseline_powers"] = out["baseline_powers"]
            info["yaws_baseline"] = out["yaws_baseline"]
            info["windspeeds_baseline"] = out["windspeeds_baseline"]

        # Calculate the reward using the reward calculator
        reward = self.reward_calculator.calculate_total_reward(
            farm_power_deque=self.farm_pow_deq,
            old_yaws=self.old_yaws,
            new_yaws=self.fs.windTurbines.yaw,
            yaw_max=self.yaw_max,
            baseline_power_deque=self.base_pow_deq if self.Baseline_comp else None,
            rated_power=self.rated_power,
            n_turbines=self.n_turb,
        )[0]  # [0] gets just the reward value, not the breakdown

        # If we are at the end of the simulation, we truncate the agents.
        # Note that this is not the same as terminating the agents.
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        if self.timestep >= self.time_max:
            # terminated = {a: True for a in self.agents}
            truncated = True
            # Clean up the flow simulation. This is to make sure that we dont have a memory leak.
            if self.cleanup_on_time_limit:
                self._cleanup_resources()
        else:
            truncated = False

        self.timestep += 1

        terminated = False

        return observation, reward, terminated, truncated, info

    def _cleanup_resources(self) -> None:
        """Close handles, delete temp dirs, drop heavy refs to avoid leaks."""
        if self.Baseline_comp:
            if self.HTC_path is not None:
                self.wts_baseline.h2.close()
            self.fs_baseline = None
            self.site_base = None

        if self.HTC_path is not None:
            # Close the connections
            self.wts.h2.close()
            self.wts_baseline.h2.close()
            # Delete the directory
            self._deleteHAWCfolder()

        self.fs = None
        self.site = None
        self.farm_measurements = None
        gc.collect()

    def _deleteHAWCfolder(self):
        """
        This deletes the HAWC2 results folder from the directory.
        This is done to make sure we keep it nice and clean
        """
        # This is the path to the results
        delete_folder = (
            self.wts.htc_lst[0].modelpath
            + os.path.split(self.wts.htc_lst[0].output.filename.values[0])[0]
        )
        shutil.rmtree(delete_folder)

        # Also delete the htc folder
        htc_folder = (
            self.wts.htc_lst[0].modelpath
            + os.path.split(
                self.wts.htc_lst[0].output.filename.values[0].replace("res", "htc")
            )[0]
        )
        shutil.rmtree(htc_folder)

        if self.Baseline_comp:
            delete_folder_baseline = (
                self.wts_baseline.htc_lst[0].modelpath
                + os.path.split(self.wts_baseline.htc_lst[0].output.filename.values[0])[
                    0
                ]
            )
            shutil.rmtree(delete_folder_baseline)

            # Also delete the htc folder
            htc_folder_baseline = (
                self.wts_baseline.htc_lst[0].modelpath
                + os.path.split(
                    self.wts_baseline.htc_lst[0]
                    .output.filename.values[0]
                    .replace("res", "htc")
                )[0]
            )
            shutil.rmtree(htc_folder_baseline)

    def render(self):
        """Render method required by Gymnasium API - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        return self.renderer.render(self.fs, fs_baseline, probes)

    def _render_frame_for_human(self, baseline=False):
        """Render the environment and return an RGB frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        return self.renderer._render_frame_for_human(
            self.fs, fs_baseline, probes, baseline, self.turbine, self.ws
        )

    def _render_frame(self, baseline=False):
        """Renders the current environment state and returns the frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        return self.renderer._render_frame(
            self.fs, fs_baseline, probes, baseline, self.turbine, self.ws
        )


    def close(self):
        """Close the environment and clean up resources."""
        self.renderer.close()
        if self.Baseline_comp:
            self.fs_baseline = None
            self.site_base = None
        self.fs = None
        self.site = None
        self.farm_measurements = None
        gc.collect()

    def plot_farm(self, baseline=False):
        """Plot the entire farm layout - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer.plot_farm(self.fs, fs_baseline, self.turbine, baseline)

    def _render_farm(self, baseline=False):
        """Internal farm rendering - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer._render_farm(self.fs, fs_baseline, baseline)

    def plot_frame(self, baseline=False):
        """Plot a single frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer.plot_frame(self.fs, fs_baseline, self.turbine, baseline)

    def _get_num_raw_features(self):
        """Calculate based on YAML config - no hardcoding!"""
        features = 0
        # Turbine-level sensors
        if self.mes_level["turb_ws"]:
            features += self.n_turb
        if self.mes_level["turb_wd"]:
            features += self.n_turb
        if self.mes_level["turb_TI"]:
            features += self.n_turb
        if self.mes_level["turb_power"]:
            features += self.n_turb

        # Farm-level sensors
        if self.mes_level["farm_ws"]:
            features += 1
        if self.mes_level["farm_wd"]:
            features += 1
        if self.mes_level["farm_TI"]:
            features += 1
        if self.mes_level["farm_power"]:
            features += 1

        return features

    @property
    def pywake_agent(self):
        """Expose pywake_agent from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager.pywake_agent
        return None

    @property
    def py_agent_mode(self):
        """Expose py_agent_mode from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager.py_agent_mode
        return None

    @property
    def _base_controller(self):
        """Expose _base_controller from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager._base_controller
        return None
