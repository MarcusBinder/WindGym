from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os
import gc
import socket
import shutil

# Dynamiks imports
from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion

# from dynamiks.dwm.particle_motion_models import ParticleMotionModel
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField, RandomTurbulence
from dynamiks.wind_turbines import PyWakeWindTurbines
from dynamiks.views import XYView
from dynamiks.dwm.added_turbulence_models import (
    SynchronizedAutoScalingIsotropicMannTurbulence,
    AutoScalingIsotropicMannTurbulence,
)

from IPython import display

# WindGym imports
from .WindEnv import WindEnv
from .MesClass import farm_mes
from .BasicControllers import local_yaw_controller, global_yaw_controller
from .Agents import PyWakeAgent

from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from collections import deque
import itertools
import yaml
from dynamiks.wind_turbines.hawc2_windturbine import HAWC2WindTurbines
from dynamiks.dwm.particle_motion_models import CutOffFrq

# For live plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CutOffFrqLio2021 = CutOffFrq(4)

"""
This is the base for the wind farm environment. This is where the magic happens.
For now it only supports the PyWakeWindTurbines, but it should be easy to expand to other types of turbines.
"""


# TODO make it so that the turbines can be other then a square grid
# TODO thrust coefficient control
# TODO for now I have just hardcoded this scaling value (1 and 25 for the wind_speed min and max). This is beacuse the wind speed is chosen from the normal distribution, but becasue of the wakes and the turbulence, we canhave cases where we go above or below these values.


class WindFarmEnv(WindEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        n_passthrough=5,
        ws_scaling_min: float = 0.0,
        ws_scaling_max: float = 30.0,
        wd_scaling_min: float = 0,
        wd_scaling_max: float = 360,
        ti_scaling_min: float = 0.0,
        ti_scaling_max: float = 1.0,
        yaw_scaling_min: float = -45,
        yaw_scaling_max: float = 45,
        TurbBox="Default",
        turbtype="MannGenerate",
        yaml_path=None,
        Baseline_comp=False,
        yaw_init=None,
        render_mode=None,
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step_sim=1,  # How many degrees the yaw angles can change pr. simulation step
        yaw_step_env=None,  # How many degrees the yaw angles can change pr. environment step
        fill_window=True,
        sample_site=None,
        HTC_path=None,
        reset_init=True,
        burn_in_passthroughs=2,  # number of passthroughs before episode starts
    ):
        """
        This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
        Args:
            turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
            n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
            TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
            TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
            TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
            turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
            yaml_path: str: The path to the yaml file that contains the configuration of the environment. TODO make a default value for this
            Baseline_comp: bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
            yaw_init: str: The method for initializing the yaw angles of the turbines. If 'Random', then the yaw angles will be random. Else they will be zeros.
            render_mode: str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
            seed: int: The seed for the environment. If None, then the seed will be random.
            dt_sim: float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
            dt_env: float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
            yaw_step_sim: float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
            fill_window: bool: If True, then the measurements will be filled up at reset.
            sample_site: pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
            HTC_path: str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
            reset_init: bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
        """
        # Check that x_pos and y_pos are the same length
        if len(x_pos) != len(y_pos):
            raise ValueError("x_pos and y_pos must be the same length")

        # Predefined values
        self.wts = None
        self.wts_baseline = None
        self.burn_in_passthroughs = burn_in_passthroughs
        # The power setpoint for the farm. This is used if the Track_power is True. (Not used yet)
        self.power_setpoint = 0.0
        self.act_var = (
            1  # number of actions pr. turbine. For now it is just the yaw angles
        )
        self.HTC_path = HTC_path
        self.fill_window = fill_window
        self.dt = dt_sim  # DWM simulation timestep
        self.dt_sim = dt_sim
        self.dt_env = dt_env  # Environment timestep
        self.sim_steps_per_env_step = int(self.dt_env / self.dt_sim)
        if self.dt_env % self.dt_sim != 0:
            raise ValueError("dt_env must be a multiple of dt_sim")

        self.sample_site = sample_site
        self.yaw_start = 15.0  # This is the limit for the initialization of the yaw angles. This is used to make sure that the yaw angles are not too large at the start, but still not zero
        # Max power pr turbine. Used in the measurement class
        self.maxturbpower = max(turbine.power(np.arange(10, 25, 1)))
        # The step size for the yaw angles. How manny degress the yaw angles can change pr. step
        # The distance between the particles. This is used in the flow simulation.
        self.d_particle = 0.2
        self.n_particles = None
        self.temporal_filter = CutOffFrqLio2021
        self.turbtype = turbtype
        self.yaw_step_sim = yaw_step_sim  # How many degrees the yaw angles can change pr. simulation step

        if yaw_step_env is None:
            self.yaw_step_env = yaw_step_sim * self.sim_steps_per_env_step
        else:
            self.yaw_step_env = yaw_step_env

        # Saves to self
        self.ws_scaling_min = ws_scaling_min
        self.ws_scaling_max = ws_scaling_max
        self.wd_scaling_min = wd_scaling_min
        self.wd_scaling_max = wd_scaling_max
        self.ti_scaling_min = ti_scaling_min
        self.ti_scaling_max = ti_scaling_max
        self.yaw_scaling_min = yaw_scaling_min
        self.yaw_scaling_max = yaw_scaling_max
        self.seed = seed
        self.TurbBox = TurbBox
        self.turbine = turbine
        # The maximum time of the simulation. This is used to make sure that the simulation doesnt run forever.
        self.time_max = 0
        # The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
        self.n_passthrough = n_passthrough
        self.timestep = 0

        self.TF_files = []
        # The initial yaw of the turbines. This is used if the yaw_init is "Defined"
        self.yaw_initial = [0]

        # Load the configuration
        self.load_config(yaml_path)
        self.yaml_path = yaml_path

        self.n_turb = len(x_pos)  # The number of turbines

        # Deques that holds the power output of the farm and the baseline farm. This is used for the power reward
        self.farm_pow_deq = deque(maxlen=self.power_avg)
        self.base_pow_deq = deque(maxlen=self.power_avg)
        self.power_len = self.power_avg

        # Sets the yaw init method. If Random, then the yaw angles will be random. Else they will be zeros
        # If yaw_init is defined (it will be if we initialize from EnvEval) then set it like this. Else just use the value from the yaml
        if yaw_init is not None:
            # We only ever have this, IF we have set the value from
            if yaw_init == "Random":
                self._yaw_init = self._randoms_uniform
            elif yaw_init == "Defined":
                self._yaw_init = self._defined_yaw
            else:
                self._yaw_init = self._return_zeros
        else:
            if self.yaw_init == "Random":
                self._yaw_init = self._randoms_uniform
            elif self.yaw_init == "Defined":
                self._yaw_init = self._defined_yaw
            else:
                self._yaw_init = self._return_zeros

        # Define the power tracking reward function TODO Not implemented yet. Also make the power_setpoint an observable parameter
        if self.Track_power:
            self.power_setpoint = 42  # ???
            self._track_rew = self.track_rew_avg
            raise NotImplementedError("The Track_power is not implemented yet")
        else:
            self._track_rew = self.track_rew_none

        # Define the power production reward function
        if self.power_reward == "Baseline":
            self._power_rew = (
                self.power_rew_baseline
            )  # The baseline power reward function
        elif self.power_reward == "Power_avg":
            self._power_rew = self.power_rew_avg  # The power_avg reward function
        elif self.power_reward == "None":
            self._power_rew = self.power_rew_none  # The no power reward function
        elif self.power_reward == "Power_diff":
            # TODO rethink this way of doing it.
            self._power_rew = self.power_rew_diff  # The power_diff reward function
            # We set this to 10, to have some space in the middle.
            self._power_wSize = self.power_avg // 10
            if self.power_avg < 40:
                # Why 40? I just chose this as the minimum value. In reality 2 could have sufficed, but to save myself a headache, I set it to 10
                raise ValueError(
                    "The Power_avg must be larger then 40 for the Power_diff reward. Also it should probably be way larger my guy"
                )
        else:
            raise ValueError(
                "The Power_reward must be either Baseline, Power_avg, None or Power_diff"
            )

        # Read in the turb boxes
        if turbtype == "MannLoad":
            if TurbBox is None or not os.path.exists(TurbBox):
                raise FileNotFoundError(
                    "turbtype is 'MannLoad' but a valid path was not provided to 'TurbBox'. "
                    "Please provide a path to your turbulence files or use turbtype='MannGenerate'."
                )

            if os.path.isfile(TurbBox):
                self.TF_files.append(TurbBox)
            else:
                for f in os.listdir(TurbBox):
                    if f.startswith("TF_") and f.endswith(".nc"):  # Be more specific
                        self.TF_files.append(os.path.join(TurbBox, f))

            if not self.TF_files:
                raise FileNotFoundError(
                    f"No valid turbulence files (TF_*.nc) found in directory: {TurbBox}"
                )

        # If we need to have a "baseline" farm, then we need to set up the baseline controller
        # This could be moved to the Power_reward check, but I have a feeling this will be expanded in the future, when we include damage.
        if self.power_reward == "Baseline" or Baseline_comp:
            self.Baseline_comp = True
        else:
            self.Baseline_comp = False

        # #Initializing the measurements class with the specified values.
        self._init_farm_mes()

        # The maximum history length of the measurements
        self.hist_max = self.farm_measurements.max_hist()

        # Figure out the ammount of steps to do at the reset
        if self.fill_window is True:
            self.steps_on_reset = self.hist_max
        elif isinstance(self.fill_window, int) and self.fill_window >= 1:
            if self.fill_window > self.hist_max:
                self.fill_window = (
                    self.hist_max
                )  # fill_window cannot be larger then the max history length
            self.steps_on_reset = self.fill_window
        elif self.fill_window is False:
            self.steps_on_reset = 1
        else:
            raise ValueError("fill_window must be True or a non-negative integer")

        # Setting up the turbines:

        self.D = turbine.diameter()

        self.x_pos = x_pos
        self.y_pos = y_pos

        # Define the observation and action space
        self.obs_var = self.farm_measurements.observed_variables()

        self._init_spaces()

        if reset_init:
            # We should have this here, to set the seeding correctly
            self.reset(seed=seed)

        # Asserting that the render_mode is valid.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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

        if self.HTC_path is not None:
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
        # Setting up the baseline controller if we need it
        if self.Baseline_comp:
            # If we compare to some baseline performance, then we also need a controller for that
            if self.BaseController == "Local":
                self._base_controller = local_yaw_controller
            elif self.BaseController == "Global":
                self._base_controller = global_yaw_controller
            elif self.BaseController.split("_")[0] == "PyWake":
                if "_" in self.BaseController:
                    self.py_agent_mode = self.BaseController.split("_")[1]
                else:
                    self.py_agent_mode = "oracle"
                    # In oracle mode we just use the global conditions always.

                if self.py_agent_mode not in ["oracle", "local"]:
                    raise ValueError(
                        "The PyWakeAgent can only be used in oracle or local mode. Please specify the mode in the BaseController string."
                    )

                # lookup_mode is true if self.py_agent_mode == "local", else it's false
                lookup_mode = self.py_agent_mode == "local"

                # This is a workarround to make the PyWakeAgent work with the baseline farm. Better must exist.
                class Fake_env:
                    def __init__(
                        self,
                        action_method="wind",
                        yaw_max=45,
                        yaw_min=-45,
                        yaw_step_env=1,
                    ):
                        self.ActionMethod = action_method
                        # Mocking the unwrapped attributes that PyWakeAgent expects
                        self.unwrapped = self  # Or a separate mock object with just the needed attributes
                        self.yaw_max = yaw_max
                        self.yaw_min = yaw_min
                        self.yaw_step_env = yaw_step_env

                # When creating temp_env:
                temp_env = Fake_env(
                    action_method="wind",
                    yaw_max=self.yaw_max,
                    yaw_min=self.yaw_min,
                    yaw_step_env=self.yaw_step_env,
                )

                self.pywake_agent = PyWakeAgent(
                    x_pos=self.x_pos,
                    y_pos=self.y_pos,
                    turbine=self.turbine,
                    yaw_max=self.yaw_max,
                    yaw_min=self.yaw_min,
                    env=temp_env,
                    look_up=lookup_mode,  #
                )
                self._base_controller = self.PyWakeAgentWrapper
            else:
                raise ValueError(
                    "The BaseController must be either Local or Global... For now"
                )
            # Definde the turbines
            if self.HTC_path is not None:
                # If we have a high fidelity turbine model, then we need to load it in
                self.wts_baseline = HAWC2WindTurbines(
                    x=self.x_pos,
                    y=self.y_pos,
                    htc_lst=[self.HTC_path],
                    case_name=name_string
                    + "_baseline",  # subfolder name in the htc, res and log folders
                    suppress_output=True,  # don't show hawc2 output in console
                )
                # Add the yaw sensor, but because the only keyword does not work with h2lib, we add another layer that then only returns the first values of them.
                self.wts_baseline.add_sensor(
                    name="yaw_getter",
                    getter="constraint bearing2 yaw_rot 1 only 1;",  #
                    expose=False,
                    ext_lst=["angle", "speed"],
                )
                self.wts_baseline.add_sensor(
                    "yaw",
                    getter=lambda wt: np.rad2deg(wt.sensors.yaw_getter[:, 0]),
                    setter=lambda wt, value: wt.h2.set_variable_sensor_value(
                        1, np.deg2rad(value).tolist()
                    ),
                    expose=True,
                )
            else:  # If we have no HTC path, use the pywake turbine
                self.wts_baseline = PyWakeWindTurbines(
                    x=self.x_pos,
                    y=self.y_pos,  # x and y position of two wind turbines
                    windTurbine=self.turbine,
                )

    def PyWakeAgentWrapper(self, fs, yaw_step=1):
        """
        This function wraps the PyWakeAgent, so that it can be used as a controller for the baseline farm.
        This is done to make sure that the agent can be used in the same way as the other controllers.
        fs: Flow simulation object. For the BASELINE
        """
        # This is mostly the _adjust_yaws method. Just slightly formatted to work with the baseline now.

        # We can run this method in two different ways. Either in oracle mode or in local mode
        # oracle mode just uses the global wind conditions, while local mode uses the local wind conditions at the turbines.
        if self.py_agent_mode == "local":
            #  This is a bit crude, and can be improved, but we just use the front most turbine for this.
            front_tb = np.argmin(self.fs_baseline.windTurbines.positions_xyz[0, :])
            ws_front = self.fs_baseline.windTurbines.get_rotor_avg_windspeed(
                include_wakes=True
            )[:, front_tb]
            ws_use = np.linalg.norm(ws_front)
            wd_use = np.rad2deg(np.arctan(ws_front[1] / ws_front[0])) + self.wd

            # Make the wd and ws update somewhat slowly, using polyak averaging
            tau = 0.05

            self.pywake_wd = (1 - tau) * self.pywake_wd + tau * wd_use
            self.pywake_ws = (1 - tau) * self.pywake_ws + tau * ws_use

            self.pywake_agent.update_wind(
                wind_speed=self.pywake_ws,
                wind_direction=self.pywake_wd,
                TI=self.ti,
            )

        # This assumes we are using the "wind" based actions.
        action = self.pywake_agent.predict()[0]

        # This unscales the actions.
        new_yaws = (action + 1.0) / 2.0 * (self.yaw_max - self.yaw_min) + self.yaw_min

        # So for the baseline controller, we also need to adjust the yaws based on the wind direction error.
        # This is done internally inside the pywakeAgent but not when used as a baseline controller.
        pywake_error = self.pywake_agent.wdir - self.wd
        actual_yaws = new_yaws - pywake_error

        yaw_max = fs.windTurbines.yaw + self.yaw_step_sim
        yaw_min = fs.windTurbines.yaw - self.yaw_step_sim

        new_yaws = np.clip(
            np.clip(actual_yaws, yaw_min, yaw_max), self.yaw_min, self.yaw_max
        )

        return new_yaws

    def load_config(self, config_path):
        """
        This loads in the yaml file, and sets a bunch of internal values.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)  # Load the YAML file

        # Set the attributes of the class based on the config file
        self.yaw_init = config.get("yaw_init")
        self.BaseController = config.get("BaseController")
        self.ActionMethod = config.get("ActionMethod")
        # self.Baseline_comp = config.get('Baseline_comp')
        self.Track_power = config.get("Track_power")

        # Unpack the farm params
        farm_params = config.get("farm")
        self.yaw_min = farm_params["yaw_min"]
        self.yaw_max = farm_params["yaw_max"]
        # self.xDist = farm_params["xDist"]
        # self.yDist = farm_params["yDist"]
        # self.nx = farm_params["nx"]
        # self.ny = farm_params["ny"]

        # get the inflow bounds. Note that these are distinct from the scaling bounds.
        wind_params = config.get("wind")
        self.ws_inflow_min = wind_params["ws_min"]
        self.ws_inflow_max = wind_params["ws_max"]
        self.TI_inflow_min = wind_params["TI_min"]
        self.TI_inflow_max = wind_params["TI_max"]
        self.wd_inflow_min = wind_params["wd_min"]
        self.wd_inflow_max = wind_params["wd_max"]

        self.act_pen = config.get("act_pen")
        self.power_def = config.get("power_def")
        self.mes_level = config.get("mes_level")
        self.ws_mes = config.get("ws_mes")
        self.wd_mes = config.get("wd_mes")
        self.yaw_mes = config.get("yaw_mes")
        self.power_mes = config.get("power_mes")

        self.ti_sample_count = self.mes_level.get("ti_sample_count", 30)

        # unpack some more, because we use these later.
        self.action_penalty = self.act_pen["action_penalty"]
        self.action_penalty_type = self.act_pen["action_penalty_type"]
        self.Power_scaling = self.power_def["Power_scaling"]
        self.power_avg = self.power_def["Power_avg"]
        self.power_reward = self.power_def["Power_reward"]

    def _init_farm_mes(self):
        """
        This function initializes the farm measurements class.
        This id done partly due to modularity, but also because we can delete it from memory later, as I suspect this might be the source of the memory leak
        """
        # Initializing the measurements class with the specified values.
        # TODO if history_length is 1, then we dont need to save the history, and we can just use the current values.
        # TODO is history_N is 1 or larger, then it is kinda implied that the rolling_mean is true.. Therefore we can change the if self.rolling_mean: check in the Mes() class, to be a if self.history_N >= 1 check... or something like that
        self.farm_measurements = farm_mes(
            n_turbines=self.n_turb,
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
        plt.ion()
        x_turb, y_turb = self.fs.windTurbines.positions_xyz[:2]

        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.a = np.linspace(-200 + min(x_turb), 1000 + max(x_turb), 250)
        self.b = np.linspace(-200 + min(y_turb), 200 + max(y_turb), 250)

        self.view = XYView(
            z=self.turbine.hub_height(), x=self.a, y=self.b, ax=self.ax, adaptive=False
        )

        plt.close()

    def _take_measurements(self):
        """
        Does the measurement and saves it to the self.
        """
        # Get the observation of the environment
        self.current_ws = np.linalg.norm(
            self.fs.windTurbines.rotor_avg_windspeed, axis=1
        )

        u_speed = self.fs.windTurbines.rotor_avg_windspeed[:, 0]
        v_speed = self.fs.windTurbines.rotor_avg_windspeed[:, 1]

        self.current_wd = np.rad2deg(np.arctan(v_speed / u_speed)) + self.wd

        self.current_yaw = self.fs.windTurbines.yaw
        self.current_powers = self.fs.windTurbines.power()  # The Power pr turbine

    def _update_measurements(self):
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

    def _get_obs(self):
        """
        Gets the sensordata from the farm_measurements class, and scales it to be between -1 and 1
        If you want to implement your own handling of the observations, then you can do that here by overwriting this function
        """

        values = self.farm_measurements.get_measurements(scaled=True)
        return np.clip(values, -1.0, 1.0, dtype=np.float32)

    def _get_info(self):
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

    def _set_windconditions(self):
        """
        Sets the global windconditions for the environment
        """

        if self.sample_site is None:
            # The wind speed is a random number between ws_min and ws_max
            self.ws = self._random_uniform(self.ws_inflow_min, self.ws_inflow_max)
            # The turbulence intensity is a random number between TI_min and TI_max
            self.ti = self._random_uniform(self.TI_inflow_min, self.TI_inflow_max)
            # The wind direction is a random number between wd_min and wd_max
            self.wd = self._random_uniform(self.wd_inflow_min, self.wd_inflow_max)
        else:
            # wind resource
            dirs = np.arange(0, 360, 1)  # wind directions
            ws = np.arange(3, 25, 1)  # wind speeds
            local_wind = self.sample_site.local_wind(x=0, y=0, wd=dirs, ws=ws)
            freqs = local_wind.Sector_frequency_ilk[0, :, 0]
            As = local_wind.Weibull_A_ilk[0, :, 0]  # weibull A
            ks = local_wind.Weibull_k_ilk[0, :, 0]  # weibull k

            self.wd, self.ws = self._sample_site(dirs, As, ks, freqs)

            self.wd = np.clip(self.wd, self.wd_inflow_min, self.wd_inflow_max)
            self.ws = np.clip(self.ws, self.ws_inflow_min, self.ws_inflow_max)

            self.ti = self._random_uniform(
                self.TI_inflow_min, self.TI_inflow_max
            )  # The TI is still uniformly distributed.

    def _sample_site(self, dirs, As, ks, freqs):
        """
        sample wind direction and wind speed from the site
        """
        idx = self.np_random.choice(np.arange(dirs.size), 1, p=freqs)
        wd = dirs[idx]
        A = As[idx]
        k = ks[idx]
        ws = A * self.np_random.weibull(k)
        return wd.item(), ws.item()

    def _def_site(self):
        """
          We choose a random turbulence box and scale it to the correct TI and wind speed.
        This is repeated for the baseline if we have that.

        The turbulence box used for the simulation can be one of the following:
        - MannLoad: The turbulence box is loaded from predefined Mann turbulence box files.
        - MannGenerate: A random turbulence box is generated.
        - MannFixed: A fixed turbulence box is used with a constant seed.
        - Random: Specifies the 'box' as random turbulence.
        - None: Zero turbulence site.
        """

        if self.turbtype == "MannLoad":
            # Load the turbbox from predefined folder somewhere
            # selects one at random from the files that were already discovered in __init__
            tf_file = self.np_random.choice(self.TF_files)
            tf_agent = MannTurbulenceField.from_netcdf(filename=tf_file)
            tf_agent.scale_TI(TI=self.ti, U=self.ws)
            self.addedTurbulenceModel = SynchronizedAutoScalingIsotropicMannTurbulence()

        elif self.turbtype == "MannGenerate":
            # Create the turbbox with a random seed.
            # TODO this can be improved in the future.
            TF_seed = self.np_random.integers(0, 100000)
            tf_agent = MannTurbulenceField.generate(
                alphaepsilon=0.1,  # use correct alphaepsilon or scale later
                L=33.6,  # length scale
                Gamma=3.9,  # anisotropy parameter
                # numbers should be even and should be large enough to cover whole farm in all dimensions and time, see above
                Nxyz=(
                    4096,
                    512,
                    64,
                ),  # Maybe 8192 would be better. This is untimately farm size specific. But for now this is good enough.
                dxyz=(self.D / 20, self.D / 10, self.D / 10),  # Liew suggest /50
                seed=TF_seed,  # seed for random generator
            )
            tf_agent.scale_TI(TI=self.ti, U=self.ws)
            self.addedTurbulenceModel = SynchronizedAutoScalingIsotropicMannTurbulence()

        elif self.turbtype == "Random":
            # Specifies the 'box' as random turbulence
            TF_seed = self.np_random.integers(0, 100000)
            tf_agent = RandomTurbulence(ti=self.ti, ws=self.ws, seed=TF_seed)
            self.addedTurbulenceModel = AutoScalingIsotropicMannTurbulence()

        elif self.turbtype == "MannFixed":
            # Generates a fixed mann box
            TF_seed = 1234  # Hardcoded for now
            tf_agent = MannTurbulenceField.generate(
                alphaepsilon=0.1,  # use correct alphaepsilon or scale later
                L=33.6,  # length scale
                Gamma=3.9,  # anisotropy parameter
                # numbers should be even and should be large enough to cover whole farm in all dimensions and time, see above
                Nxyz=(2048, 512, 64),
                dxyz=(3.0, 3.0, 3.0),
                seed=TF_seed,  # seed for random generator
            )
            tf_agent.scale_TI(TI=self.ti, U=self.ws)
            self.addedTurbulenceModel = SynchronizedAutoScalingIsotropicMannTurbulence()

        elif self.turbtype == "None":
            # Zero turbulence site.
            tf_agent_seed = self.np_random.integers(
                2**31
            )  # Generate a seed from the main RNG
            tf_agent = RandomTurbulence(ti=0, ws=self.ws, seed=tf_agent_seed)

            self.addedTurbulenceModel = None  # AutoScalingIsotropicMannTurbulence()
        else:
            # Throw and error:
            raise ValueError("Invalid turbulence type specified")

        self.site = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_agent)

        if self.Baseline_comp:  # I am pretty sure we need to have 2 sites, as the flow simulation is run on the site, and the measurements are taken from the site.
            tf_base = copy.deepcopy(tf_agent)
            self.site_base = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_base)
            tf_base = None
        tf_agent = None
        tf_agent = None
        gc.collect()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. This is called at the start of every episode.
        - The wind conditions are sampled, and the site is set.
        - The flow simulation is run for the time it takes for the flow to develop.
        - The measurements are filled up with the initial values.

        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.timestep = 0

        # Sample global wind conditions and set the site
        self._set_windconditions()
        self._def_site()
        # Restart the measurement class. This is done to make sure that the measurements are not carried over from the last episode
        self._init_farm_mes()

        # Setup the wind turbines
        self._init_wts()

        if hasattr(self, "farm_measurements") and self.farm_measurements is not None:
            self.farm_measurements.np_random = self.np_random
            # print(f"DEBUG RESET WindFarmEnv: farm_measurements.np_random ID = {id(self.farm_measurements.np_random)}, state_key[:2] = {self.farm_measurements.np_random.bit_generator.state['state']['key'][:2]}")
        else:
            print(
                "WARNING: farm_measurements was not initialized prior to attempting to set its np_random in reset."
            )

        # This is the rated poweroutput of the turbine at the given ws. Used for reward scaling.
        self.rated_power = self.turbine.power(self.ws)

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
        )  # NOTE, we need this particlemotion to capture the yaw

        # Set the yaw angles of the farm
        # NOTE that I use yaw_start and not yaw_min/yaw_max. This is to make sure that the yaw angles are not too large at the start, but still not zero
        self.fs.windTurbines.yaw = self._yaw_init(
            min_val=-self.yaw_start,
            max_val=self.yaw_start,
            n=self.n_turb,
            yaws=self.yaw_initial,
        )

        # Calulate the time it takes for the flow to develop.
        turb_xpos = self.fs.windTurbines.rotor_positions_xyz[0, :]
        dist = turb_xpos.max() - turb_xpos.min()

        # Time it takes for the flow to travel from one side of the farm to the other
        t_inflow = dist / self.ws
        # The time it takes for the flow to develop. Also a bit extra.
        t_developed = int(t_inflow * self.burn_in_passthroughs)

        # Max allowed timesteps
        self.time_max = int(t_inflow * self.n_passthrough)

        if self.n_turb == 1:
            # For a single turbine, base the time on flow passing the rotor diameter
            self.time_max = int((self.D * self.n_passthrough) / self.ws)

        # Ensure time_max is at least 1 to allow at least one step
        self.time_max = max(1, self.time_max)

        # first we run the simulation the time it takes the flow to develop
        self.fs.run(t_developed)

        # Fill up our measurement queue first, with the ammount of steps we need to fill up
        for __ in range(self.steps_on_reset):
            windspeeds = []
            winddirs = []
            yaws = []
            powers = []

            for _ in range(self.sim_steps_per_env_step):
                # Step the flow simulation
                self.fs.step()

                # Make the measurements from the sensor
                self._take_measurements()

                if self.farm_measurements.turb_TI or self.farm_measurements.farm_TI:
                    for i in range(self.n_turb):
                        self.farm_measurements.turb_mes[i].add_hf_ws(self.current_ws[i])
                    if self.farm_measurements.farm_TI:
                        self.farm_measurements.farm_mes.add_hf_ws(
                            np.mean(self.current_ws)
                        )

                # Put them into the lists
                windspeeds.append(self.current_ws)
                winddirs.append(self.current_wd)
                yaws.append(self.current_yaw)
                powers.append(self.current_powers)

            mean_windspeed = np.mean(windspeeds, axis=0)
            mean_winddir = np.mean(winddirs, axis=0)
            mean_yaw = np.mean(yaws, axis=0)
            mean_power = np.mean(powers, axis=0)

            # Put them into the mes class, such that the _get_obs() call works as intendet.
            self.farm_measurements.add_measurements(
                mean_windspeed, mean_winddir, mean_yaw, mean_power
            )

            self.farm_pow_deq.append(mean_power.sum())

        # Do the same for the baseline farm
        if self.Baseline_comp:
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

            self.fs_baseline.windTurbines.yaw = self.fs.windTurbines.yaw
            self.fs_baseline.run(t_developed)

            if self.BaseController.split("_")[0] == "PyWake":
                # If we are using the PyWake agent as a baseline, we need to set it up
                self.pywake_agent.update_wind(
                    wind_speed=self.ws,
                    wind_direction=self.wd,
                    TI=self.ti,
                )
                self.pywake_ws = self.ws
                self.pywake_wd = self.wd
            for __ in range(self.hist_max):
                baseline_powers = []
                for _ in range(self.sim_steps_per_env_step):
                    self.fs_baseline.step()
                    baseline_powers.append(self.fs_baseline.windTurbines.power().sum())

                self.base_pow_deq.append(np.mean(baseline_powers, axis=0))

        observation = self._get_obs()
        info = self._get_info()

        # Init render can now be called as fs needs to be created first
        if self.render_mode == "human":
            self.init_render()

        return observation, info

    def _action_penalty(self):
        """
        This function calculates a penalty for the actions. This is used to penalize the agent for taking actions, and try and make it more stable
        """
        if (
            self.action_penalty < 0.001
        ):  # If the penalty is very small, then we dont need to calculate it
            return 0

        elif self.action_penalty_type == "Change":
            # The penalty is dependent on the change in values
            pen_val = np.mean(np.abs(self.old_yaws - self.fs.windTurbines.yaw))
        elif self.action_penalty_type == "Total":
            # The penalty is dependent on the total values
            pen_val = np.mean(np.abs(self.fs.windTurbines.yaw)) / self.yaw_max

        return self.action_penalty * pen_val

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

    def track_rew_none(self):
        """If we are not using power tracking, then just return 0"""
        return 0.0

    def track_rew_avg(self):
        """
        The reward is the negative difference between the power output and the power setpoint squared
        The reward is: - (power_agent - power_setpoint)^2
        """
        power_agent = np.mean(self.farm_pow_deq)
        return -((power_agent - self.power_setpoint) ** 2)

    def power_rew_baseline(self):
        """Calculate reward based on baseline farm comparison using available history"""

        # Use whatever history we have so far for averaging
        power_agent_avg = np.mean(self.farm_pow_deq)
        power_baseline_avg = np.mean(self.base_pow_deq)

        if power_baseline_avg == 0:
            print("The baseline power is zero. This is probably not good")
            print("self.farm_pow_deq: ", self.farm_pow_deq)
            print("self.base_pow_deq: ", self.base_pow_deq)
            0 / 0  # This will raise an error

        reward = power_agent_avg / power_baseline_avg - 1
        return reward

    def power_rew_avg(self):
        """Calculate power reward based on available history"""
        power_agent = np.mean(self.farm_pow_deq)
        reward = power_agent / self.n_turb / self.rated_power
        return reward

    def power_rew_none(self):
        """Return zero for the power reward"""
        return 0.0

    def power_rew_diff(self):
        """Calculate reward based on power difference over time"""
        power_latest = np.mean(
            list(
                itertools.islice(
                    self.farm_pow_deq,
                    self.power_len - self._power_wSize,
                    self.power_len,
                )
            )
        )
        power_oldest = np.mean(
            list(itertools.islice(self.farm_pow_deq, 0, self._power_wSize))
        )
        return (power_latest - power_oldest) / self.n_turb

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

        # Run multiple simulation steps for each environment step
        # Initialize list to store observations

        time_array = np.zeros(self.sim_steps_per_env_step, dtype=np.float32)
        windspeeds = np.zeros(
            (self.sim_steps_per_env_step, self.n_turb), dtype=np.float32
        )
        winddirs = np.zeros(
            (self.sim_steps_per_env_step, self.n_turb), dtype=np.float32
        )
        yaws = np.zeros((self.sim_steps_per_env_step, self.n_turb), dtype=np.float32)
        powers = np.zeros((self.sim_steps_per_env_step, self.n_turb), dtype=np.float32)
        baseline_powers = np.zeros(
            (self.sim_steps_per_env_step, self.n_turb), dtype=np.float32
        )
        yaws_baseline = np.zeros(
            (self.sim_steps_per_env_step, self.n_turb), dtype=np.float32
        )
        windspeeds_baseline = np.zeros(
            (self.sim_steps_per_env_step, self.n_turb), dtype=np.float32
        )

        if self.ActionMethod == "yaw":
            self.action_remaining = (
                action * self.yaw_step_env
            )  # Over all steps we want to move this ammout

        for j in range(self.sim_steps_per_env_step):
            self._adjust_yaws(action)  # Adjust the yaw angles of the agent farm

            # Step the flow simulation
            self.fs.step()

            # If we have baseline comparison, step it too
            if self.Baseline_comp:
                new_baseline_yaws = self._base_controller(
                    fs=self.fs_baseline, yaw_step=self.yaw_step_sim
                )
                self.fs_baseline.windTurbines.yaw = new_baseline_yaws
                self.fs_baseline.step()

                baseline_powers[j] = self.fs_baseline.windTurbines.power()
                yaws_baseline[j] = self.fs_baseline.windTurbines.yaw
                windspeeds_baseline[j] = np.linalg.norm(
                    self.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
                )
            # Make the measurements from the sensor
            self._take_measurements()

            if self.farm_measurements.turb_TI or self.farm_measurements.farm_TI:
                for i in range(self.n_turb):
                    self.farm_measurements.turb_mes[i].add_hf_ws(self.current_ws[i])
                # Also populate the farm-level hf buffer if it's being used for farm_TI
                if self.farm_measurements.farm_TI:
                    self.farm_measurements.farm_mes.add_hf_ws(np.mean(self.current_ws))

            # Put them into the lists
            windspeeds[j] = self.current_ws
            winddirs[j] = self.current_wd
            yaws[j] = self.current_yaw
            powers[j] = self.current_powers
            time_array[j] = self.fs.time

        mean_windspeed = np.mean(windspeeds, axis=0)
        mean_winddir = np.mean(winddirs, axis=0)
        mean_yaw = np.mean(yaws, axis=0)

        mean_power = np.mean(powers, axis=0)  # This is pr turbine

        # Put them into the mes class.
        self.farm_measurements.add_measurements(
            mean_windspeed, mean_winddir, mean_yaw, mean_power
        )
        self.farm_pow_deq.append(
            mean_power.sum()
        )  # Do the sum, because we want for the whole farm.
        if self.Baseline_comp:
            self.base_pow_deq.append(np.mean(baseline_powers, axis=0).sum())
        if np.any(np.isnan(self.farm_pow_deq)):
            raise Exception("NaN Power")

        observation = self._get_obs()
        info = self._get_info()

        # Add extra vals to info dict. Used for the agent_eval_fast
        info["time_array"] = time_array
        info["windspeeds"] = windspeeds
        # info['winddirs'] = winddirs
        info["yaws"] = yaws
        info["powers"] = powers

        if self.Baseline_comp:
            info["baseline_powers"] = baseline_powers
            info["yaws_baseline"] = yaws_baseline
            info["windspeeds_baseline"] = windspeeds_baseline

        # self.fs_time = self.fs.time  # Save the flow simulation timestep.
        # Save the power output of the farm
        # Calculate the reward
        # The power production reward with the scaling
        power_rew = self._power_rew() * self.Power_scaling
        # track_rew = self._track_rew()  #The power tracking reward. This is just a placeholder so far.
        track_rew = 0.0

        action_penalty = self._action_penalty()  # The penalty for the actions

        # The reward is: power reward - action penalty. This makes it possible to add a reward for power tracking, and/or damage, easily.
        reward = power_rew + track_rew - action_penalty

        # If we are at the end of the simulation, we truncate the agents.
        # Note that this is not the same as terminating the agents.
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        if self.timestep >= self.time_max:
            # terminated = {a: True for a in self.agents}
            truncated = True
            # Clean up the flow simulation. This is to make sure that we dont have a memory leak.
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
        else:
            truncated = False

        self.timestep += 1

        terminated = False

        return observation, reward, terminated, truncated, info

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
        """Render method required by PettingZoo API."""
        if self.render_mode == "rgb_array":
            # Return the RGB frame (for recording, saving, etc.)
            return self._render_frame()
        elif self.render_mode == "human":
            # Show the frame in a window
            frame = self._render_frame_for_human()
            plt.imshow(frame)
            plt.axis("off")
            plt.title("Wind Farm Environment - Render")
            plt.show(block=False)
            plt.pause(0.001)  # Pause to allow window to update

    def _render_frame_for_human(self, baseline=False):
        """Render the environment and return an RGB frame as a numpy array."""
        plt.ioff()  # Non-interactive mode
        fig, ax1 = plt.subplots(figsize=(18, 6))  # Create new figure for rendering

        if baseline:
            fs_use = self.fs_baseline
        else:
            fs_use = self.fs

        x_turb, y_turb = self.fs.windTurbines.positions_xyz[:2]
        self.a = np.linspace(-200 + min(x_turb), 1000 + max(x_turb), 250)
        self.b = np.linspace(-200 + min(y_turb), 200 + max(y_turb), 250)

        self.view = XYView(
            z=self.turbine.hub_height(), x=self.a, y=self.b, adaptive=False
        )

        uvw = fs_use.get_windspeed(self.view, include_wakes=True, xarray=True)

        wt = fs_use.windTurbines
        x_turb, y_turb = fs_use.windTurbines.positions_xyz[:2]
        yaw, tilt = wt.yaw_tilt()

        mesh = ax1.pcolormesh(
            uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest", cmap="viridis"
        )
        plt.colorbar(mesh, ax=ax1, label="Wind Speed (m/s)")

        # ax1.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest")
        WindTurbinesPW.plot_xy(
            fs_use.windTurbines,
            x_turb,
            y_turb,
            types=fs_use.windTurbines.types,
            wd=fs_use.wind_direction,
            ax=ax1,
            yaw=yaw,
            tilt=tilt,
        )

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # Get RGB only

        plt.close(fig)  # Avoid memory leaks
        return frame

    def _render_frame(self, baseline=False):
        """
        This is the rendering function.
        It renders the flow field and the wind turbines
        Can be much improved, but it is a start
        """

        plt.ion()
        ax1 = plt.gca()

        if baseline:
            fs_use = self.fs_baseline
        else:
            fs_use = self.fs

        uvw = fs_use.get_windspeed(self.view, include_wakes=True, xarray=True)

        wt = fs_use.windTurbines
        x_turb, y_turb = fs_use.windTurbines.positions_xyz[:2]
        yaw, tilt = wt.yaw_tilt()

        # Redudante init_render?
        """
        #Init render can now be called as fs needs to be created first
        #if self.render_mode == "human":
        #    self.init_render()
        """
        # [0] is the u component of the wind speed
        plt.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest")
        WindTurbinesPW.plot_xy(
            fs_use.windTurbines,
            x_turb,
            y_turb,
            types=fs_use.windTurbines.types,
            wd=fs_use.wind_direction,
            ax=ax1,
            yaw=yaw,
            tilt=tilt,
        )
        ax1.set_title("Flow field at {} s".format(fs_use.time))
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if self.render_mode == "human":
            pass

        else:
            # If we have the RGB mode.
            pass

    def close(self):
        plt.close()
        if self.Baseline_comp:
            self.fs_baseline = None
            self.site_base = None
        self.fs = None
        self.site = None
        self.farm_measurements = None
        self.fs = None
        self.site = None
        self.farm_measurements = None
        gc.collect()

    def plot_frame(self, baseline=False):
        """
        Plots a single frame of the flow field and the wind turbines
        """
        self.init_render()
        self._render_frame(baseline=baseline)

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
