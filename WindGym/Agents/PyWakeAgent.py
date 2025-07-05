import numpy as np
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.hornsrev1 import V80

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

import matplotlib.pyplot as plt
from .BaseAgent import BaseAgent
import warnings

from scipy.interpolate import RegularGridInterpolator

"""
The PyWakeAgent is a class that is used to optimize the yaw angles of a wind farm using the PyWake library.
It interfaces with the AgentEval class in the dtu_wind_gym library.
Based on the global wind conditons it can optimize the yaw angles and then use them during the simulation.
"""


class PyWakeAgent(BaseAgent):
    def __init__(
        self,
        x_pos,
        y_pos,
        wind_speed=8,
        wind_dir=270,
        TI=0.07,
        yaw_max=45,
        yaw_min=-45,
        refine_pass_n=6,
        yaw_n=7,
        look_up=False,  # If true use interpolation to get the yaw angles
        turbine=V80(),
        env=None,
    ):
        # This is used in a hasattr in the AgentEval class.
        self.pywakeagent = True
        self.optimized = False  # Is false before we have optimized the farm.
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        self.look_up = look_up

        self.UseEnv = True
        self.env = env

        # choosing the flow cases for the optimization
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_dir])
        self.TI = TI

        self.refine_pass_n = refine_pass_n
        self.yaw_n = yaw_n

        # Define the farm.
        site = LillgrundSite()
        self.turbine = turbine

        # Check if x_pos or y_pos are lists if so then convert them to numpy arrays
        if isinstance(x_pos, list):  # pragma: no cover
            x_pos = np.array(x_pos)
        if isinstance(y_pos, list):  # pragma: no cover
            y_pos = np.array(y_pos)
        if len(x_pos) != len(y_pos):  # pragma: no cover
            raise ValueError("x_pos and y_pos must have the same length.")

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_wt = len(x_pos)

        site.initial_position = np.array([x_pos, y_pos]).T

        self.wf_model = Blondel_Cathelain_2020(
            site,
            turbine,
            turbulenceModel=CrespoHernandez(),
            deflectionModel=JimenezWakeDeflection(),
        )

        # initial condition of yaw angles
        self.yaw_zero = np.zeros((self.n_wt, 1, 1))
        self.reset()

        if self.look_up:
            # If we want to use interpolation, we create a lookup table
            self.make_lookup()

    def update_wind(self, wind_speed, wind_direction, TI):
        """
        Update the wind conditions for the agent.
        """
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_direction])
        self.TI = TI
        self.reset()

    def make_lookup(self):
        """
        Create a lookup table for the yaw angles.
        This is done as we can save time by doing it once and then use it later.
        """

        wd_array = np.arange(self.env.wd_min - 10, self.env.wd_max + 10 + 1, 1)
        ws_array = np.arange(5, 25 + 1, 1)
        TIs = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]

        # Create a grid of wind directions and wind speeds
        yaw_results = np.zeros((self.n_wt, len(wd_array), len(ws_array), len(TIs)))

        # Do the optimization
        for j in range(len(TIs)):
            yaw_array = yaw_optimizer_srf_vect(
                x=self.x_pos,
                y=self.y_pos,
                wffm=self.wf_model,
                wd=wd_array,
                ws=ws_array,
                ti=np.array([TIs[j]]),
                refine_pass_n=self.refine_pass_n,
                yaw_n=self.yaw_n,
                nn_cpu=1,
                sort_reverse=False,
            )

            yaw_results[:, :, :, j] = yaw_array

        # Move the axes so that the last axis is the turbine axis
        yaw_results = np.moveaxis(yaw_results, 0, -1)
        self.interpolator = RegularGridInterpolator(
            (wd_array, ws_array, TIs),  # axes
            yaw_results,  # data for turbine i
            bounds_error=False,  # allow extrapolation
            fill_value=None,  # extrapolate instead of NaN
        )

    def use_lookup(self):
        """
        Use the lookup table to get the yaw angles for the current wind conditions.
        """
        # Get the yaw angles from the interpolator
        yaws = self.interpolator((self.wdir, self.wsp, self.TI))
        self.optimized_yaws = yaws.squeeze()
        # return yaws.squeeze()s

    def reset(self):
        """
        Reset the wind things for the objective.
        """
        self.optimized = False

    def optimize(self):
        """
        Optimizes the yaw angles of the wind farm.
        """
        yaws = yaw_optimizer_srf_vect(
            x=self.x_pos,
            y=self.y_pos,
            wffm=self.wf_model,
            wd=self.wdir,
            ws=self.wsp,
            ti=self.TI,
            refine_pass_n=self.refine_pass_n,
            yaw_n=self.yaw_n,
            nn_cpu=1,
            sort_reverse=False,
        )

        self.optimized_yaws = yaws.squeeze()
        self.optimized = True

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        # Only optimize if we have not done it yet, and if we are not using the lookup table.
        if not self.look_up and self.optimized is False:
            self.optimize()

        if self.look_up:
            self.use_lookup()

        # Get the optimal yaw angles.
        optimal_yaws = self.optimized_yaws

        if self.env.ActionMethod == "wind":
            # If the action method is 'wind', we return the set point yaw angles directly.
            # print("Using wind action method")
            action = self.scale_yaw(optimal_yaws)
        elif self.env.ActionMethod == "yaw":
            # If using yaw based steering, we need to retun the yaw angles differently

            yaw_goal = optimal_yaws
            yaw_offset = self.env.current_yaw
            yaw_step = self.env.yaw_step_env  # How much we can change pr step

            step_dir = np.sign(yaw_goal - yaw_offset)
            step_scale = np.abs(yaw_goal - yaw_offset)
            # here we replace all values that are larger then the max, with the max
            step_scale[step_scale > yaw_step] = yaw_step
            action = step_dir * step_scale / self.env.yaw_step_env

        return action, None

    def calc_power(self, yaws):
        """
        Calculates the power of the farm, given the yaw angles.
        Inputs are the yaw angles in degrees.
        Returns the total power of the farm.
        """
        power = (
            self.wf_model(
                x=self.x_pos,
                y=self.y_pos,
                ws=self.wsp,
                wd=self.wdir,
                TI=self.TI,
                tilt=0,
                yaw=yaws,
            )["Power"]
            .sum()
            .values
        )
        return power

    def plot_flow(self):
        """
        Plot the flowfield of the wind farm.
        """
        if self.optimized is False:
            self.optimize()
        simulationResult = self.wf_model(
            self.x_pos,
            self.y_pos,
            wd=self.wdir,
            ws=self.wsp,
            yaw=self.optimized_yaws,
            tilt=0,
        )
        plt.figure(figsize=(12, 4))
        simulationResult.flow_map().plot_wake_map()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()


def yaw_optimizer_srf_vect(
    x, y, wffm, wd, ws, ti=0.04, refine_pass_n=4, yaw_n=5, nn_cpu=1, sort_reverse=False
):
    """
    This is the Serial-Refine Method for yaw optimization, implemented in PyWake.
    This was done by Deniz. I just copied the function into this file.

    Optimizes turbine yaw angles over arrays of wind directions and wind speeds
    using a Serial-Refine Method for PyWake.

    This version vectorizes the wind direction (wd) dimension by evaluating the candidate
    yaw configurations for all wind directions at once. The candidate offset dimension is looped
    over (typically small), while wd is fed in vectorized.

    Parameters
    ----------
    x : array_like, shape (n_wt,)
        x coordinates of the turbines.
    y : array_like, shape (n_wt,)
        y coordinates of the turbines.
    wffm : EngineeringWindFarmModel Object
        PyWake wind farm flow model.
    wd : array_like or float
        Wind direction(s) (in meteorological convention, degrees).
    ws : array_like or float
        Wind speed(s) (m/s).
    ti : array_like or float
        Turbulence intensity.
    refine_pass_n : int, optional
        Number of refine passes.
    yaw_n : int, optional
        Number of candidate yaw offsets to test at each update step.
    nn_cpu : int, optional
        Number of CPUs to use.
    sort_reverse : bool, optional
        Whether to reverse turbine sorting (downstream-to-upstream).

    Returns
    -------
    yaw_opt : ndarray, shape (n_wt, n_wd, n_ws)
        The optimized yaw angles for each turbine, wind direction, and wind speed.
    """
    yaw_max = 30  # Maximum yaw angle (degrees)
    # add delta to wd to fix perfect alignment cases with 2 maxima
    wd = np.atleast_1d(wd) + 1e-3  # shape: (n_wd,)
    ws = np.atleast_1d(ws)  # shape: (n_ws,)
    ti = np.atleast_1d(ti)
    n_wt = len(x)
    n_wd = len(wd)
    n_ws = len(ws)

    # Initialize yaw angles: shape (n_wt, n_wd, n_ws)
    yaw_opt = np.zeros((n_wt, n_wd, n_ws))

    # Compute baseline power for all conditions.
    po = wffm(x, y, wd=wd, ws=ws, TI=ti, yaw=yaw_opt, tilt=0, n_cpu=nn_cpu).Power.values
    # Sum power over turbines -> shape (n_wd, n_ws)
    power_current = np.sum(po, axis=0)

    # Compute turbine ordering for each wind direction.
    # Convert meteorological wd to mathematical angle (so that x aligns with wind)
    theta = np.radians((270 - wd) % 360)  # shape: (n_wd,)
    # Compute rotated x-coordinate: shape (n_wd, n_wt)
    x_rotated = (
        x[None, :] * np.cos(theta)[:, None] + y[None, :] * np.sin(theta)[:, None]
    )
    # For each wind direction, sort turbines upstream-to-downstream.
    turbines_ordered = np.argsort(x_rotated, axis=1)  # shape: (n_wd, n_wt)
    if sort_reverse:
        turbines_ordered = np.argsort(-x_rotated, axis=1)
        # print('serial-refine sorting upstream to downstream')

    current_offset_range = yaw_max

    # Begin refine passes.
    for s in range(refine_pass_n):
        # print(f"Serial refine pass {s + 1}/{refine_pass_n}")
        # Create a symmetric grid of candidate offsets.
        candidate_offsets = np.linspace(
            -current_offset_range, current_offset_range, yaw_n
        )
        current_offset_range /= 2.0

        # Loop over turbine ordering positions.
        for pos in range(n_wt):
            # For each wd, select the turbine at ordering position "pos"
            turb_idx = turbines_ordered[:, pos]  # shape: (n_wd,)
            # Get current yaw for these turbines (for each wd): shape (n_wd, n_ws)
            current_yaws = yaw_opt[turb_idx, np.arange(n_wd), :]
            # Compute candidate yaw values: for each wd, candidate_yaws has shape (yaw_n, n_ws)
            candidate_yaws = current_yaws[:, None, :] + candidate_offsets[None, :, None]

            # Prepare to store candidate total power (summed over turbines) for each wd and candidate.
            candidate_power = np.empty((n_wd, yaw_n, n_ws))

            # Loop over candidate offsets (yaw_n is typically small)
            for j in range(yaw_n):
                # Create a candidate yaw configuration for all turbines, for all wd and ws.
                candidate_yaw_config = np.copy(yaw_opt)  # shape: (n_wt, n_wd, n_ws)
                # For each wd, update only the turbine being updated with its candidate yaw.
                candidate_yaw_config[turb_idx, np.arange(n_wd), :] = candidate_yaws[
                    :, j, :
                ]

                # Evaluate the candidate configuration for all wd at once.
                p = wffm(
                    x,
                    y,
                    wd=wd,
                    ws=ws,
                    TI=ti,
                    yaw=candidate_yaw_config,
                    tilt=0,
                    n_cpu=nn_cpu,
                ).Power.values
                # Sum power over turbines: shape (n_wd, n_ws)
                candidate_power[:, j, :] = np.sum(p, axis=0)

            # For each wd and wind speed, select the candidate offset that yields the highest power.
            best_candidate_idx = np.argmax(
                candidate_power, axis=1
            )  # shape: (n_wd, n_ws)
            best_candidate_power = np.max(
                candidate_power, axis=1
            )  # shape: (n_wd, n_ws)

            # Update yaw if improvement is found.
            ws_idx = np.arange(n_ws)
            for i in range(n_wd):
                improvement_mask = best_candidate_power[i] > power_current[i]
                if np.any(improvement_mask):
                    # For element-wise selection, index with ws_idx.
                    best_yaws = candidate_yaws[
                        i, best_candidate_idx[i], ws_idx
                    ]  # shape: (n_ws,)
                    yaw_opt[turb_idx[i], i, improvement_mask] = best_yaws[
                        improvement_mask
                    ]
                    power_current[i, improvement_mask] = best_candidate_power[
                        i, improvement_mask
                    ]
    if np.any(yaw_opt < -yaw_max) or np.any(yaw_opt > yaw_max):
        warnings.warn("Optimal setpoints outside [-yaw_max, yaw_max] range.")
        yaw_opt = np.clip(yaw_opt, a_min=-yaw_max, a_max=yaw_max)
    return yaw_opt
