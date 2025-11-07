"""
This file contains the basic controllers for the wind farm environment.
These can be used either as a base controller for comparing the performance directly in hte reward function, or it can used to evaluate the farm using a "normal" base controller.

"""

from typing import Any
import numpy as np
import numpy.typing as npt


def _compute_step_limited_adjustment(
    offset: npt.NDArray[np.float64], max_step: float
) -> npt.NDArray[np.float64]:
    """
    Compute step-limited adjustment towards target, preventing overshoot.

    Args:
        offset: Desired adjustment (signed values)
        max_step: Maximum absolute step size

    Returns:
        Step-limited adjustment (same sign as offset, magnitude <= max_step)
    """
    step_dir = np.sign(offset)
    step_scale = np.abs(offset)
    step_scale[step_scale > max_step] = max_step
    return step_dir * step_scale


def local_yaw_controller(fs: Any, yaw_step: float = 1) -> npt.NDArray[np.float64]:
    """
    Compute yaw control action based on local turbine wind conditions.

    Aligns each turbine with its local wind direction (computed from rotor-averaged wind speed).
    Uses greedy step-limited control to minimize yaw offset without overshooting.

    Args:
        fs: Flow simulation object with windTurbines attribute
        yaw_step: Maximum yaw change per step (degrees)

    Returns:
        Array of new yaw angles for each turbine (degrees)
    """
    # Fist we get the current yaw offset, in relation to the "global wind"
    yaw_baseline = fs.windTurbines.yaw

    # Then by taking the inverse tan of the wind speed components, we get the LOCAL wind direction
    wind_dir_baseline = np.rad2deg(
        np.arctan(
            fs.windTurbines.rotor_avg_windspeed[:, 1]
            / fs.windTurbines.rotor_avg_windspeed[:, 0]
        )
    )

    # The desired yaw offset is the difference between the baseline yaw and the baseline wind direction
    yaw_offset = wind_dir_baseline - yaw_baseline

    # Compute step-limited adjustment
    yaw_action = _compute_step_limited_adjustment(yaw_offset, yaw_step)

    new_yaw = yaw_baseline + yaw_action

    return new_yaw


def global_yaw_controller(fs: Any, yaw_step: float = 1) -> npt.NDArray[np.float64]:
    """
    Compute yaw control action based on global wind direction.

    Aligns all turbines with the global wind direction (ignores local wake effects).
    Uses greedy step-limited control to minimize yaw offset without overshooting.

    Args:
        fs: Flow simulation object with windTurbines attribute
        yaw_step: Maximum yaw change per step (degrees)

    Returns:
        Array of new yaw angles for each turbine (degrees)
    """

    # The current yaw offset, in relation to the "global wind"
    yaw_offset = fs.windTurbines.yaw

    # Compute step-limited adjustment
    yaw_action = _compute_step_limited_adjustment(yaw_offset, yaw_step)

    new_yaw = yaw_offset - yaw_action

    return new_yaw
