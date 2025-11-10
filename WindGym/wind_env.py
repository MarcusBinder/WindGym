from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from typing import Any

"""This is the base class for the wind environment
All the other wind environments will inherit from this class.

This just contains some helper functions that are used in the other classes
"""


class WindEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def _scale_val(self, val: NDArray[np.floating], min_val: float, max_val: float) -> NDArray[np.floating]:
        # Scale the value from -1 to 1
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def _random_choice(self, min_val: float, max_val: float, n: int) -> float:
        # Return a random choice between min_val and max_val
        return self.np_random.choice(np.linspace(min_val, max_val, n))

    def _random_uniform(self, min_val: float, max_val: float, n: int | None = None) -> float:
        # Return a random value between min_val and max_val
        # Note that we ignore the n parameter
        return self.np_random.uniform(low=min_val, high=max_val)

    def _return_zeros(self, **kwargs: Any) -> NDArray[np.floating]:
        # Return a numpy array of zeros with the specified shape
        return np.zeros(kwargs["n"])

    def _randoms_uniform(self, **kwargs: Any) -> NDArray[np.floating]:
        # Return a numpy array of n random values between min_val and max_val
        return self.np_random.uniform(
            low=kwargs["min_val"], high=kwargs["max_val"], size=kwargs["n"]
        )

    def _random_normal(self, **kwargs: Any) -> NDArray[np.floating]:
        # Return a numpy array of n random values with a normal distribution
        # print("Distributing with mean: ", kwargs["mean"], " and std: ", kwargs["std"])
        return self.np_random.normal(
            loc=kwargs["mean"], scale=kwargs["std"], size=kwargs["n"]
        )

    def _defined_yaw(self, **kwargs: Any) -> NDArray[np.floating]:
        # Set the yaw values to a specified value
        # If the length of the yaw values is equal to the number of turbines, return the yaw values
        # If the length of the yaw values is 1, we assume that all turbines should have that yaw value
        yaw_vals = kwargs["yaws"]
        if len(yaw_vals) == self.n_turb:
            return yaw_vals
        elif len(yaw_vals) == 1:
            return np.ones(self.n_turb) * yaw_vals[0]
        else:
            raise ValueError(
                "The specified yaw values are not the right length. Expected either 1 or {self.n_turb} values."
            )
