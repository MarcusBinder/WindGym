"""
Base container for all "basic" agents.
We MUST have a predict function, that returns the action and the state.

The scale_yaw function is used to scale the yaw angles to be between -1 and 1.
"""

from typing import Any, Tuple
import numpy as np
import numpy.typing as npt


class BaseAgent:
    def __init__(self, yaw_max: float = 45, yaw_min: float = -45) -> None:
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[npt.NDArray[np.float64], None]:
        """
        Predict the action given the observation.

        Returns:
            Tuple of (action, state) where state is typically None for stateless agents
        """
        pass

    def scale_yaw(self, yaws: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Scale the yaw angles to be between -1 and 1.

        Args:
            yaws: Yaw angles in degrees

        Returns:
            Scaled yaw angles in range [-1, 1]
        """
        action = (yaws - self.yaw_min) / (self.yaw_max - self.yaw_min) * 2 - 1

        return action

    def unscale_yaw(self, action: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Unscale the action to the yaw range.

        Args:
            action: Scaled action in range [-1, 1]

        Returns:
            Yaw angles in degrees
        """
        yaws = (action + 1.0) / 2.0 * (self.yaw_max - self.yaw_min) + self.yaw_min

        return yaws
