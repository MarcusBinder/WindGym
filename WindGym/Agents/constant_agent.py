from typing import Any, List, Tuple, Union
import numpy as np
import numpy.typing as npt
from .base_agent import BaseAgent

"""
The ConstantAgent is a class used for evaluating the performance, under predefinned constant yaw angles.
"""


class ConstantAgent(BaseAgent):
    def __init__(
        self,
        yaw_angles: Union[List[float], npt.NDArray[np.float64]],
        yaw_max: float = 45,
        yaw_min: float = -45,
    ) -> None:
        """
        Initialize constant yaw agent.

        Args:
            yaw_angles: Fixed yaw angles to apply (in degrees)
            yaw_max: Maximum yaw angle
            yaw_min: Minimum yaw angle
        """
        self.UseEnv = True
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min

        # Turn the yaw_angles into a numpy array if it is a list
        if isinstance(yaw_angles, list):
            yaw_angles = np.array(yaw_angles)

        self.yaw_angles = yaw_angles

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[npt.NDArray[np.float64], None]:
        """
        Return constant yaw angles as scaled actions.

        Note: Ignores obs and deterministic arguments.

        Returns:
            Tuple of (scaled_constant_yaw_action, None)
        """

        action = self.scale_yaw(self.yaw_angles)

        return action, None
