from typing import Any, Literal, Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..BasicControllers import (
    local_yaw_controller,
    global_yaw_controller,
)
from .base_agent import BaseAgent

"""
This is the basic agent class. It is used to create a simple agent that can be used in the AgentEval class.
The agent will always try and get to zero yaw offset
"""


class GreedyAgent(BaseAgent):
    def __init__(
        self,
        type: Literal["local", "global"] = "local",
        yaw_max: float = 45,
        yaw_min: float = -45,
        yaw_step: float = 1,
        env: Optional[Any] = None,
    ) -> None:
        """
        Initialize greedy yaw controller agent.

        Args:
            type: Controller type - "local" for turbine-local or "global" for farm-global control
            yaw_max: Maximum yaw angle
            yaw_min: Minimum yaw angle
            yaw_step: Maximum yaw change per step (degrees)
            env: Gymnasium environment instance
        """
        # This is used in a hasattr in the AgentEval class/function.
        self.UseEnv = True
        self.env = env

        # these are initial values, but they should be overwritten in the eval call
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        # This should be 1, as the action is scaled to be between -1 and 1.
        self.yaw_step = yaw_step

        if type == "local":
            self.controller = local_yaw_controller
        elif type == "global":
            self.controller = global_yaw_controller

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[npt.NDArray[np.float64], None]:
        """
        Compute greedy yaw control action to minimize yaw offset.

        Note: Ignores obs and deterministic arguments.

        Returns:
            Tuple of (scaled_yaw_action, None)
        """
        yaw_goal = self.controller(fs=self.env.fs, yaw_step=self.yaw_step)

        action = self.scale_yaw(yaw_goal)

        return action, None
