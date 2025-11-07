from typing import Any, Optional, Tuple
import numpy as np
import numpy.typing as npt
from .base_agent import BaseAgent

"""
This agent takes random actions, and is used to test the environment.
"""


class RandomAgent(BaseAgent):
    def __init__(self, env: Optional[Any] = None) -> None:
        """
        Initialize random agent.

        Args:
            env: Gymnasium environment with action_space attribute
        """
        # This is used in a hasattr in the AgentEval class/function.
        self.UseEnv = True
        self.env = env

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[npt.NDArray[np.float64], None]:
        """
        Sample a random action from the environment's action space.

        Note: Ignores obs and deterministic arguments.

        Returns:
            Tuple of (random_action, None)
        """

        action = self.env.action_space.sample()

        return action, None
