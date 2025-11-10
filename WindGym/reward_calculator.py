"""
Reward calculation utilities for wind farm reinforcement learning environments.

This module provides a RewardCalculator class that computes different reward
components for wind farm control optimization.
"""

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .wind_farm_env import WindFarmEnv

# Constants for reward calculations
NEGLIGIBLE_ACTION_PENALTY_THRESHOLD = 0.001


class RewardCalculator:
    """
    Calculates rewards for wind farm control optimization.

    Supports multiple reward schemes:
    - Power-based rewards (baseline comparison, average, difference over time)
    - Power tracking rewards (for setpoint tracking tasks)
    - Action penalties (to encourage stability)
    """

    def __init__(self, env: "WindFarmEnv") -> None:
        """
        Initialize reward calculator with environment reference.

        Args:
            env: The WindFarmEnv instance to calculate rewards for
        """
        self.env = env

    def action_penalty(self) -> float:
        """
        Calculate penalty for control actions to encourage stability.

        Supports two penalty types:
        - "change": Penalizes yaw angle changes between steps
        - "total": Penalizes absolute yaw misalignment

        Returns:
            Action penalty value (non-negative)
        """
        if self.env.action_penalty < NEGLIGIBLE_ACTION_PENALTY_THRESHOLD:
            # Skip calculation for negligible penalties
            return 0.0

        penalty_type = (self.env.action_penalty_type or "").lower()

        if penalty_type == "change":
            # Penalize yaw changes from previous step
            pen_val = float(
                np.mean(np.abs(self.env.old_yaws - self.env.fs.windTurbines.yaw))
            )
        elif penalty_type == "total":
            # Penalize absolute yaw offset from zero
            pen_val = float(
                np.mean(np.abs(self.env.fs.windTurbines.yaw))
                / max(1e-6, self.env.yaw_max)
            )
        else:
            pen_val = 0.0

        return float(self.env.action_penalty) * pen_val

    def track_reward_none(self) -> float:
        """Return zero when power tracking is disabled."""
        return 0.0

    def track_reward_avg(self) -> float:
        """
        Calculate reward for tracking a power setpoint.

        Returns negative squared error between farm power and setpoint.
        Lower values (closer to zero) indicate better tracking.

        Returns:
            Negative squared tracking error
        """
        power_agent = np.mean(self.env.farm_pow_deq)
        return -((power_agent - self.env.power_setpoint) ** 2)

    def power_reward_baseline(self) -> float:
        """
        Calculate reward based on farm power relative to baseline controller.

        Compares agent's farm power to a baseline greedy controller.
        Positive values indicate the agent outperforms the baseline.

        Returns:
            Relative power gain compared to baseline (unbounded)

        Raises:
            ZeroDivisionError: If baseline power is zero (indicates config error)
        """
        power_agent_avg = np.mean(self.env.farm_pow_deq)
        power_baseline_avg = np.mean(self.env.base_pow_deq)

        if power_baseline_avg == 0:
            # Note: Preserving original behavior which triggers ZeroDivisionError
            # This indicates a critical configuration issue
            # Intentionally raise using division to preserve original behavior
            # pylint: disable=pointless-statement
            0 / 0  # noqa: B018

        return power_agent_avg / power_baseline_avg - 1

    def power_reward_avg(self) -> float:
        """
        Calculate normalized power reward based on rated capacity.

        Returns farm power normalized by number of turbines and rated power.
        Values typically in range [0, 1] for normal operation.

        Returns:
            Normalized power production
        """
        power_agent = np.mean(self.env.farm_pow_deq)
        return power_agent / self.env.n_turb / self.env.rated_power

    def power_reward_none(self) -> float:
        """Return zero when power reward is disabled."""
        return 0.0

    def power_reward_diff(self) -> float:
        """
        Calculate reward based on power improvement over time window.

        Compares recent power production to older measurements in the buffer.
        Positive values indicate increasing power trend.

        Returns:
            Power difference between recent and old window (per turbine)
        """
        window_size = self.env._power_wSize  # noqa: SLF001

        # Average power from most recent window
        power_latest = np.mean(
            list(
                itertools.islice(
                    self.env.farm_pow_deq,
                    self.env.power_len - window_size,
                    self.env.power_len,
                )
            )
        )

        # Average power from oldest window
        power_oldest = np.mean(
            list(itertools.islice(self.env.farm_pow_deq, 0, window_size))
        )

        return (power_latest - power_oldest) / self.env.n_turb

    def calculate_reward(self) -> float:
        """
        Calculate total reward for the current step.

        Combines power reward, tracking reward, and action penalty based on
        the environment's configured reward scheme.

        Returns:
            Total scalar reward for the current step
        """
        # Calculate power production reward component
        power_rew = self.env._power_rew() * self.env.Power_scaling  # noqa: SLF001

        # Calculate tracking reward component (currently unused)
        track_rew = self.env._track_rew()  # noqa: SLF001

        # Calculate action penalty component
        action_pen = self.action_penalty()

        # Total reward: power + tracking - penalty
        return power_rew + track_rew - action_pen
