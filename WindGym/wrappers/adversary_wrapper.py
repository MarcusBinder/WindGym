from gymnasium.vector.vector_env import ArrayType, VectorEnv
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Dict, Any, List, Optional


class AdversaryWrapper(gym.wrappers.vector.RecordEpisodeStatistics):
    """
    This wraps the RecordEpisodeStatistics Wrapper.
    It also adds a queue to store the mean power of the episodes. This is used for the logging during training.
    Could also be expanded upon to include more statistics if wanted.
    """

    def __init__(self, env: VectorEnv, buffer_length=100):
        super().__init__(env=env, buffer_length=buffer_length)

        # The Queue to store the mean power of the episodes.
        self.mean_power_queue = deque(maxlen=buffer_length)
        # These will be initialized to the correct `num_envs` size in `reset`
        self.episode_powers: np.ndarray = np.zeros(())
        self.last_dones: np.ndarray = np.zeros((), dtype=bool)

        # Also add the baseline here:
        self.mean_power_queue_baseline = deque(maxlen=buffer_length)
        self.episode_powers_baseline: np.ndarray = np.zeros(
            ()
        )  # Will be correctly set in reset

    def reset(self, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # Initialize to num_envs length, as they were 0-dim before
        self.episode_powers = np.zeros(self.num_envs)
        self.episode_powers_baseline = np.zeros(self.num_envs)

        # `super().reset()` already sets `self.prev_dones` based on the initial state of the environments.
        # `self.last_dones` is used in `step()` to know which environments were done *before* the current step,
        # so it should be initialized with the `prev_dones` from this reset.
        self.last_dones = self.prev_dones.copy()

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[
        ObsType, ArrayType, ArrayType, ArrayType, List[Dict[str, Any]]
    ]:  # Corrected return type hint
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            aggregated_infos,  # Renamed for clarity, this is the dict from AsyncVectorEnv
        ) = super().step(actions)  # Call to parent `RecordEpisodeStatistics.step()`

        # --- Phase 1: De-aggregate `infos` for easier processing ---
        # Create a list of dictionaries, one for each environment
        deaggregated_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        for key, value in aggregated_infos.items():  # Changed `value_list` to `value`
            if key == "episode" or key == "final_info":
                # These keys are special. `RecordEpisodeStatistics` handles them to be lists
                # (where each element is a dict or None for the respective env).
                # However, for num_envs=1, sometimes they might not be lists.
                if (
                    isinstance(value, (list, np.ndarray))
                    and len(value) == self.num_envs
                ):
                    for i in range(self.num_envs):
                        if value[i] is not None:
                            deaggregated_infos[i][key] = value[i]
                elif self.num_envs == 1:
                    # If num_envs is 1, and it's not a list, assume it's the direct dict for env 0.
                    if value is not None:
                        deaggregated_infos[0][key] = value
                # else:
                #       print(f"DEBUG: Skipping unhandled 'episode'/'final_info' structure for key {key}. Value type: {type(value)}")
            elif isinstance(value, (list, np.ndarray)) and len(value) == self.num_envs:
                # This branch handles most aggregated keys (e.g., 'Power agent', 'yaw/agent_turbine_0')
                # where the value is a list/array of length `num_envs`.
                for i in range(self.num_envs):
                    deaggregated_infos[i][key] = value[i]
            else:
                # This `else` block is for keys where `value` is NOT a list/array of length `num_envs`.
                # This would catch global scalars (e.g., 'time_since_reset' if added globally).
                # Copy this scalar value to all deaggregated info dicts.
                for i in range(self.num_envs):
                    deaggregated_infos[i][key] = value
        # --- End Phase 1 ---

        # --- Phase 2: Accumulate episode-wise power statistics ---
        # Accumulate power for ALL environments in this step, even if they terminate in this step.
        for i in range(self.num_envs):
            current_info = deaggregated_infos[i]

            if "Power agent" in current_info:
                # current_info["Power agent"] is expected to be a scalar from ParametricAdversarialEnv
                self.episode_powers[i] += current_info["Power agent"]
            # else: print(f"Warning: 'Power agent' missing in deaggregated info for env {i}. Assuming 0 contribution.")

            if "Power baseline" in current_info:
                self.episode_powers_baseline[i] += current_info["Power baseline"]
            # else: print(f"Warning: 'Power baseline' missing in deaggregated info for env {i}. Assuming 0 contribution.")
        # --- End Phase 2 ---

        # --- Phase 3: Populate queues and `episode` summary for environments that just finished ---
        # `self.prev_dones` (from the parent class) identifies which environments completed in THIS step.
        dones_in_this_step = np.logical_or(terminations, truncations)
        if np.sum(dones_in_this_step) > 0:  # Check if any environment just finished
            for i in np.where(dones_in_this_step)[
                0
            ]:  # Iterate over indices of completed episodes
                # Ensure episode_lengths is non-zero to avoid division by zero
                # The parent class (RecordEpisodeStatistics) is responsible for updating episode_lengths
                if self.episode_lengths[i] > 0:
                    mean_power_agent_for_episode = (
                        self.episode_powers[i] / self.episode_lengths[i]
                    )
                    self.mean_power_queue.append(mean_power_agent_for_episode)

                    # Only append baseline if it was present in the info for this episode
                    if "Power baseline" in deaggregated_infos[i]:
                        mean_power_baseline_for_episode = (
                            self.episode_powers_baseline[i] / self.episode_lengths[i]
                        )
                        self.mean_power_queue_baseline.append(
                            mean_power_baseline_for_episode
                        )

                    # Append these custom metrics to the 'episode' summary within the deaggregated info dict
                    # This is how `VectorEnvLogger` will access them.
                    # `deaggregated_infos[i]['episode']` should exist if the episode finished and was handled by parent.
                    if (
                        "episode" in deaggregated_infos[i]
                        and deaggregated_infos[i]["episode"] is not None
                    ):
                        deaggregated_infos[i]["episode"]["mean_power_agent"] = (
                            mean_power_agent_for_episode
                        )
                        if "Power baseline" in deaggregated_infos[i]:
                            deaggregated_infos[i]["episode"]["mean_power_baseline"] = (
                                mean_power_baseline_for_episode
                            )
                    # else:
                    #     # This warning means the parent RecordEpisodeStatistics didn't create an 'episode'
                    #     # entry, which is unexpected if `dones_in_this_step[i]` is True.
                    #     print(f"Warning: No 'episode' entry in deaggregated_infos[{i}] for a completed episode.")

                # After appending to queues, reset accumulators for the *next* episode for these specific environments
                self.episode_powers[i] = 0.0
                self.episode_powers_baseline[i] = 0.0

        # Important: `self.last_dones` (used for the *next* step's power accumulation logic)
        # needs to reflect the `dones` state of *this* current step.
        # The parent `RecordEpisodeStatistics` handles `self.prev_dones` correctly (it updates it BEFORE our step method runs)
        # The parent's `super().step` call effectively updates `self.prev_dones`
        # for the *next* step's operations (like resetting `episode_returns` for the parent).
        # Our `self.last_dones` isn't strictly needed if we just use `dones_in_this_step` for current processing
        # and rely on the parent for `prev_dones` for subsequent steps if needed.
        # For clarity and to avoid conflicts, let's remove the redundant `self.last_dones` update.
        # If `self.last_dones` is critical for *your own* logic in the *next* step,
        # then `self.last_dones = dones_in_this_step.copy()` would be the correct line.
        # But based on the previous structure, it seemed more of a leftover.

        # Return the de-aggregated list of info dictionaries
        return (
            observations,
            rewards,
            terminations,
            truncations,
            deaggregated_infos,  # Return the de-aggregated infos
        )
