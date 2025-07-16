# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Use specific WindGym imports based on your provided files
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager, MeasurementType
from WindGym.Agents.PyWakeAgent import NoisyPyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid

# Keep these imports as they were in your original cleanRL script
from WindGym.wrappers import AdversaryWrapper, CurriculumWrapper

# Import turbine type from PyWake
from py_wake.examples.data.hornsrev1 import V80

import yaml  # For loading temporary YAML


# # --- SECTION 1: Argument Parsing and Configuration ---
@dataclass
class Args:
    max_eps: int = 10
    """the maximum number inflow passes"""
    exp_name: str = field(
        default_factory=lambda: os.path.basename(__file__)[: -len(".py")]
    )
    """the name of this experiment"""
    save_interval: int = 100
    """the interval to save the model NOT USED ATM"""
    yaml_path: str = (
        "/work/users/manils/wesc/envs/env40.yaml"  # Assuming a default valid path
    )
    """the path to the yaml file"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLC_comp"  # Original project name
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: int = 1  # Integer instead of bool. Should (hopefully) work.
    """whether to save model into the `runs/{run_name}` folder"""
    turbtype: str = "V80"  # Changed to V80 as used in ParametricAdversarialEnv
    """the type of the wind turbine"""
    TI_type: str = "None"  # Retaining original
    """the type of the turbulence model"""
    dt_sim: int = 1
    """the time step of the simulation"""
    dt_env: int = 1
    """the time step of the environment"""

    # Adversarial Environment Specific Arguments (from original adversary script)
    n_passthrough: float = 10.0
    """Number of flow passthroughs for episode length in the base env"""
    max_bias_ws: float = 1.0
    """Max wind speed bias for per-turbine sensors (m/s)."""
    max_bias_wd: float = 5.0
    """Max wind direction bias for per-turbine sensors (deg)."""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1  # Keeping original num_envs from your cleanRL script as 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: int = 0  # I changed the bool to int. Hope its fine.
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# # SECTION 2: Parametric Adversarial Environment (Refined)


# ParametricAdversarialEnv definition (as provided in the original script, with info refinement)
class ParametricAdversarialEnv(gym.Env):
    """
    An adversarial environment where a dynamic agent learns to apply
    bias changes for individual turbine sensors at each step, creating
    smooth temporal evolution of the bias parameters.

    The info dictionary returned by step() is now meticulously constructed
    to avoid issues with vectorized environments, containing all necessary
    metrics for detailed logging.
    """

    def __init__(
        self,
        yaml_path: str,
        turbine: object,
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        constraints: dict,
        n_passthrough: float,
    ):
        super().__init__()
        self.base_env = WindFarmEnv(
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            yaml_path=yaml_path,
            Baseline_comp=True,
            reset_init=False,
            n_passthrough=n_passthrough,
            dt_env=10,
            dt_sim=1,
            yaw_step_sim=1,
        )
        self.measurement_manager = MeasurementManager(env=self.base_env)
        self.pywake_agent = NoisyPyWakeAgent(
            measurement_manager=self.measurement_manager,
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine,
            look_up=True,
            wd_min=0,
            wd_max=360,
        )
        self.constraints = constraints
        self.controlled_params = sorted(self.constraints.keys())
        self.num_controlled_params = len(self.controlled_params)

        # Action space now represents bias CHANGES (deltas), not absolute values
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_controlled_params,),  # Only bias changes, no std_dev
            dtype=np.float32,
        )
        self.observation_space = self.base_env.observation_space

        # Track current bias state
        self.current_bias_state = {}
        self.current_clean_obs = None

        # Episode tracking for internal use (mean/sum for current episode)
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_ws_values = []
        self.episode_wd_values = []

    def _scale_noise(
        self, unscaled_noise: float, min_val: float, max_val: float
    ) -> float:
        """Scales a noise delta value to the [-1, 1] normalized space."""
        span = max_val - min_val
        return unscaled_noise * 2.0 / span if span != 0 else 0.0

    def _scale_value(
        self, physical_val: float, min_val: float, max_val: float
    ) -> float:
        """Scales a physical value to the [-1, 1] normalized space."""
        span = max_val - min_val
        return 2 * (physical_val - min_val) / span - 1 if span != 0 else 0.0

    def _apply_bias_change(
        self, bias_change_action: np.ndarray
    ) -> Dict[str, float]:  # Returns unscaled physical bias
        """
        Applies small changes to the current bias state for smooth evolution.
        Updates self.current_bias_state with physical bias values (e.g., degrees, m/s).

        Args:
            bias_change_action: Array of bias change actions in [-1, 1].
                These are deltas in the adversary's action space.

        Returns:
            Dict[str, float]: Updated bias parameters (physical values).
        """
        updated_physical_biases = {}

        for i, param_name in enumerate(self.controlled_params):
            max_bias_for_param = self.constraints[param_name]["max_bias"]
            # Max allowed physical change per environment step (e.g., 10% of max bias for smoothness)
            max_physical_change_per_step = max_bias_for_param * 0.1

            # Convert action (which is in [-1, 1]) to a physical bias change amount
            bias_change_physical_delta = (
                bias_change_action[i] * max_physical_change_per_step
            )

            # Initialize current bias state if not present
            if param_name not in self.current_bias_state:
                self.current_bias_state[param_name] = 0.0  # Physical units

            new_bias_value_physical = (
                self.current_bias_state[param_name] + bias_change_physical_delta
            )

            # Clamp the accumulated bias to the defined [-max_bias, max_bias] range
            new_bias_value_physical = np.clip(
                new_bias_value_physical, -max_bias_for_param, max_bias_for_param
            )

            self.current_bias_state[param_name] = new_bias_value_physical
            updated_physical_biases[param_name] = new_bias_value_physical

        return updated_physical_biases

    def _recalculate_farm_obs(self, obs_vector: np.ndarray) -> np.ndarray:
        """Recalculates farm-level ws/wd based on the average of noisy turbine sensors."""
        modified_obs = obs_vector.copy()
        noisy_turbine_ws, noisy_turbine_wd = [], []
        farm_ws_spec, farm_wd_spec = None, None

        for spec in self.measurement_manager.specs:
            if "current" not in spec.name:
                continue
            if spec.turbine_id is not None:
                val_scaled = modified_obs[spec.index_range[0] : spec.index_range[1]]
                val_physical = self.pywake_agent._unscale(
                    val_scaled, spec.min_val, spec.max_val
                )
                if spec.measurement_type == MeasurementType.WIND_SPEED:
                    noisy_turbine_ws.append(np.mean(val_physical))
                elif spec.measurement_type == MeasurementType.WIND_DIRECTION:
                    noisy_turbine_wd.append(np.mean(val_physical))
            elif spec.name == "farm/ws_current":
                farm_ws_spec = spec
            elif spec.name == "farm/wd_current":
                farm_wd_spec = spec

        if farm_ws_spec and noisy_turbine_ws:
            new_farm_ws_phys = np.mean(noisy_turbine_ws)
            new_farm_ws_scaled = self._scale_value(
                new_farm_ws_phys, farm_ws_spec.min_val, farm_ws_spec.max_val
            )
            modified_obs[farm_ws_spec.index_range[0] : farm_ws_spec.index_range[1]] = (
                new_farm_ws_scaled
            )

        if farm_wd_spec and noisy_turbine_wd:
            new_farm_wd_phys = np.mean(noisy_turbine_wd)
            new_farm_wd_scaled = self._scale_value(
                new_farm_wd_phys, farm_wd_spec.min_val, farm_wd_spec.max_val
            )
            modified_obs[farm_wd_spec.index_range[0] : farm_wd_spec.index_range[1]] = (
                new_farm_wd_scaled
            )

        return modified_obs

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # base_env.reset returns (obs, info). We store the clean obs and info internally.
        self.current_clean_obs, self.info = self.base_env.reset(seed=seed)

        # Reset bias state and episode tracking for the new episode
        self.current_bias_state = {param: 10.0 for param in self.controlled_params}
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_ws_values = []
        self.episode_wd_values = []

        # Return minimal info on reset to the vector environment to prevent serialization issues.
        # All detailed info will be pulled by the logger/callback.
        return self.current_clean_obs, self.info

    def step(
        self, adversary_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_clean_obs is None:
            # If step is called before reset (shouldn't happen in normal Gym loop), reset first.
            self.reset()

        # Apply bias changes for smooth temporal evolution
        step_params = self._apply_bias_change(adversary_action)
        perturbation = np.zeros_like(self.current_clean_obs, dtype=np.float32)

        # This will be the info dictionary returned by *this* ParametricAdversarialEnv.step() method.
        # It aggregates all data for comprehensive logging.
        info_to_return = {}

        # 1. Apply adversary's bias to the clean observation
        for spec in self.measurement_manager.specs:
            if spec.name in self.controlled_params:
                bias_physical = step_params[spec.name]
                # Log adversary's bias for this step
                info_to_return[f"adv_params/{spec.name}_bias_phys"] = bias_physical

                scaled_bias = self._scale_noise(
                    bias_physical, spec.min_val, spec.max_val
                )
                perturbation[spec.index_range[0] : spec.index_range[1]] += scaled_bias

        noisy_obs = self.current_clean_obs + perturbation

        # Ensure wind direction measurements wrap correctly
        for spec in self.measurement_manager.specs:
            if spec.measurement_type == MeasurementType.WIND_DIRECTION:
                val_scaled = noisy_obs[spec.index_range[0] : spec.index_range[1]]
                val_physical = self.pywake_agent._unscale(
                    val_scaled, spec.min_val, spec.max_val
                )
                val_physical_wrapped = np.array(val_physical) % 360
                val_scaled_wrapped = self._scale_value(
                    val_physical_wrapped, spec.min_val, spec.max_val
                )
                noisy_obs[spec.index_range[0] : spec.index_range[1]] = (
                    val_scaled_wrapped
                )

        noisy_obs = np.clip(noisy_obs, -1.0, 1.0).astype(np.float32)
        noisy_obs = self._recalculate_farm_obs(
            noisy_obs
        )  # Recalculate farm-level from noisy turbine data

        # 2. Protagonist (PyWakeAgent) acts based on noisy observation
        est_ws, est_wd = self.pywake_agent._estimate_wind_from_obs(noisy_obs)
        self.pywake_agent.update_wind(
            wind_speed=est_ws, wind_direction=est_wd, TI=self.pywake_agent.TI
        )
        pywake_action, _ = self.pywake_agent.predict(noisy_obs)

        # 3. Step the base WindFarmEnv with protagonist's action
        next_clean_obs, protag_reward, terminated, truncated, base_env_info = (
            self.base_env.step(pywake_action)
        )
        self.current_clean_obs = next_clean_obs
        adversary_reward = (
            -protag_reward
        )  # Adversary's reward is negative of protagonist's reward

        # 4. Update internal episode tracking for ParametricAdversarialEnv
        self.episode_step += 1
        self.episode_rewards.append(adversary_reward)

        global_ws = base_env_info.get("Wind speed Global", 0.0)
        global_wd = base_env_info.get("Wind direction Global", 0.0)
        self.episode_ws_values.append(global_ws)
        self.episode_wd_values.append(global_wd)

        # 5. Populate the info dictionary for logging
        info_to_return["perturbation/total_L2_norm"] = np.linalg.norm(perturbation)
        info_to_return["reward/episode_step"] = self.episode_step
        info_to_return["reward/stepwise"] = adversary_reward
        info_to_return["reward/episode_mean"] = np.mean(self.episode_rewards)
        info_to_return["reward/episode_sum"] = np.sum(self.episode_rewards)
        info_to_return["global_wind/ws"] = global_ws
        info_to_return["global_wind/wd"] = global_wd
        info_to_return["global_wind/ws_episode_mean"] = np.mean(self.episode_ws_values)
        info_to_return["global_wind/wd_episode_mean"] = np.mean(self.episode_wd_values)

        for param_name in self.controlled_params:
            if param_name in self.current_bias_state:
                info_to_return[f"bias_state/{param_name}"] = self.current_bias_state[
                    param_name
                ]

        # Transfer ALL relevant keys from `base_env_info` into `info_to_return`.
        # This includes powers, yaw angles, true winds, etc. These are scalar or arrays.
        # This is CRUCIAL for AdversaryWrapper to find 'Power agent' and 'Power baseline'.
        for k, v in base_env_info.items():
            info_to_return[k] = v

        # Add sensed values from the perturbed observation
        for spec in self.measurement_manager.specs:
            if "current" not in spec.name:
                continue
            val_scaled = noisy_obs[spec.index_range[0] : spec.index_range[1]]
            val_physical = self.pywake_agent._unscale(
                val_scaled, spec.min_val, spec.max_val
            )
            if spec.turbine_id is not None:
                if spec.measurement_type == MeasurementType.WIND_SPEED:
                    info_to_return[f"ws_sensed/turbine_{spec.turbine_id}"] = np.mean(
                        val_physical
                    )
                elif spec.measurement_type == MeasurementType.WIND_DIRECTION:
                    info_to_return[f"wd_sensed/turbine_{spec.turbine_id}"] = np.mean(
                        val_physical
                    )
            else:  # Farm-level signals
                if spec.measurement_type == MeasurementType.WIND_SPEED:
                    info_to_return["ws_sensed/farm"] = np.mean(val_physical)
                elif spec.measurement_type == MeasurementType.WIND_DIRECTION:
                    info_to_return["wd_sensed/farm"] = np.mean(val_physical)

        return next_clean_obs, adversary_reward, terminated, truncated, info_to_return

    def close(self):
        self.base_env.close()


# # SECTION 3: Custom Logger for TensorBoard/WandB (Robust)
class VectorEnvLogger:
    """
    A custom logger to handle detailed info dictionaries from vectorized environments.
    It's designed to log all available numerical data to TensorBoard/WandB,
    with proper tagging for individual environments and turbines.
    Assumes `AdversaryWrapper` returns a list[Dict[str, Any]] for `infos`.
    """

    def __init__(self, writer: SummaryWriter, num_envs: int, n_turbines: int = -1):
        self.writer = writer
        self.num_envs = num_envs
        # Pass the number of turbines to the logger for robust identification of per-turbine arrays.
        # This `n_turbines` should be obtained from your environment's configuration.
        # e.g., in main loop: `n_turb = config_from_yaml["farm"]["nx"] * config_from_yaml["farm"]["ny"]`
        # and then `vector_logger = VectorEnvLogger(writer, args.num_envs, n_turbines=n_turb)`
        self.n_turbines = n_turbines

    def log(self, global_step: int, infos: list[Dict[str, Any]]):
        """
        Logs information for all parallel environments at the current global step.

        Args:
            global_step (int): The current global timestep.
            infos (List[Dict[str, Any]]): A list of info dictionaries, one for each environment.
                                         This is the format returned by `AdversaryWrapper.step()`
                                         after de-aggregation.
        """
        for i, info_entry in enumerate(infos):
            # Each `info_entry` is a dictionary corresponding to a single environment.
            if not info_entry:  # Skip if the info dictionary for this env is empty
                continue

            # --- Internal Helper Function for Recursive Logging ---
            # This function handles the actual logging for a given key and value,
            # prepending the environment ID and a base prefix.
            def _log_value(base_tag: str, value: Any):
                if isinstance(value, (int, float, np.number)):
                    # Direct scalar or 0D NumPy array
                    self.writer.add_scalar(
                        base_tag,
                        value.item() if isinstance(value, np.ndarray) else value,
                        global_step,
                    )
                elif isinstance(value, np.ndarray) and value.ndim == 1:
                    # Handle 1D arrays: check if it's a per-turbine array
                    # We compare `value.shape[0]` against `self.n_turbines` for robustness.
                    # If `self.n_turbines` is not set correctly or is -1, this check needs care.
                    if self.n_turbines != -1 and value.shape[0] == self.n_turbines:
                        # Assume it's a per-turbine array if its length matches `n_turbines`
                        for turb_idx, val in enumerate(value):
                            if isinstance(val, (int, float, np.number)):
                                self.writer.add_scalar(
                                    f"{base_tag}/turbine_{turb_idx}", val, global_step
                                )
                            # else: print(f"Warning: Non-numeric value in per-turbine array: {base_tag}/turbine_{turb_idx} = {val}")
                    else:
                        # Other 1D arrays (e.g., history windows, or arrays not matching n_turbines)
                        # Log its mean, and if it has only one element, log it directly as a scalar.
                        if value.size == 1 and isinstance(
                            value.item(), (int, float, np.number)
                        ):
                            self.writer.add_scalar(base_tag, value.item(), global_step)
                        elif value.size > 1 and isinstance(
                            value[0], (int, float, np.number)
                        ):
                            self.writer.add_scalar(
                                f"{base_tag}_mean", np.mean(value), global_step
                            )
                            # You could add min/max/std if desired:
                            # self.writer.add_scalar(f"{base_tag}_std", np.std(value), global_step)
                            # self.writer.add_scalar(f"{base_tag}_min", np.min(value), global_step)
                            # self.writer.add_scalar(f"{base_tag}_max", np.max(value), global_step)
                        # else: print(f"Warning: Skipping complex 1D array: {base_tag} = {value}")
                elif isinstance(value, dict):
                    # Recursively log nested dictionaries
                    for sub_k, sub_v in value.items():
                        _log_value(f"{base_tag}/{sub_k}", sub_v)
                # elif isinstance(value, (list, tuple)): # Handles lists/tuples that are NOT np.ndarray
                #    # Can convert to numpy array and re-process, or just skip.
                #    if len(value) > 0 and isinstance(value[0], (int, float, np.number)):
                #        _log_value(base_tag, np.array(value)) # Convert to numpy for consistent handling
                #    # else: print(f"Warning: Skipping non-numeric list/tuple: {base_tag} = {value}")
                # else:
                #     # This catches strings, complex objects, etc. that cannot be logged as scalars.
                #     # print(f"Warning: Skipping non-numeric/unhandled type for logging: {base_tag} = {type(value)}")
                #     pass # Do nothing for unhandled types like strings or arbitrary objects

            # --- Main Logging Loop for info_entry ---
            # Log episode summary (from AdversaryWrapper) if present
            # `AdversaryWrapper` places 'r', 'l', 'mean_power_queue', 'mean_power_queue_baseline' here.
            if "episode" in info_entry and info_entry["episode"] is not None:
                episode_stats = info_entry["episode"]
                _log_value(f"charts/episodic_return_env{i}", episode_stats["r"])
                _log_value(f"charts/episodic_length_env{i}", episode_stats["l"])

                # Check if power metrics exist before logging them
                if "mean_power_agent" in episode_stats:
                    _log_value(
                        f"charts/episodic_power_agent_env{i}",
                        episode_stats["mean_power_agent"],
                    )
                if "mean_power_baseline" in episode_stats:
                    _log_value(
                        f"charts/episodic_power_baseline_env{i}",
                        episode_stats["mean_power_baseline"],
                    )

            # Log all other detailed step-wise metrics
            for k, v in info_entry.items():
                # Filter out keys that are internal or handled separately
                if k in [
                    "final_info",
                    "clean_obs",
                    "episode",
                    "time_array",  # 'time_array' might be verbose per step
                    "windspeeds",
                    "winddirs",
                    "yaws",
                    "powers",  # Raw env data passed to AdversaryWrapper
                    "baseline_powers",
                    "yaws_baseline",
                    "windspeeds_baseline",  # Raw env data
                ]:  # Add any other keys you explicitly want to skip logging directly
                    continue

                # Start recursive logging for the current key-value pair
                _log_value(f"env_{i}/{k}", v)


# # SECTION 4: Environment Creation Factory ---
def make_env(env_id: int, args: Args, yaml_filepath: str, adversary_constraints: dict):
    """
    Creates a single instance of the ParametricAdversarialEnv.
    AdversaryWrapper will be applied to the *vectorized* environment outside this function.
    """

    def thunk():
        # Load config directly here to avoid issues with function closures and mutable defaults.
        # This config is hardcoded as per your original script's YAML_CONFIG_STR.
        config = {
            "yaw_init": "Zeros",
            "noise": "None",
            "BaseController": "PyWake",
            "ActionMethod": "yaw",
            "farm": {
                "yaw_min": -500,
                "yaw_max": 500,
                "nx": 2,
                "ny": 1,
                "xDist": 7,
                "yDist": 7,
            },
            "wind": {
                "ws_min": 6,
                "ws_max": 10,
                "TI_min": 0.02,
                "TI_max": 0.07,
                "wd_min": 250,
                "wd_max": 290,
            },
            "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
            "power_def": {
                "Power_reward": "Baseline",
                "Power_avg": 10,
                "Power_scaling": 1.0,
            },
            "mes_level": {
                "turb_ws": True,
                "turb_wd": True,
                "farm_ws": True,
                "farm_wd": True,
                "turb_TI": False,
                "turb_power": False,
                "farm_TI": False,
                "farm_power": False,
            },
            "ws_mes": {
                "ws_current": True,
                "ws_rolling_mean": True,
                "ws_history_N": 3,
                "ws_history_length": 20,
                "ws_window_length": 5,
            },
            "wd_mes": {
                "wd_current": True,
                "wd_rolling_mean": True,
                "wd_history_N": 3,
                "wd_history_length": 20,
                "wd_window_length": 5,
            },
            "yaw_mes": {
                "yaw_current": False,
                "yaw_rolling_mean": False,
                "yaw_history_N": 1,
                "yaw_history_length": 1,
                "yaw_window_length": 1,
            },
            "power_mes": {
                "power_current": False,
                "power_rolling_mean": False,
                "power_history_N": 1,
                "power_history_length": 1,
                "power_window_length": 1,
            },
        }

        n_turb = config["farm"]["nx"] * config["farm"]["ny"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=config["farm"]["nx"],
            ny=config["farm"]["ny"],
            xDist=config["farm"]["xDist"],
            yDist=config["farm"]["yDist"],
        )

        env = ParametricAdversarialEnv(
            yaml_path=yaml_filepath,
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            constraints=adversary_constraints,
            n_passthrough=args.n_passthrough,
        )

        return env

    return thunk


# # SECTION 5: Agent Network Definition ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes neural network layer weights and biases."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Defines the neural network for the PPO agent (actor-critic).
    """

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), 64
                )  # Original hidden layer size from your cleanRL script
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), 64
                )  # Original hidden layer size from your cleanRL script
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# # SECTION 6: Main Training Loop ---
if __name__ == "__main__":
    args = tyro.cli(Args)

    # Generate static YAML config for the base environment
    YAML_CONFIG_STR = """
yaw_init: "Zeros"
noise: "None"
BaseController: "PyWake"
ActionMethod: "yaw"
farm: {yaw_min: -30, yaw_max: 30, nx: 2, ny: 1, xDist: 7, yDist: 7}
wind: {ws_min: 6, ws_max: 10, TI_min: 0.02, TI_max: 0.07, wd_min: 250, wd_max: 290}
act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
power_def: {Power_reward: "Baseline", Power_avg: 10, Power_scaling: 1.0}
mes_level: {turb_ws: True, turb_wd: True, farm_ws: True, farm_wd: True, turb_TI: False, turb_power: False, farm_TI: False, farm_power: False}
ws_mes: {ws_current: True, ws_rolling_mean: True, ws_history_N: 3, ws_history_length: 20, ws_window_length: 5}
wd_mes: {wd_current: True, wd_rolling_mean: True, wd_history_N: 3, wd_history_length: 20, wd_window_length: 5}
yaw_mes: {yaw_current: False, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: False, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""
    # Create a temporary YAML file for the base environment
    import yaml
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_yaml_file:
        tmp_yaml_file.write(YAML_CONFIG_STR)
        yaml_filepath = tmp_yaml_file.name

    # Define adversary constraints based on args
    config_from_yaml = yaml.safe_load(YAML_CONFIG_STR)
    n_turb = config_from_yaml["farm"]["nx"] * config_from_yaml["farm"]["ny"]
    adversary_constraints = {}
    for i in range(n_turb):
        adversary_constraints[f"turb_{i}/ws_current"] = {
            "max_bias": args.max_bias_ws,
            "max_std_dev": 0.0,
        }
        adversary_constraints[f"turb_{i}/wd_current"] = {
            "max_bias": args.max_bias_wd,
            "max_std_dev": 0.0,
        }

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,  # This enables gymnasium env monitoring
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Define wind_turbine as V80 as per ParametricAdversarialEnv init
    wind_turbine = V80

    # env setup
    # Make multiple environments using gym.vector.AsyncVectorEnv
    # The make_env function now returns a thunk that creates a single env (ParametricAdversarialEnv)
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(i, args, yaml_filepath, adversary_constraints)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # Back to SAME_STEP from original cleanRL script
    )
    # NOW apply AdversaryWrapper to the *vectorized* environment, as it expects a VectorEnv
    envs = AdversaryWrapper(envs)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize custom logger
    vector_logger = VectorEnvLogger(writer, args.num_envs)

    # ALGO Logic: Storage setup
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  # Initial reset
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # infos will be a list of dictionaries, one per sub-environment.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            # Log detailed metrics using the custom logger
            # vector_logger.log now receives the `infos` list directly
            vector_logger.log(global_step, infos)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    if args.save_model:
        # Save the model
        model_path = f"V80_runs/{run_name}/{args.exp_name}_{global_step}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
    # Clean up the temporary YAML file
    import os

    os.remove(yaml_filepath)
