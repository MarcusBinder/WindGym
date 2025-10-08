# two_turbs/run.py

import os
import yaml
import tempfile
from dataclasses import dataclass
from noise_definitions import create_procedural_noise_model
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tyro
from stable_baselines3 import PPO
from tqdm import tqdm

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Agents import PyWakeAgent, NoisyPyWakeAgent
from WindGym.Measurement_Manager import (
    MeasurementManager,
    NoisyWindFarmEnv,
    NoiseModel,
    MeasurementType,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    HybridNoiseModel,
)
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

DETERMINISTIC = True


# --- Argument Parsing ---
@dataclass
class Args:
    agent_type: str
    """The type of agent to evaluate ('ppo' or 'pywake')."""
    scenario: str
    """The evaluation scenario ('clean', 'procedural', or 'adversarial')."""
    output_path: str
    """Path to save the output CSV file."""
    protagonist_path: str = ""
    """Path to the protagonist agent model (required for agent_type='ppo')."""
    antagonist_path: str = ""
    """Path to the antagonist agent model (required for scenario='adversarial')."""
    sim_time: int = 1000
    """Total simulation time in seconds."""
    seed: int = 42
    """Random seed for the environment."""
    config_path: str = "two_turbs/juqu.yaml"
    """Path to the environment configuration YAML file."""
    antagonist_arch: str = "64,64"
    """Network architecture for the antagonist (if loading a .pt file)."""


# --- Agent Network for loading .pt models ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, obs_space, act_space, net_arch=[64, 64]):
        super().__init__()
        obs_shape_prod = np.array(obs_space.shape).prod()
        act_shape_prod = np.prod(act_space.shape)
        actor_layers, critic_layers = [], []
        last_layer_size = obs_shape_prod
        for layer_size in net_arch:
            actor_layers.extend(
                [layer_init(nn.Linear(last_layer_size, layer_size)), nn.Tanh()]
            )
            critic_layers.extend(
                [layer_init(nn.Linear(last_layer_size, layer_size)), nn.Tanh()]
            )
            last_layer_size = layer_size
        self.critic = nn.Sequential(
            *critic_layers, layer_init(nn.Linear(last_layer_size, 1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            *actor_layers,
            layer_init(nn.Linear(last_layer_size, act_shape_prod), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape_prod))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        from torch.distributions.normal import Normal

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


class PPOAgentWrapper:
    def __init__(self, agent_model: PPOAgent, device: torch.device):
        self.model, self.device = agent_model, device
        self.model.eval()

    @torch.no_grad()
    def predict(self, obs, deterministic=DETERMINISTIC):
        obs_tensor = torch.Tensor(obs).to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        action = (
            self.model.actor_mean(obs_tensor)
            if deterministic
            else self.model.get_action_and_value(obs_tensor)[0]
        )
        return action.squeeze(0).cpu().numpy(), None


def load_agent(
    path: str,
    dummy_env: gym.Env,
    device: torch.device,
    net_arch_str: str = "64,64",
    action_space=None,
):
    """Loads an agent model from either a .zip or .pt file."""
    if path.endswith(".zip"):
        return PPO.load(path, device=device)
    elif path.endswith(".pt"):
        act_space = action_space if action_space is not None else dummy_env.action_space
        net_arch = [int(x) for x in net_arch_str.split(",")]
        model = PPOAgent(dummy_env.observation_space, act_space, net_arch=net_arch).to(
            device
        )
        model.load_state_dict(torch.load(path, map_location=device))
        return PPOAgentWrapper(model, device)
    raise ValueError(f"Unknown agent file type for path: {path}")


class AdversarialNoiseModel(NoiseModel):
    def __init__(self, antagonist_agent, constraints, device):
        super().__init__()
        self.antagonist = antagonist_agent
        self.constraints = constraints
        self.device = device
        self.current_bias_state = {}

    def reset_noise(self, specs: list, rng: np.random.Generator):
        self.current_bias_state = {}

    def apply_noise(self, clean_observations, specs, rng):
        with torch.no_grad():
            antagonist_action, _ = self.antagonist.predict(
                clean_observations, deterministic=DETERMINISTIC
            )

        noisy_obs = clean_observations.copy()
        action_idx = 0

        # Determine the bias to apply based on the antagonist's action
        bias_to_apply = {}
        for spec in specs:
            is_ws = "ws" in spec.name and "turb" in spec.name
            is_wd = "wd" in spec.name and "turb" in spec.name

            # The antagonist provides one action per turbine, for WS and WD bias.
            # We map the action to the bias value here.
            if (is_ws or is_wd) and spec.name.endswith("_current"):
                max_bias = (
                    self.constraints["max_bias_ws"]
                    if is_ws
                    else self.constraints["max_bias_wd"]
                )
                max_physical_change_per_step = max_bias * 0.1
                bias_change_physical_delta = (
                    antagonist_action[action_idx] * max_physical_change_per_step
                )

                current_physical_bias = self.current_bias_state.get(spec.name, 0.0)
                new_physical_bias = current_physical_bias + bias_change_physical_delta
                new_physical_bias = np.clip(new_physical_bias, -max_bias, max_bias)
                self.current_bias_state[spec.name] = new_physical_bias
                bias_to_apply[spec.name] = new_physical_bias

                action_idx += 1

        # Apply the determined bias to ALL relevant specs, including rolling means.
        for spec in specs:
            if "wd" in spec.name and "turb" in spec.name:
                # Use the bias calculated for the 'current' wind direction
                source_spec_name = f"turb_{spec.turbine_id}/wd_current"
                if source_spec_name in bias_to_apply:
                    physical_bias = bias_to_apply[source_spec_name]
                    span = spec.max_val - spec.min_val
                    if span > 0:
                        scaled_bias = (physical_bias * 2.0) / span
                        noisy_obs[spec.index_range[0] : spec.index_range[1]] += (
                            scaled_bias
                        )

        return np.clip(noisy_obs, -1.0, 1.0)

    def get_info(self):
        return {
            "noise_type": "adversarial (stateful)",
            "applied_bias (physical_units)": self.current_bias_state.copy(),
        }


def main(args: Args):
    print(
        f"--- Generating Time Series for Agent '{args.agent_type.upper()}' in Scenario: {args.scenario.upper()} ---"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Environment Setup ---
    with open(args.config_path, "r") as f:
        config_data = yaml.safe_load(f)
    with open(args.config_path, "r") as f:
        YAML_CONFIG_STR = f.read()

    farm_params = config_data["farm"]
    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine=turbine_obj,
        nx=farm_params["nx"],
        ny=farm_params["ny"],
        xDist=farm_params.get("xDist", 7),
        yDist=farm_params.get("yDist", 7),
    )

    base_env_kwargs = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "turbine": turbine_obj,
        "config": YAML_CONFIG_STR,
        "reset_init": True,
        "Baseline_comp": True,
        "n_passthrough": 10,
        "burn_in_passthroughs": 2.0,
        "fill_window": True,
        "dt_sim": 1,
        "dt_env": 10,
        "turbtype": "None",
    }

    # Create a dummy environment for agent loading
    dummy_env = WindFarmEnv(**base_env_kwargs)

    # --- Agent Loading ---
    agent = None
    if args.agent_type.lower() == "ppo":
        if not args.protagonist_path:
            raise ValueError("--protagonist-path is required for agent_type 'ppo'")
        agent = load_agent(args.protagonist_path, dummy_env, device)

    # --- Scenario-specific Environment and Agent Configuration ---
    # Create the clean base environment once. This will be either used directly or wrapped.
    base_env = WindFarmEnv(**base_env_kwargs, seed=args.seed)
    mm = MeasurementManager(base_env, seed=args.seed)

    if args.scenario == "clean":
        # In the clean case, we do not need a noise wrapper.
        # The agent should be the 'perfect' PyWakeAgent, not the noisy one.
        env = base_env
        if args.agent_type.lower() == "pywake":
            print(
                "Instantiating standard PyWakeAgent (Oracle) for a clean environment."
            )
            agent = PyWakeAgent(
                x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj, env=base_env
            )

    elif args.scenario == "procedural":
        mm.set_noise_model(create_procedural_noise_model())
        # The environment is now a wrapper that applies the procedural noise
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs, seed=args.seed)
        if args.agent_type.lower() == "pywake":
            print(
                "Instantiating NoisyPyWakeAgent (Sensing) for a procedural environment."
            )
            agent = NoisyPyWakeAgent(
                measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj
            )

    elif args.scenario == "adversarial":
        if not args.antagonist_path:
            raise ValueError("--antagonist-path is required for adversarial scenario.")
        antagonist_act_space = gym.spaces.Box(
            low=-1, high=1, shape=(dummy_env.n_turb * 2,), dtype=np.float32
        )
        antagonist_model = load_agent(
            path=args.antagonist_path,
            dummy_env=dummy_env,
            device=device,
            net_arch_str=args.antagonist_arch,
            action_space=antagonist_act_space,
        )
        constraints = {"max_bias_ws": 2.0, "max_bias_wd": 10.0}
        mm.set_noise_model(AdversarialNoiseModel(antagonist_model, constraints, device))
        # The environment is now a wrapper that applies the adversarial noise
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs, seed=args.seed)
        if args.agent_type.lower() == "pywake":
            print(
                "Instantiating NoisyPyWakeAgent (Sensing) for an adversarial environment."
            )
            agent = NoisyPyWakeAgent(
                measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj
            )

    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    if agent is None:
        raise ValueError(
            f"Agent could not be created for agent_type: {args.agent_type}"
        )

    dummy_env.close()

    # --- Simulation Loop ---
    log = []
    # NOTE: The agent.update_wind call needs to be here for the PyWakeAgent in the clean scenario
    # to have the correct initial conditions for its first action.
    if isinstance(agent, PyWakeAgent) and not isinstance(agent, NoisyPyWakeAgent):
        # This is for the OraclePyWakeAgent in the 'clean' scenario
        agent.update_wind(env.ws, env.wd, env.ti)

    obs, info = env.reset()
    terminated = truncated = False

    with tqdm(total=args.sim_time, desc="Simulating") as pbar:
        last_time = 0
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            sensed_wd_for_log = info.get(
                "obs_sensed/farm/wd_current",
                np.mean(info["Wind direction at turbines"]),
            )
            if "obs_sensed/turb_0/wd_hist_1" in info:
                sensed_wd_for_log = np.mean(
                    [
                        info["obs_sensed/turb_0/wd_hist_1"],
                        info["obs_sensed/turb_1/wd_hist_1"],
                    ]
                )
            # if args.scenario == 'adversarial' and hasattr(agent, 'wdir') and agent.wdir is not None:
            #    sensed_wd_for_log = np.mean(agent.wdir)

            for i in range(len(info["time_array"])):
                log_entry = {
                    "time": info["time_array"][i],
                    "power_agent": info["powers"][i].sum(),
                    "power_baseline": info["baseline_powers"][i].sum(),
                    "true_wd": info["Wind direction at turbines"].mean(),
                    "sensed_wd": sensed_wd_for_log,  # info.get('obs_sensed/farm/wd_current', np.mean(info['Wind direction at turbines']))
                }
                for t_idx in range(env.unwrapped.n_turb):
                    log_entry[f"yaw_t{t_idx}"] = info["yaws"][i][t_idx]

                log.append(log_entry)
                # if last_time > 900: hey

            current_time = info["time_array"][-1]
            pbar.update(current_time - last_time)
            last_time = current_time

            if current_time >= args.sim_time:
                break

    env.close()

    log_df = pd.DataFrame(log)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    log_df.to_csv(args.output_path, index=False)
    print(f"\n✅ Time series data saved to '{args.output_path}'")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
