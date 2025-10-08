import os
import tyro
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass, field
from typing import Optional, List, Type
import yaml
import wandb
import time

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import (
    MeasurementManager,
    NoisyWindFarmEnv,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    HybridNoiseModel,
    MeasurementType,
)
from noise_definitions import create_procedural_noise_model
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80


# --- Command-line Arguments ---
@dataclass
class Args:
    project_name: str = "WindGym_Protagonist_Training"
    run_name_prefix: str = "PPO_Protagonist"
    seed: int = 42
    total_timesteps: int = 500000
    n_envs: int = 4

    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"

    noise_type: str = "procedural"  # 'none', 'procedural', or 'adversarial'

    # Noise model parameters (for 'procedural' or 'adversarial')
    max_bias_ws: float = 2.0
    max_bias_wd: float = 20.0

    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: str = "128,128"

    models_dir: str = "models/protagonist_training"

    # PPO-specific arguments
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01


def make_env_factory(args: Args) -> callable:
    """
    Creates a factory function for a single environment instance.
    This factory encapsulates all necessary setup, including the noise model.
    """
    with open(args.yaml_config_path, "r") as f:
        base_env_config = yaml.safe_load(f)

    # Use a fixed layout for reproducibility
    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine=turbine_obj,
        nx=base_env_config["farm"]["nx"],
        ny=base_env_config["farm"]["ny"],
        xDist=base_env_config["farm"]["xDist"],
        yDist=base_env_config["farm"]["yDist"],
    )

    # Base keyword arguments for the WindFarmEnv constructor
    base_env_kwargs = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "turbine": turbine_obj,
        "config": args.yaml_config_path,
        "turbtype": base_env_config["turbtype"],
        "dt_sim": 1,
        "dt_env": 10,
        "reset_init": True,
        "Baseline_comp": True,
    }

    # Setup the noise model based on the scenario
    # Note: For 'adversarial', you'd need to load and pass the antagonist model.
    if args.noise_type == "none":
        noise_model = None
    elif args.noise_type == "procedural":
        noise_model = create_procedural_noise_model()
    else:
        raise ValueError(f"Unsupported noise_type: {args.noise_type}")

    def env_thunk():
        # Each worker process gets its own seeded environment and MeasurementManager
        seed_offset = np.random.randint(0, 10000)
        env_seed = args.seed + seed_offset

        env_instance = WindFarmEnv(**base_env_kwargs, seed=env_seed)

        # The MeasurementManager must be instantiated per environment instance
        mm = MeasurementManager(env_instance, seed=env_seed)
        if noise_model is not None:
            mm.set_noise_model(noise_model)
            wrapped_env = NoisyWindFarmEnv(
                base_env_class=WindFarmEnv,
                measurement_manager=mm,
                **env_instance.kwargs,
            )
        else:
            wrapped_env = env_instance

        # Monitor is a crucial wrapper from SB3 for logging episode stats
        return Monitor(wrapped_env)

    return env_thunk


def main(args: Args):
    # Ensure the YAML file path exists before starting
    if not os.path.exists(args.yaml_config_path):
        raise FileNotFoundError(
            f"YAML configuration file not found at: {args.yaml_config_path}"
        )

    # Initialize wandb run
    run_name = f"{args.run_name_prefix}_{args.noise_type}_{int(time.time())}"
    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
        save_code=True,
    )

    # Setup directories
    models_save_path = os.path.join(args.models_dir, wandb.run.id)
    os.makedirs(models_save_path, exist_ok=True)

    # Create the vectorized environment
    env_factory = make_env_factory(args)
    vec_env = make_vec_env(env_factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    # Setup policy and model
    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=net_arch_list)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/{wandb.run.id}",
        verbose=1,
    )

    callbacks = CallbackList(
        [
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=os.path.join(models_save_path, "checkpoints"),
                verbose=2,
                model_save_freq=10,
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )
        model.save(os.path.join(models_save_path, "final_model"))
        print(f"\n✅ Training complete. Final model saved to '{models_save_path}'")
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
