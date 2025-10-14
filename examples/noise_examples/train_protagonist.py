# filename: examples/noise_examples/train_protagonist.py (Updated)

import os
import tyro
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass
import yaml
import wandb
import time

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager, NoisyWindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from noise_definitions import (
    create_procedural_noise_model,
    AdversarialNoiseModel,
)  # <-- NEW: Import AdversarialNoiseModel


# --- Command-line Arguments ---
@dataclass
class Args:
    """Script arguments"""

    project_name: str = "WindGym_Protagonist_Training"
    run_name_prefix: str = "PPO_Protagonist"
    seed: int = 42
    total_timesteps: int = 500000
    n_envs: int = 4

    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"

    noise_type: str = "procedural"  # 'none', 'procedural', or 'adversarial' # <-- NEW: Added 'adversarial' option

    antagonist_path: str = ""  # <-- NEW: Path to the antagonist model, only used if noise_type is 'adversarial'

    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: str = "128,128"

    models_dir: str = "models/protagonist_training"

    # PPO-specific arguments
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01


def make_env_factory(args: Args, rank: int) -> callable:
    """
    Creates a factory function for a single environment instance.
    This factory encapsulates all necessary setup, including the noise model.
    """

    def _init() -> gym.Env:
        """Initializes and wraps the environment."""
        with open(args.yaml_config_path, "r") as f:
            config = yaml.safe_load(f)

        turbine_obj = V80()
        x_pos, y_pos = generate_square_grid(
            turbine=turbine_obj,
            nx=config["farm"]["nx"],
            ny=config["farm"]["ny"],
            xDist=config["farm"]["xDist"],
            yDist=config["farm"]["yDist"],
        )

        base_env_kwargs = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "turbine": turbine_obj,
            "config": args.yaml_config_path,
            "turbtype": config.get("turbtype", "None"),
            "dt_sim": 1,
            "dt_env": 10,
            "reset_init": True,
            "Baseline_comp": True,
        }

        env_seed = args.seed + rank
        env_instance = WindFarmEnv(**base_env_kwargs, seed=env_seed)

        if args.noise_type != "none":
            mm = MeasurementManager(env_instance, seed=env_seed)
            noise_model = None

            if args.noise_type == "procedural":
                print(f"Rank {rank}: Using procedural noise.")
                noise_model = create_procedural_noise_model()

            elif args.noise_type == "adversarial":
                print(
                    f"Rank {rank}: Using adversarial noise from '{args.antagonist_path}'."
                )
                if not args.antagonist_path:
                    raise ValueError(
                        "--antagonist-path is required for adversarial training."
                    )

                # Load the pre-trained antagonist agent
                antagonist_agent = PPO.load(args.antagonist_path, device="cpu")

                # Create the adversarial noise model
                noise_model = AdversarialNoiseModel(
                    antagonist_agent=antagonist_agent, device="cpu"
                )

            else:
                raise ValueError(f"Unsupported noise_type: {args.noise_type}")

            if noise_model:
                mm.set_noise_model(noise_model)

            # The NoisyWindFarmEnv will wrap the base env and use the configured MeasurementManager
            wrapped_env = NoisyWindFarmEnv(
                base_env_class=WindFarmEnv,
                measurement_manager=mm,
                **env_instance.kwargs,
            )
        else:
            wrapped_env = env_instance

        return Monitor(wrapped_env)

    return _init


def main(args: Args):
    if not os.path.exists(args.yaml_config_path):
        raise FileNotFoundError(
            f"YAML configuration file not found at: {args.yaml_config_path}"
        )

    # --- NEW: Create a more descriptive run name ---
    run_name = f"{args.run_name_prefix}_vs_{args.noise_type}"
    if args.noise_type == "adversarial":
        antagonist_id = os.path.basename(os.path.dirname(args.antagonist_path))
        run_name += f"_{antagonist_id}"
    run_name += f"_{int(time.time())}"
    # --- END NEW ---

    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
        save_code=True,
    )

    models_save_path = os.path.join(args.models_dir, wandb.run.id)
    os.makedirs(models_save_path, exist_ok=True)

    env_factories = [make_env_factory(args, i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_factories)

    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=dict(pi=net_arch_list, vf=net_arch_list))

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
                gradient_save_freq=0,
                model_save_path=os.path.join(models_save_path, "checkpoints"),
                verbose=2,
                model_save_freq=50_000 // args.n_envs,
            ),
        ]
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        model.save(os.path.join(models_save_path, "final_protagonist_model"))
        print(f"\n✅ Training complete. Final model saved to '{models_save_path}'")
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    args = tyro.cli(Args)
    main(args)
