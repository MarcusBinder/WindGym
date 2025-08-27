import gymnasium as gym
import yaml
import numpy as np
import os
import tempfile
import argparse  # For command-line arguments
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from wandb.integration.sb3 import WandbCallback
import wandb

from WindGym import WindFarmEnv  # Using WindFarmEnv for training
from WindGym.Measurement_Manager import (
    MeasurementManager,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    HybridNoiseModel,
    NoisyWindFarmEnv,
    MeasurementType,
)
from py_wake.examples.data.hornsrev1 import V80

# --- Base YAML Configuration String (will be formatted) ---
BASE_YAML_CONFIG_STRING = """
# WindGym Configuration (Parameterized)

# --- Initial Settings ---
yaw_init: "Zeros"
BaseController: "PyWake"
ActionMethod: "yaw"
Track_power: False

# --- Farm Parameters ---
farm:
  yaw_min: -30
  yaw_max: 30
  xDist: 5
  yDist: 3
  nx: 2
  ny: 1

# --- Wind Conditions (Sampling Range) ---
wind:
  ws_min: 7
  ws_max: 12
  TI_min: 0.05
  TI_max: 0.15
  wd_min: 265
  wd_max: 275

# --- Action Penalty ---
act_pen:
  action_penalty: 0.001
  action_penalty_type: "Change"

# --- Power Reward Definition ---
power_def:
  Power_reward: "Baseline"
  Power_avg: 10
  Power_scaling: 1.0

# --- Measurement Levels ---
mes_level:
  turb_ws: True
  turb_wd: {include_turbine_wd_yaml} # Set by --disable_turbine_wd
  turb_TI: True
  turb_power: False
  farm_ws: False
  farm_wd: {include_farm_wd_yaml}   # Set by --disable_farm_wd
  farm_TI: False
  farm_power: False

# --- Wind Speed Measurement Details ---
ws_mes:
  ws_current: True
  ws_rolling_mean: True
  ws_history_N: 3
  ws_history_length: 20
  ws_window_length: 5

# --- Wind Direction Measurement Details ---
wd_mes:
  wd_current: True
  wd_rolling_mean: True
  wd_history_N: 3
  wd_history_length: 20
  wd_window_length: 5

# --- Yaw Angle Measurement Details ---
yaw_mes:
  yaw_current: True
  yaw_rolling_mean: True
  yaw_history_N: 2
  yaw_history_length: 10
  yaw_window_length: 1

# --- Power Measurement Details ---
power_mes:
  power_current: True
  power_rolling_mean: True
  power_history_N: 3
  power_history_length: 20
  power_window_length: 5
"""


class WindGymCustomMonitor(BaseCallback):
    """
    A custom callback for logging detailed environment-specific metrics from WindGym to Weights & Biases.
    This version assumes per-turbine TI is available in the `info` dictionary.
    """

    def __init__(self, verbose=0):
        super(WindGymCustomMonitor, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This function is called after each step in the training process.
        It iterates through the `infos` dictionary from each parallel environment
        and logs the data to Weights & Biases.
        """
        infos = self.locals["infos"]

        for env_idx in range(len(infos)):
            info = infos[env_idx]
            log_data = {}

            # 1. Log global environment conditions
            log_data[f"custom_env_{env_idx}/global_wind_speed"] = info.get(
                "Wind speed Global"
            )
            log_data[f"custom_env_{env_idx}/global_wind_dir"] = info.get(
                "Wind direction Global"
            )
            log_data[f"custom_env_{env_idx}/global_ti"] = info.get(
                "Turbulence intensity"
            )

            # 2. Log farm-level power metrics
            agent_power = info.get("Power agent", 0)
            baseline_power = info.get("Power baseline")

            log_data[f"custom_env_{env_idx}/total_agent_power"] = agent_power

            if baseline_power is not None:
                log_data[f"custom_env_{env_idx}/total_baseline_power"] = baseline_power
                if baseline_power > 0:
                    power_gain = agent_power / baseline_power
                    log_data[f"custom_env_{env_idx}/power_gain_vs_baseline"] = (
                        power_gain
                    )

            # 3. Log per-turbine data from the 'info' dictionary
            num_turbines = len(info.get("yaw angles agent", []))

            # Since per-turbine TI is now in the info dict, we can get it here once
            turbine_tis = info.get("Turbulence intensity at turbines")

            for i in range(num_turbines):
                # Log yaw angle
                log_data[f"custom_env_{env_idx}/turbine_{i}/yaw_angle"] = info.get(
                    "yaw angles agent", [np.nan] * num_turbines
                )[i]

                # Log power output
                log_data[f"custom_env_{env_idx}/turbine_{i}/power"] = info.get(
                    "Power pr turbine agent", [np.nan] * num_turbines
                )[i]

                # Log local wind speed
                log_data[f"custom_env_{env_idx}/turbine_{i}/local_wind_speed"] = (
                    info.get("Wind speed at turbines", [np.nan] * num_turbines)[i]
                )

                # Log local wind direction (if available)
                if "Wind direction at turbines" in info:
                    log_data[f"custom_env_{env_idx}/turbine_{i}/local_wind_dir"] = (
                        info.get(
                            "Wind direction at turbines", [np.nan] * num_turbines
                        )[i]
                    )

                # Log TI directly from the info dictionary
                if turbine_tis is not None:
                    log_data[
                        f"custom_env_{env_idx}/turbine_{i}/turbulence_intensity"
                    ] = turbine_tis[i]

            for key in info:
                if key.startswith("obs_true/") or key.startswith("obs_sensed/"):
                    # Add to log_data, preserving the sub-path from the info key
                    log_data[f"custom_env_{env_idx}/{key}"] = info[key]

            # Log noise information if available
            if "noise_info" in info:
                noise_info = info["noise_info"]
                log_data[f"custom_env_{env_idx}/noise_type"] = noise_info.get(
                    "type", "unknown"
                )

                # Log specific noise metrics based on type
                if "bias_vector_norm (scaled)" in noise_info:
                    log_data[f"custom_env_{env_idx}/bias_vector_norm"] = noise_info[
                        "bias_vector_norm (scaled)"
                    ]

                if "applied_bias (physical_units)" in noise_info:
                    # Log a sample of the biases
                    bias_dict = noise_info["applied_bias (physical_units)"]
                    if bias_dict:
                        # Just log the first turbine's biases as an example
                        for key, value in list(bias_dict.items())[:3]:
                            log_data[f"custom_env_{env_idx}/noise_bias_{key}"] = value

            for key in info:
                if key.startswith("obs_true/"):
                    log_data[f"custom_env_{env_idx}/{key}"] = info[key]
                elif key.startswith("obs_sensed/"):
                    log_data[f"custom_env_{env_idx}/{key}"] = info[key]

            # Log all the collected data for this step to wandb
            wandb.log(log_data, step=self.num_timesteps)

        return True


def create_noise_model(noise_type_str, include_turbine_wd, include_farm_wd):
    """
    Create appropriate noise model based on configuration.

    Args:
        noise_type_str (str): "None", "White", "Bias", or "Hybrid"
        include_turbine_wd (bool): Whether turbine wind direction is included (impacts what measurements are available for noise)
        include_farm_wd (bool): Whether farm wind direction is included (impacts what measurements are available for noise)

    Returns:
        NoiseModel instance or None
    """
    if noise_type_str == "None":
        return None
    elif noise_type_str == "White":
        # Define noise standard deviations in physical units
        white_noise_std_devs = {
            MeasurementType.WIND_SPEED: 0.5,  # 0.5 m/s standard deviation
            MeasurementType.WIND_DIRECTION: 2.0,  # 2 degrees standard deviation
            MeasurementType.YAW_ANGLE: 0.5,  # 0.5 degrees standard deviation
            MeasurementType.TURBULENCE_INTENSITY: 0.01,  # 0.01 TI standard deviation
            MeasurementType.POWER: 10000.0,  # 10 kW standard deviation
        }
        return WhiteNoiseModel(white_noise_std_devs)
    elif noise_type_str == "Bias":
        # Define bias ranges in physical units
        # These ranges are for uniform sampling [min, max] for the bias
        episodic_bias_ranges = {
            MeasurementType.WIND_SPEED: (-1.0, 1.0),  # +/- 1.0 m/s bias
            MeasurementType.WIND_DIRECTION: (-5.0, 5.0),  # +/- 5.0 degrees bias
            MeasurementType.YAW_ANGLE: (-2.0, 2.0),  # +/- 2.0 degrees bias
        }
        return EpisodicBiasNoiseModel(episodic_bias_ranges)
    elif noise_type_str == "Hybrid":
        # Combine both white noise and episodic bias
        white_noise_std_devs = {
            MeasurementType.WIND_SPEED: 0.5,
            MeasurementType.WIND_DIRECTION: 2.0,
            MeasurementType.YAW_ANGLE: 0.5,
            MeasurementType.TURBULENCE_INTENSITY: 0.01,
            MeasurementType.POWER: 10000.0,
        }
        episodic_bias_ranges = {
            MeasurementType.WIND_SPEED: (-1.0, 1.0),
            MeasurementType.WIND_DIRECTION: (-5.0, 5.0),
            MeasurementType.YAW_ANGLE: (-2.0, 2.0),
        }
        return HybridNoiseModel(
            models=[
                WhiteNoiseModel(white_noise_std_devs),
                EpisodicBiasNoiseModel(episodic_bias_ranges),
            ]
        )
    else:
        raise ValueError(
            f"Unknown noise type: {noise_type_str}. Choose from 'None', 'White', 'Bias', 'Hybrid'."
        )


def make_env(
    temp_yaml_path,
    turbine_obj,
    env_init_params,
    seed=0,
    noise_type_str="None",
    include_turbine_wd=True,
    include_farm_wd=True,
):
    def _init():
        # 1. LOAD THE CONFIG
        with open(temp_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 2. GET FARM PARAMS
        farm_params = config["farm"]
        nx = farm_params["nx"]
        ny = farm_params["ny"]
        x_dist = farm_params["xDist"]
        y_dist = farm_params["yDist"]

        # 3. GET DIAMETER from the turbine object
        D = turbine_obj.diameter()

        # 4. CALCULATE COORDINATES
        x_coords = np.arange(nx) * x_dist * D
        y_coords = np.arange(ny) * y_dist * D
        x_pos, y_pos = np.meshgrid(x_coords, y_coords)
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()

        # 5. CREATE BASE ENVIRONMENT (without noise)
        temp_base_env_for_mm_setup = WindFarmEnv(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            config=temp_yaml_path,
            seed=seed,  # Pass the seed for reproducibility of internal randomness
            dt_sim=env_init_params.get("dt_sim", 1),
            dt_env=env_init_params.get("dt_env", 10),
            yaw_step_sim=env_init_params.get("yaw_step", 1.0),
            # Use 'MannGenerate' as default if not specified or TurbBox not provided,
            # to avoid FileNotFoundError if 'MannLoad' is default and path is missing.
            turbtype=env_init_params.get("turbtype", "MannGenerate"),
            TurbBox=env_init_params.get("TurbBox"),
            n_passthrough=env_init_params.get("n_passthrough", 10),
            fill_window=env_init_params.get("fill_window", True),
            Baseline_comp=True,
            # Ensure MeasurementManager's `_build_from_env` gets the correct scaling ranges from the `temp_base_env_for_mm_setup`
            # These are usually passed via yaml_path now, but explicitly setting them here
            # ensures the MeasurementSpec min/max are correct for scaling noise.
            TI_min_mes=config["mes_level"].get(
                "TI_min_mes", 0.0
            ),  # Assuming these are defined in mes_level if needed
            TI_max_mes=config["mes_level"].get("TI_max_mes", 0.50),
            ws_scaling_min=config["wind"].get(
                "ws_min", 0.0
            ),  # Using wind min/max for scaling range in MesClass
            ws_scaling_max=config["wind"].get("ws_max", 30.0),
            wd_scaling_min=config["wind"].get("wd_min", 0.0),
            wd_scaling_max=config["wind"].get("wd_max", 360.0),
            yaw_scaling_min=config["farm"].get("yaw_min", -45.0),
            yaw_scaling_max=config["farm"].get("yaw_max", 45.0),
        )

        # 6. CREATE MEASUREMENT MANAGER AND NOISE MODEL
        #    The MeasurementManager is initialized with the temporary base environment
        #    to allow it to build `self.specs` correctly based on the env's observation
        #    space structure and scaling ranges.
        measurement_manager = MeasurementManager(temp_base_env_for_mm_setup, seed=seed)
        noise_model = create_noise_model(
            noise_type_str, include_turbine_wd, include_farm_wd
        )

        final_env_kwargs = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "turbine": turbine_obj,
            "config": temp_yaml_path,
            "seed": seed,
            "dt_sim": env_init_params.get("dt_sim", 1),
            "dt_env": env_init_params.get("dt_env", 10),
            "yaw_step_sim": env_init_params.get("yaw_step", 1.0),
            "turbtype": env_init_params.get("turbtype", "MannGenerate"),
            "TurbBox": env_init_params.get("TurbBox"),
            "n_passthrough": env_init_params.get("n_passthrough", 10),
            "fill_window": env_init_params.get("fill_window", True),
            "Baseline_comp": True,
            # Pass these scaling parameters for the *actual* WindFarmEnv instance
            # that NoisyWindFarmEnv will create internally.
            "TI_min_mes": config["mes_level"].get("TI_min_mes", 0.0),
            "TI_max_mes": config["mes_level"].get("TI_max_mes", 0.50),
            "ws_scaling_min": config["wind"].get("ws_min", 0.0),
            "ws_scaling_max": config["wind"].get("ws_max", 30.0),
            "wd_scaling_min": config["wind"].get("wd_min", 0.0),
            "wd_scaling_max": config["wind"].get("wd_max", 360.0),
            "yaw_scaling_min": config["farm"].get("yaw_min", -45.0),
            "yaw_scaling_max": config["farm"].get("yaw_max", 45.0),
        }

        if noise_model is not None:
            measurement_manager.set_noise_model(noise_model)
            # Wrap with NoisyWindFarmEnv. Note: NoisyWindFarmEnv *re-creates* the base environment
            # using the base_env_class and the provided **env_kwargs.
            # This is why the `temp_base_env_for_mm_setup` above is created and then implicitly
            # discarded (by Python's garbage collector) after `measurement_manager.specs` are built.
            env = NoisyWindFarmEnv(
                base_env_class=WindFarmEnv,
                measurement_manager=measurement_manager,
                **final_env_kwargs,  # Pass all necessary kwargs for base_env_class
            )
        else:
            # If no noise, use the plain WindFarmEnv instance.
            # In this case, `temp_base_env_for_mm_setup` *is* the actual environment.
            env = temp_base_env_for_mm_setup

        return env

    return _init


def train_agent(args):
    # Format the YAML string based on command-line arguments
    include_turbine_wd_yaml_str = (
        "false" if args.disable_turbine_wd else "true"
    )  # YAML expects lowercase true/false
    include_farm_wd_yaml_str = "false" if args.disable_farm_wd else "true"

    current_yaml_config_string = BASE_YAML_CONFIG_STRING.format(
        include_turbine_wd_yaml=include_turbine_wd_yaml_str,
        include_farm_wd_yaml=include_farm_wd_yaml_str,
    )
    print("--- Using YAML Configuration ---")
    print(current_yaml_config_string)
    print("--------------------------------")
    if args.noise_type == "None":  # Changed from noise_level to noise_type
        print("INFO: No observation noise will be applied by MeasurementManager.")
    else:
        print(
            f"INFO: MeasurementManager will apply '{args.noise_type}' noise to observations."
        )  # Changed from noise_level to noise_type
    if args.disable_turbine_wd:
        print("INFO: Turbine wind direction measurements will be DISABLED.")
    if args.disable_farm_wd:
        print("INFO: Farm wind direction measurements will be DISABLED.")

    # W&B and Output Directory Setup
    run_name = f"{args.run_name_prefix}_noise-{args.noise_type}_turbWD-{not args.disable_turbine_wd}_farmWD-{not args.disable_farm_wd}_{wandb.util.generate_id()}"  # Changed from noise_level to noise_type

    run = wandb.init(
        project=args.project_name,
        config={
            **vars(args),
            "yaml_config_string": current_yaml_config_string,
        },  # Log all args and YAML
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    models_save_base = os.path.join(args.models_base_dir, run.id)
    tensorboard_log_base = os.path.join(args.tensorboard_base_dir, run.id)
    os.makedirs(models_save_base, exist_ok=True)
    os.makedirs(tensorboard_log_base, exist_ok=True)

    # Ensure TurbBox path is valid
    turbbox_path_train = args.turbbox_path
    if not os.path.isfile(turbbox_path_train) and args.turbtype == "MannLoad":
        potential_path_cwd = os.path.join(os.getcwd(), turbbox_path_train)
        if os.path.isfile(potential_path_cwd):
            turbbox_path_train = potential_path_cwd
            print(f"Using TurbBox found in CWD: {turbbox_path_train}")
        else:
            print(
                f"ERROR: Turbulence box file for MannLoad not found at '{turbbox_path_train}'."
            )
            exit()

    env_init_params = {
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step": args.yaw_step,
        "turbtype": args.turbtype,
        "TurbBox": turbbox_path_train,
        "n_passthrough": args.n_passthrough,
        "fill_window": args.fill_window,
    }

    turbine = V80()
    temp_yaml_filepath = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as tmp_yaml_file:
            tmp_yaml_file.write(current_yaml_config_string)
            temp_yaml_filepath = tmp_yaml_file.name
        # The 'with' block ensures the file is closed here after writing.

        print(f"Temporary YAML config for training: {temp_yaml_filepath}")

        # Now, create the SubprocVecEnv after the file is definitely written and closed.
        vec_env = SubprocVecEnv(
            [
                make_env(
                    temp_yaml_filepath,
                    turbine,
                    env_init_params,
                    seed=args.seed + i,
                    noise_type_str=args.noise_type,  # Changed from noise_level to noise_type
                    include_turbine_wd=not args.disable_turbine_wd,
                    include_farm_wd=not args.disable_farm_wd,
                )
                for i in range(args.n_envs)
            ]
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=os.path.join(models_save_base, "checkpoints"),
            name_prefix="yaw_agent_model",
        )

        # Eval Env setup:
        # Create a single environment for evaluation. The make_env returns a factory, so call it.
        eval_env_instance_factory = make_env(
            temp_yaml_filepath,
            turbine,
            env_init_params,
            seed=args.seed + 1000,
            noise_type_str=args.noise_type,  # Changed from noise_level to noise_type
            include_turbine_wd=not args.disable_turbine_wd,
            include_farm_wd=not args.disable_farm_wd,
        )
        eval_env_instance = (
            eval_env_instance_factory()
        )  # Instantiate the environment for evaluation

        eval_callback = EvalCallback(
            eval_env_instance,
            best_model_save_path=os.path.join(models_save_base, "best_model"),
            log_path=os.path.join(models_save_base, "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            deterministic=True,
            render=False,
        )

        wandb_sb3_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=os.path.join(models_save_base, "wandb_models"),
            model_save_freq=max(args.save_freq // args.n_envs, 1),  # Same as checkpoint
            verbose=2,
        )

        custom_monitor_callback = WindGymCustomMonitor()
        all_callbacks = [
            wandb_sb3_callback,
            checkpoint_callback,
            eval_callback,
            custom_monitor_callback,
        ]

        policy_kwargs = dict(
            net_arch=dict(
                pi=[int(x) for x in args.net_arch_pi.split(",")],
                vf=[int(x) for x in args.net_arch_vf.split(",")],
            )
        )

        model = PPO(
            args.policy_type,
            vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log_base,
            seed=args.seed,
        )

        print(f"Starting training for run: {run.name}")
        print(
            f"Model and logs will be saved in subdirectories of: {args.models_base_dir} and {args.tensorboard_base_dir}"
        )
        model.learn(total_timesteps=args.total_timesteps, callback=all_callbacks)

        model.save(os.path.join(models_save_base, "final_model"))
        print(
            f"Final model saved to: {os.path.join(models_save_base, 'final_model.zip')}"
        )

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()
        raise (e)
    finally:
        if "vec_env" in locals():
            vec_env.close()
        if "eval_env_instance" in locals():
            eval_env_instance.close()
        if temp_yaml_filepath and os.path.exists(temp_yaml_filepath):
            os.remove(temp_yaml_filepath)
            print(f"Temporary YAML file {temp_yaml_filepath} deleted.")
        if "run" in locals():
            run.finish()
        print("Training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for WindGym.")

    # Experiment Setup
    parser.add_argument(
        "--project_name",
        type=str,
        default="WindGym_Parameterized_Training",
        help="WandB project name.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="PPO_WindGym",
        help="Prefix for WandB run name.",
    )
    parser.add_argument(
        "--models_base_dir",
        type=str,
        default="./trained_models_param",
        help="Base directory to save models.",
    )
    parser.add_argument(
        "--tensorboard_base_dir",
        type=str,
        default="./sb3_tensorboard_logs_param",
        help="Base directory for SB3 TensorBoard logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["None", "White", "Bias", "Hybrid"],
        default="Bias",
        help="Observation noise type (handled by MeasurementManager).",
    )

    parser.add_argument(
        "--disable_turbine_wd",
        action="store_true",
        help="Disable turbine wind direction measurements.",
    )
    parser.add_argument(
        "--disable_farm_wd",
        action="store_true",
        help="Disable farm wind direction measurements.",
    )

    # WindFarmEnv Direct Parameters
    parser.add_argument(
        "--dt_sim", type=float, default=1.0, help="DWM simulation timestep (s)."
    )
    parser.add_argument(
        "--dt_env", type=float, default=10.0, help="Agent environment timestep (s)."
    )
    parser.add_argument(
        "--yaw_step",
        type=float,
        default=1.0,
        help="Max yaw change per env step (degrees).",
    )
    parser.add_argument(
        "--turbtype",
        type=str,
        default="MannGenerate",
        choices=["MannLoad", "MannGenerate", "MannFixed", "Random", "None"],
        help="Turbulence type.",
    )
    parser.add_argument(
        "--turbbox_path",
        type=str,
        default="Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc",
        help="Path to turbulence box file (if MannLoad).",
    )
    parser.add_argument(
        "--n_passthrough",
        type=int,
        default=10,
        help="Number of passthroughs for episode length calculation.",
    )
    parser.add_argument(
        "--fill_window",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Fill observation window at reset (True/False).",
    )

    # PPO Hyperparameters
    parser.add_argument(
        "--policy_type", type=str, default="MlpPolicy", help="Policy type for PPO."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps for training.",
    )
    parser.add_argument(
        "--n_envs", type=int, default=4, help="Number of parallel environments."
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per environment per PPO update.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Minibatch size for PPO."
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs when optimizing the surrogate loss.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="Factor for GAE."
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy coefficient."
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--net_arch_pi",
        type=str,
        default="128,128",
        help="Policy network architecture (comma-separated ints).",
    )
    parser.add_argument(
        "--net_arch_vf",
        type=str,
        default="128,128",
        help="Value network architecture (comma-separated ints).",
    )

    # Callback Frequencies
    parser.add_argument(
        "--save_freq",
        type=int,
        default=50_000,
        help="Frequency to save checkpoints (global steps).",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=25_000,
        help="Frequency to run evaluations (global steps).",
    )

    args = parser.parse_args()
    train_agent(args)
