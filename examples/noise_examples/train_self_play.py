# filename: examples/noise_examples/train_self_play.py

import os
import time
import tyro
import yaml
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from dataclasses import dataclass
import wandb
from pettingzoo import ParallelEnv
from typing import Dict, List

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from noise_definitions import AdversarialNoiseModel, get_adversarial_constraints


# ============================================================================
# 1. PettingZoo Environment for Self-Play
# ============================================================================
class SelfPlayPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "WindFarmSelfPlay-v0"}

    def __init__(self, base_env: WindFarmEnv, constraints: dict):
        super().__init__()
        self.base_env = base_env

        self.agents = ["protagonist", "antagonist"]
        self.possible_agents = self.agents.copy()

        self.noise_model = AdversarialNoiseModel(constraints=constraints)
        self.mm = MeasurementManager(self.base_env, seed=self.base_env.seed)
        self.mm.set_noise_model(self.noise_model)

        self._observation_spaces = {
            "protagonist": self.base_env.observation_space,
            "antagonist": self.base_env.observation_space,
        }
        self._action_spaces = {
            "protagonist": self.base_env.action_space,
            "antagonist": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.base_env.n_turb * len(constraints),),
                dtype=np.float32,
            ),
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        clean_obs, info = self.base_env.reset(seed=seed)
        self.noise_model.reset_noise(self.mm.specs, self.base_env.np_random)

        self.agents = self.possible_agents[:]
        observations = {agent: clean_obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # This check is crucial for when the training loop calls step() after an episode is done.
        if not self.agents:
            # PettingZoo environments should be reset after an episode ends.
            # This path is a safeguard against unexpected calls.
            obs, info = self.reset()
            return (
                obs,
                {a: 0 for a in self.possible_agents},
                {a: False for a in self.possible_agents},
                {a: False for a in self.possible_agents},
                info,
            )

        # 1. Execute protagonist's action to get the next state (t+1)
        next_clean_obs, prot_reward, terminated, truncated, info = self.base_env.step(
            actions["protagonist"]
        )

        done = terminated or truncated

        # 2. If the episode is ongoing, generate noisy observation for the protagonist.
        if not done:
            noisy_obs_for_protagonist = self.noise_model.apply_noise(
                next_clean_obs,
                self.mm.specs,
                self.base_env.np_random,
                adv_action=actions["antagonist"],
            )
            observations = {
                "protagonist": noisy_obs_for_protagonist,
                "antagonist": next_clean_obs,
            }
        else:
            self.agents = []
            # On the final step, observations don't matter but need to have the right shape
            observations = {
                agent: np.zeros_like(self.observation_space(agent).sample())
                for agent in self.possible_agents
            }

        # 3. Prepare return dictionaries
        rewards = {"protagonist": prot_reward, "antagonist": -prot_reward}
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {agent: info for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.base_env.close()


# ============================================================================
# 2. PPO Agent and Training Arguments
# ============================================================================
@dataclass
class Args:
    project_name: str = "WindGym_SelfPlay"
    run_name: str = "PPO_SelfPlay"
    seed: int = 42
    total_timesteps: int = 3000000
    n_envs: int = (
        1  # Co-training is complex with SubprocVecEnv, let's use 1 for this script
    )
    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: str = "128,128"
    models_dir: str = "models/self_play"


class Agent(nn.Module):
    def __init__(self, obs_space, act_space, net_arch):
        super().__init__()
        obs_shape = np.array(obs_space.shape).prod()
        act_shape = np.prod(act_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, net_arch[0])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[0], net_arch[1])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[1], 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, net_arch[0])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[0], net_arch[1])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[1], act_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape))

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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# 3. Main Training Loop
# ============================================================================
def main(args: Args):
    run_name = f"{args.run_name}_{int(time.time())}"
    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
    )

    # Seeding for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.yaml_config_path, "r") as f:
        config = yaml.safe_load(f)
    x_pos, y_pos = generate_square_grid(
        turbine=V80(),
        nx=config["farm"]["nx"],
        ny=config["farm"]["ny"],
        xDist=config["farm"]["xDist"],
        yDist=config["farm"]["yDist"],
    )
    base_env = WindFarmEnv(
        x_pos=x_pos,
        y_pos=y_pos,
        turbine=V80(),
        config=args.yaml_config_path,
        turbtype="None",
        seed=args.seed,
        reset_init=False,
    )
    env = SelfPlayPettingZooEnv(base_env, get_adversarial_constraints())

    agents = {}
    optimizers = {}
    net_arch = [int(x) for x in args.net_arch.split(",")]
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(
            env.observation_space(agent_id), env.action_space(agent_id), net_arch
        ).to(device)
        optimizers[agent_id] = optim.Adam(
            agents[agent_id].parameters(), lr=args.learning_rate, eps=1e-5
        )

    batch_size = int(args.n_envs * args.n_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // batch_size

    obs_shape = {aid: env.observation_space(aid).shape for aid in env.possible_agents}
    act_shape = {aid: env.action_space(aid).shape for aid in env.possible_agents}

    obs = {
        aid: torch.zeros((args.n_steps, args.n_envs) + obs_shape[aid]).to(device)
        for aid in env.possible_agents
    }
    actions = {
        aid: torch.zeros((args.n_steps, args.n_envs) + act_shape[aid]).to(device)
        for aid in env.possible_agents
    }
    logprobs = {
        aid: torch.zeros((args.n_steps, args.n_envs)).to(device)
        for aid in env.possible_agents
    }
    rewards = {
        aid: torch.zeros((args.n_steps, args.n_envs)).to(device)
        for aid in env.possible_agents
    }
    dones = torch.zeros((args.n_steps, args.n_envs)).to(device)
    values = {
        aid: torch.zeros((args.n_steps, args.n_envs)).to(device)
        for aid in env.possible_agents
    }

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = {
        agent_id: torch.Tensor(ob).to(device) for agent_id, ob in next_obs.items()
    }
    next_done = torch.zeros(args.n_envs).to(device)

    for update in range(1, num_updates + 1):
        for step in range(0, args.n_steps):
            global_step += 1 * args.n_envs
            dones[step] = next_done

            step_actions = {}
            with torch.no_grad():
                for agent_id in env.agents:
                    obs[agent_id][step] = next_obs[agent_id]
                    action, logprob, _, value = agents[agent_id].get_action_and_value(
                        next_obs[agent_id].unsqueeze(0)
                    )
                    values[agent_id][step] = value.flatten()
                    actions[agent_id][step] = action.squeeze(0)
                    logprobs[agent_id][step] = logprob.flatten()
                    step_actions[agent_id] = action.squeeze(0).cpu().numpy()

            (
                next_obs_dict,
                rewards_dict,
                terminations_dict,
                truncations_dict,
                info_dict,
            ) = env.step(step_actions)

            done_for_this_step = any(terminations_dict.values()) or any(
                truncations_dict.values()
            )

            for agent_id in env.possible_agents:
                rewards[agent_id][step] = (
                    torch.tensor(rewards_dict.get(agent_id, 0)).to(device).view(-1)
                )

            if done_for_this_step:
                next_obs_dict, _ = env.reset()

            next_obs = {
                agent_id: torch.Tensor(ob).to(device)
                for agent_id, ob in next_obs_dict.items()
            }
            next_done = torch.Tensor([done_for_this_step] * args.n_envs).to(device)

        # --- Update Phase ---
        for agent_id in env.possible_agents:
            with torch.no_grad():
                next_value = (
                    agents[agent_id]
                    .get_value(next_obs[agent_id].unsqueeze(0))
                    .reshape(1, -1)
                )
                advantages = torch.zeros_like(rewards[agent_id]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.n_steps)):
                    nextnonterminal = 1.0 - (
                        dones[t + 1] if t < args.n_steps - 1 else next_done
                    )
                    nextvalues = (
                        values[agent_id][t + 1] if t < args.n_steps - 1 else next_value
                    )
                    delta = (
                        rewards[agent_id][t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[agent_id][t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values[agent_id]

            b_obs = obs[agent_id].reshape((-1,) + env.observation_space(agent_id).shape)
            b_logprobs = logprobs[agent_id].reshape(-1)
            b_actions = actions[agent_id].reshape(
                (-1,) + env.action_space(agent_id).shape
            )
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values[agent_id].reshape(-1)

            b_inds = np.arange(batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agents[
                        agent_id
                    ].get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages
                        * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    v_loss = (
                        0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                    )
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )
                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agents[agent_id].parameters(), args.max_grad_norm
                    )
                    optimizers[agent_id].step()

        wandb.log(
            {
                "global_step": global_step,
                "protagonist_reward_mean": torch.mean(rewards["protagonist"]).item(),
                "antagonist_reward_mean": torch.mean(rewards["antagonist"]).item(),
                "SPS": int(global_step / (time.time() - start_time)),
            }
        )

    os.makedirs(args.models_dir, exist_ok=True)
    for agent_id, agent_model in agents.items():
        model_path = os.path.join(args.models_dir, f"{run_name}_{agent_id}.pt")
        torch.save(agent_model.state_dict(), model_path)
        print(f"Saved {agent_id} model to {model_path}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
