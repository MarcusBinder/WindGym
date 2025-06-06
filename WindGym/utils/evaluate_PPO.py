from windgym.WindGym import FarmEval
from windgym.WindGym.wrappers import RecordEpisodeVals
import gymnasty as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
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


def make_eval_env(idx, wind_turbine, yaml_path, dt_sim, dt_env):
    """
    Makes the evaluation environment. Used for the evaluation of the model
    """
    ws_list = [
        10,
        11,
        12,
        10,
        11,
        12,
        10,
        11,
        12,
        10,
        11,
        12,
        10,
        11,
        12,
        10,
        11,
        12,
        10,
        11,
        12,
    ]
    wd_list = [
        255,
        260,
        265,
        270,
        275,
        280,
        285,
        255,
        260,
        265,
        270,
        275,
        280,
        285,
        255,
        260,
        265,
        270,
        275,
        280,
        285,
    ]
    env = FarmEval(
        turbine=wind_turbine(),
        n_passthrough=4000,  # this value is not used, as it is overridden in the env.
        yaw_init="Zeros",
        #   TurbBox="/work/users/manils/wesc/Boxes/V80env/",
        TurbBox="/work/users/manils/p2t2t/Boxes/V80env/",
        yaml_path=yaml_path,
        dt_sim=dt_sim,
        dt_env=dt_env,
        reset_init=False,
        turbtype="None",  # the type of turbulence.
    )

    ws = ws_list[idx]
    wd = wd_list[idx]

    env.set_wind_vals(ws=ws, ti=0.05, wd=wd)
    return env


def evaluate(
    trained_agent,
    n_envs=21,
    n_steps=1000,
):
    """Function to evaluate the performance of the model.
    Heavily based on the code from cleanrl. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/evals/ppo_eval.py
    """

    # First we make the eval envs.
    eval_envs = gym.vector.AsyncVectorEnv(
        [lambda: make_eval_env(idx) for idx in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    eval_envs = RecordEpisodeVals(eval_envs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the agent that we want to evaluate, and copy the weights from the trained agent.
    eval_agent = Agent(eval_envs).to(device)

    eval_agent.load_state_dict(trained_agent.state_dict())

    eval_agent.eval()

    # Now the agent is ready to be trained. We then reset the environment, and do n_steps.
    obs, _ = eval_envs.reset()

    for _ in range(n_steps):
        actions, _, _, _ = eval_agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())
        obs = next_obs

    # The eval score is the total power produced by the wind farm, divided by the number of environments and the number of turbines, and the steps.
    eval_score = (
        eval_envs.episode_powers.sum()
        / eval_envs.num_envs
        / eval_envs.single_action_space.shape[0]
        / n_steps
    )

    # Clean up
    eval_envs.close()
    del eval_envs
    del eval_agent

    return eval_score
