from gymnasium.envs.registration import register
from .wind_farm_env import WindFarmEnv as WindFarmEnv
from .agent_eval import AgentEval as AgentEval
from .Agents import PyWakeAgent as PyWakeAgent
from .farm_eval import FarmEval as FarmEval
from .wind_env_multi import WindFarmEnvMulti as WindFarmEnvMulti
from .agent_eval import eval_single_fast as AgentEvalFast

__all__ = [
    "WindFarmEnv",
    "AgentEval",
    "PyWakeAgent",
    "FarmEval",
    "WindFarmEnvMulti",
    "AgentEvalFast",
]

register(
    id="WindGym/WindFarmEnv-v0",
    entry_point="WindGym.envs:WindFarmEnv",
)
