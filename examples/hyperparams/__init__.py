from rsl_rl.algorithms import PPO, DPPO, SAC
from .dppo import dppo_hyperparams
from .ppo import ppo_hyperparams
from .sac import sac_hyperparams

hyperparams = {DPPO.__name__: dppo_hyperparams, PPO.__name__: ppo_hyperparams, SAC.__name__: sac_hyperparams}

__all__ = ["hyperparams"]
