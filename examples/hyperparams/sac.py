import copy
import numpy as np

default = dict()
default["env_kwargs"] = dict(environment_count=1)
default["runner_kwargs"] = dict(num_steps_per_env=2048)
default["agent_kwargs"] = dict(
    actor_activations=["elu", "elu", "elu", "linear"],
    actor_hidden_dims=[128, 64, 32],
    actor_input_normalization=False,
    actor_noise_std=np.exp(0.0),
    batch_count=(default["env_kwargs"]["environment_count"] * default["runner_kwargs"]["num_steps_per_env"] // 64),
    critic_activations=["elu", "elu", "linear"],
    critic_hidden_dims=[128, 64, 32],
    critic_input_normalization=False,
)

"""
Standard:
[I 2023-01-17 07:43:46,884] Trial 125 finished with value: 150.23491836690064 and parameters: {
    'actor_net_activation': 'relu',
    'actor_net_arch': 'large',
    'actor_noise_std': 0.8504545432069994,
    'batch_count': 10,
    'clip_ratio': 0.1,
    'critic_net_activation': 'relu',
    'critic_net_arch': 'medium',
    'entropy_coeff': 0.0916881539697197,
    'env_count': 256,
    'gae_lambda': 0.95,
    'gamma': 0.955285858564339,
    'gradient_clip': 2.0,
    'learning_rate': 0.4762365866431558,
    'steps_per_env': 16,
    'recurrent': False,
    'target_kl': 0.19991906392721126,
    'value_coeff': 0.4434793554275927
}. Best is trial 125 with value: 150.23491836690064.
Hardcore:
[I 2023-01-09 06:25:44,000] Trial 262 finished with value: 2.290071208278338 and parameters: {
    'actor_noise_std': 0.2710521003644249,
    'batch_count': 6,
    'clip_ratio': 0.1,
    'entropy_coeff': 0.005105282891378981,
    'env_count': 16,
    'gae_lambda': 1.0,
    'gamma': 0.9718119008688937,
    'gradient_clip': 0.1,
    'learning_rate': 0.4569184610431825,
    'steps_per_env': 256,
    'target_kl': 0.11068348002480229,
    'value_coeff': 0.19453900570701116,
    'actor_net_arch': 'small',
    'critic_net_arch': 'medium',
    'actor_net_activation': 'relu',
    'critic_net_activation': 'relu'
}. Best is trial 262 with value: 2.290071208278338.
"""
bipedal_walker_v3 = copy.deepcopy(default)
bipedal_walker_v3["env_kwargs"]["environment_count"] = 256
bipedal_walker_v3["runner_kwargs"]["num_steps_per_env"] = 16
bipedal_walker_v3["agent_kwargs"]["actor_activations"] = ["elu", "elu", "elu", "linear"]
bipedal_walker_v3["agent_kwargs"]["actor_hidden_dims"] = [128, 64, 32]
bipedal_walker_v3["agent_kwargs"]["critic_activations"] = ["elu", "elu", "elu", "linear"]
bipedal_walker_v3["agent_kwargs"]["critic_hidden_dims"] = [128, 64, 32]
bipedal_walker_v3["agent_kwargs"]["action_max"] = 1.0
bipedal_walker_v3["agent_kwargs"]["action_min"] = -1.0
bipedal_walker_v3["agent_kwargs"]["actor_lr"] = 1e-4
bipedal_walker_v3["agent_kwargs"]["actor_noise_std"] = 0.8505
bipedal_walker_v3["agent_kwargs"]["alpha"] = 0.2
bipedal_walker_v3["agent_kwargs"]["alpha_lr"] = 0.00073
bipedal_walker_v3["agent_kwargs"]["critic_lr"] = 0.00073
bipedal_walker_v3["agent_kwargs"]["gradient_clip"] = 1.0
bipedal_walker_v3["agent_kwargs"]["log_std_max"] = 4.0
bipedal_walker_v3["agent_kwargs"]["log_std_min"] = -20.0
bipedal_walker_v3["agent_kwargs"]["storage_initial_size"] = 0
bipedal_walker_v3["agent_kwargs"]["storage_size"] = 300000
bipedal_walker_v3["agent_kwargs"]["target_entropy"] = 0.01
bipedal_walker_v3["agent_kwargs"]["batch_size"] = 256
bipedal_walker_v3["agent_kwargs"]["batch_count"] = 10

sac_hyperparams = {
    "BipedalWalker-v3": bipedal_walker_v3,
}
