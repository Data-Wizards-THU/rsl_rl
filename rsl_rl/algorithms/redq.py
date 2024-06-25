import numpy as np
import torch
from torch import nn, optim
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.modules import Network, GaussianChimeraNetwork, GaussianNetwork
from rsl_rl.storage.replay_storage import ReplayStorage
from rsl_rl.storage.storage import Dataset
from rsl_rl.utils import get_probabilistic_num_min


class REDQ(AbstractActorCritic):
    """Randomized Ensembled Double Q-Learning algorithm.

    This is an implementation of the REDQ algorithm by Chen et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/2101.05982

    """

    critic_network: Type[nn.Module] = Network

    def __init__(
        self,
        env: VecEnv,
        action_max: float = 100.0,
        action_min: float = -100.0,
        actor_lr: float = 1e-4,
        actor_noise_std: float = 1.0,
        alpha: float = 0.2,
        alpha_lr: float = 1e-3,
        chimera: bool = True,
        critic_lr: float = 1e-3,
        gradient_clip: float = 1.0,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        storage_initial_size: int = 0,
        storage_size: int = 100000,
        target_entropy: float = None,
        num_Q: int = 10,
        num_sample: int = 2,
        q_target_mode: str = "min",
        policy_update_delay: int = 20,
        **kwargs
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
            actor_lr (float): Learning rate for the actor.
            alpha (float): Initial entropy regularization coefficient.
            alpha_lr (float): Learning rate for entropy regularization coefficient.
            chimera (bool): Whether to use separate heads for computing action mean and std (True) or treat the std as a
                tunable parameter (True).
            critic_lr (float): Learning rate for the critic.
            gradient_clip (float): Gradient clip value.
            log_std_max (float): Maximum log standard deviation.
            log_std_min (float): Minimum log standard deviation.
            storage_initial_size (int): Initial size of the replay storage.
            storage_size (int): Maximum size of the replay storage.
            target_entropy (float): Target entropy for the actor policy. Defaults to action space dimensionality.
            num_Q (int): Number of Q networks in the Q ensemble.
            num_sample (int): Number of sampled Q values to take minimal from.
            q_target_mode (str): 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture, 
                currently only support min
            policy_update_delay (int): How many updates the critic do until we update policy network.
        """
        super().__init__(env, action_max=action_max, action_min=action_min, **kwargs)

        self.storage = ReplayStorage(
            self.env.num_envs, storage_size, device=self.device, initial_size=storage_initial_size
        )

        self._register_serializable("storage")

        assert self._action_max < np.inf, 'Parameter "action_max" needs to be set for SAC.'
        assert self._action_min > -np.inf, 'Parameter "action_min" needs to be set for SAC.'

        self._action_delta = 0.5 * (self._action_max - self._action_min)
        self._action_offset = 0.5 * (self._action_max + self._action_min)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32).requires_grad_()
        self._gradient_clip = gradient_clip
        self._target_entropy = target_entropy if target_entropy else -self._action_size
        self.num_Q = num_Q
        self.num_sample = num_sample
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay

        self._register_serializable("log_alpha", "_gradient_clip")

        network_class = GaussianChimeraNetwork if chimera else GaussianNetwork
        self.actor = network_class(
            self._actor_input_size,
            self._action_size,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            std_init=actor_noise_std,
            **self._actor_network_kwargs
        )

        # self.critic_1 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        # self.critic_2 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)

        # self.target_critic_1 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        # self.target_critic_2 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        # self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        # self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.critics = []
        self.target_critics = []
        for _ in range(num_Q):
            critic = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
            target_critic = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
            target_critic.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.target_critics.append(target_critic)

        # self._register_serializable("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
        self._register_serializable("actor", "critics", "target_critics")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        # self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.critic_optimizers = []
        for critic in self.critics:
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=critic_lr))

        # self._register_serializable(
        #     "actor_optimizer", "log_alpha_optimizer", "critic_1_optimizer", "critic_2_optimizer"
        # )
        self._register_serializable("actor_optimizer", "log_alpha_optimizer", "critic_optimizers")

        self.critic = self.critics[0]

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def draw_actions(
        self, obs: torch.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        actor_obs, critic_obs = self._process_observations(obs, env_info)

        action = self._sample_action(actor_obs, compute_logp=False)
        data = {"actor_observations": actor_obs.clone(), "critic_observations": critic_obs.clone()}

        return action, data

    def eval_mode(self) -> AbstractActorCritic:
        super().eval_mode()

        self.actor.eval()
        for critic in self.critics:
            critic.eval()
        for target_critic in self.target_critics:
            target_critic.eval()

        return self

    def get_inference_policy(self, device=None) -> Callable:
        self.to(device)
        self.eval_mode()

        def policy(obs, env_info=None):
            obs, _ = self._process_observations(obs, env_info)
            actions = self._scale_actions(self.actor.forward(obs))

            # actions, _ = self.draw_actions(obs, env_info)

            return actions

        return policy

    def register_terminations(self, terminations: torch.Tensor) -> None:
        pass

    def to(self, device: str) -> AbstractActorCritic:
        """Transfers agent parameters to device."""
        super().to(device)

        self.actor.to(device)
        for critic in self.critics:
            critic.to(device)
        for target_critic in self.target_critics:
            target_critic.to(device)
        self.log_alpha.to(device)

        return self

    def train_mode(self) -> AbstractActorCritic:
        super().train_mode()

        self.actor.train()
        for critic in self.critics:
            critic.train()
        for target_critic in self.target_critics:
            target_critic.train()

        return self

    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        super().update(dataset)

        if not self.initialized:
            return {}

        total_actor_loss = []
        total_alpha_loss = []
        total_critic_loss = [torch.zeros(self._batch_count) for _ in range(self.num_Q)]

        for idx, batch in enumerate(self.storage.batch_generator(self._batch_size, self._batch_count)):
            actor_obs = batch["actor_observations"]
            critic_obs = batch["critic_observations"]
            actions = batch["actions"].reshape(-1, self._action_size)
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"]

            critic_loss = self._update_critic(
                critic_obs, actions, rewards, dones, actor_next_obs, critic_next_obs
            ) # list of critic loss

            # delayed policy update
            if (idx + 1) % self.policy_update_delay == 0 or idx == self._batch_count - 1:
                actor_loss, alpha_loss = self._update_actor_and_alpha(actor_obs, critic_obs)
                total_actor_loss.append(actor_loss.item())
                total_alpha_loss.append(alpha_loss.item())

            # Update Target Networks
            for critic, target_critic in zip(self.critics, self.target_critics):
                self._update_target(critic, target_critic)

            for i in range(self.num_Q):
                total_critic_loss[i][idx] = critic_loss[i].item()

        stats = {
            "actor": sum(total_actor_loss) / len(total_actor_loss),
            "alpha": sum(total_alpha_loss) / len(total_alpha_loss),
        }
        for i in range(self.num_Q):
            stats[f"critic_{i}"] = total_critic_loss[i].mean().item()

        return stats

    def _sample_action(
        self, observation: torch.Tensor, compute_logp=True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """Samples and action from the policy.

        Args:
            observation (torch.Tensor): The observation to sample an action for.
            compute_logp (bool): Whether to compute and return the action log probability. Default to True.
        Returns:
            Either the action as a torch.Tensor or, if compute_logp is set to true, a tuple containing the actions as a
            torch.Tensor and the action log probability.
        """
        mean, std = self.actor.forward(observation, compute_std=True)
        dist = torch.distributions.Normal(mean, std)

        actions = dist.rsample()
        actions_normalized, actions_scaled = self._scale_actions(actions, intermediate=True)

        if not compute_logp:
            return actions_scaled

        action_logp = dist.log_prob(actions).sum(-1) - torch.log(1.0 - actions_normalized.pow(2) + 1e-6).sum(-1)

        return actions_scaled, action_logp

    def _scale_actions(self, actions: torch.Tensor, intermediate=False) -> torch.Tensor:
        actions = actions.reshape(-1, self._action_size)
        action_normalized = torch.tanh(actions)
        # action_scaled = super()._process_actions(action_normalized * self._action_delta + self._action_offset)
        action_scaled = super()._process_actions(actions)

        if intermediate:
            return action_normalized, action_scaled

        return action_scaled

    def _update_actor_and_alpha(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_prediction, actor_prediction_logp = self._sample_action(actor_obs)

        # Update alpha (also called temperature / entropy coefficient)
        alpha_loss = -(self.log_alpha * (actor_prediction_logp + self._target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Update actor
        evaluation_input = self._critic_input(critic_obs, actor_prediction)
        evaluations = []
        for idx in range(self.num_Q):
            self.critics[idx].requires_grad_(False)
            evaluations.append(self.critics[idx].forward(evaluation_input))
        evaluations_cat = torch.stack(evaluations)
        ave_eval = evaluations_cat.mean(dim=0)
        actor_loss = (self.alpha.detach() * actor_prediction_logp - ave_eval).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self._gradient_clip)
        self.actor_optimizer.step()

        for idx in range(self.num_Q):
            self.critics[idx].requires_grad_(True)

        return actor_loss, alpha_loss

    def _get_sampled_q_target_no_grad(
        self,
        actor_next_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
    ) -> torch.Tensor:
        # sample self.num_sample Q values from the ensemble
        num_mins_to_use = get_probabilistic_num_min(self.num_sample)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            if self.q_target_mode == 'min':
                """Q target is min of a subset of Q values"""
                target_action, target_action_logp = self._sample_action(actor_next_obs)
                target_critic_input = self._critic_input(critic_next_obs, target_action)
                for sample_idx in sample_idxs:
                    target_critic_prediction = self.target_critics[sample_idx].forward(target_critic_input)
                    if sample_idx == sample_idxs[0]:
                        min_target_prediction = target_critic_prediction
                    else:
                        min_target_prediction = torch.min(min_target_prediction, target_critic_prediction)
                q_target = min_target_prediction - self.alpha * target_action_logp
            else:
                # TODO: add other types
                pass
        return q_target

    def _update_critic(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        actor_next_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
    ) -> List[torch.Tensor]:
        with torch.no_grad():
            # target_action, target_action_logp = self._sample_action(actor_next_obs)
            # target_critic_input = self._critic_input(critic_next_obs, target_action)
            # target_critic_prediction_1 = self.target_critic_1.forward(target_critic_input)
            # target_critic_prediction_2 = self.target_critic_2.forward(target_critic_input)

            # target_next = (
            #     torch.min(target_critic_prediction_1, target_critic_prediction_2) - self.alpha * target_action_logp
            # )
            target_next = self._get_sampled_q_target_no_grad(actor_next_obs, critic_next_obs)
            target = rewards + self._discount_factor * (1 - dones) * target_next

        critic_input = self._critic_input(critic_obs, actions)
        critic_loss = []
        for critic, critic_optimizer in zip(self.critics, self.critic_optimizers):
            critic_prediction = critic.forward(critic_input)
            loss = nn.functional.mse_loss(critic_prediction, target)

            critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self._gradient_clip)
            critic_optimizer.step()
            critic_loss.append(loss)
        # critic_prediction_1 = self.critic_1.forward(critic_input)
        # critic_1_loss = nn.functional.mse_loss(critic_prediction_1, target)

        # self.critic_1_optimizer.zero_grad()
        # critic_1_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_1.parameters(), self._gradient_clip)
        # self.critic_1_optimizer.step()

        # critic_prediction_2 = self.critic_2.forward(critic_input)
        # critic_2_loss = nn.functional.mse_loss(critic_prediction_2, target)

        # self.critic_2_optimizer.zero_grad()
        # critic_2_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_2.parameters(), self._gradient_clip)
        # self.critic_2_optimizer.step()

        return critic_loss
