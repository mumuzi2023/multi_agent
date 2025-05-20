import random
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Callable, Union
import torch

# import torch # already imported
# import torch.nn as nn # already imported
# from memory import ReplayMemory, Transition # Defined above
# from dqn import DQN # Defined above

@dataclass
class DqnAgentConfig:
    # env info
    obs_dim: int = None
    act_dim: int = None
    hidden_dim: int = 128
    # training
    batch_size: int = 128
    lr: float = 1e-4
    grad_clip_value: float = 100
    # gamma: discount factor
    gamma: float = 0.99
    # epsilon: exploration probability
    eps_start: float = 0.9
    eps_decay: float = 0.995  # Pursuit might need slower decay
    eps_min: float = 0.05
    # replay memory
    mem_size: int = 10_000

    def validate(self) -> None:
        assert (self.obs_dim is not None)
        assert (self.act_dim is not None)

    def to_dict(self) -> dict:
        return asdict(self)


# CqlAgentConfig and CqlAgent are omitted as they are not used for IQL

class DqnAgent:
    def __init__(self, sid: str, config: DqnAgentConfig, act_sampler: Callable, device=None):
        self.sid = sid
        self.config = config
        self.config.validate()
        self.act_sampler = act_sampler  # action sampler function
        self.replay_memory = ReplayMemory(capacity=self.config.mem_size)
        self.eps = self.config.eps_start
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else  # mps check
            'cpu'
        )
        # DQN
        self.policy_net = DQN(n_obs=self.config.obs_dim, n_act=self.config.act_dim, n_hidden=self.config.hidden_dim).to(
            self.device)
        self.target_net = DQN(n_obs=self.config.obs_dim, n_act=self.config.act_dim, n_hidden=self.config.hidden_dim).to(
            self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # opt & loss criterion
        self.opt = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: torch.Tensor, **kwargs):  # state is already a tensor
        return self._select_action_eps(state, dqn=self.policy_net, **kwargs)

    def select_action_greedy(self, state: torch.Tensor, dqn: nn.Module, **kwargs):  # state is already a tensor
        return self._select_action_eps(state, dqn=dqn, eps=0, **kwargs)

    def _select_action_eps(self, state: torch.Tensor, dqn: nn.Module, eps: float = -1, **kwargs):
        """
        input shape: 1 x obs_dim
        output shape: 1 x 1 (tensor containing the action index)
        """
        if eps == -1:
            eps = self.eps
        if random.random() < eps:
            sample_res = torch.tensor([[self.act_sampler()]], device=self.device, dtype=torch.long)
            return sample_res
        else:
            with torch.no_grad():
                q_values: torch.Tensor = dqn(state)  # state is already 1 x obs_dim
                sel_res = q_values.argmax(dim=1).reshape(1, 1)
                return sel_res

    def train(self) -> None:  # Returns loss value or None
        if len(self.replay_memory) < self.config.batch_size:
            return None  # Indicating no training was done

        transitions = self.replay_memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        # Handle cases where all next_states might be None
        non_final_nxt_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_nxt_states_list) > 0:
            non_final_nxt_states = torch.cat(non_final_nxt_states_list)
        else:  # Create an empty tensor with correct shape if all next states are None
            non_final_nxt_states = torch.empty((0, self.config.obs_dim), device=self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values_batch: torch.Tensor = self.policy_net(state_batch)
        state_action_q_values = q_values_batch.gather(1, action_batch)

        next_state_best_q_values = torch.zeros(self.config.batch_size, device=self.device)
        if non_final_nxt_states.nelement() > 0:  # Check if tensor is not empty
            with torch.no_grad():
                next_state_best_q_values[non_final_mask] = self.target_net(non_final_nxt_states).max(1).values

        expected_state_action_q_values = reward_batch + (
                    self.config.gamma * next_state_best_q_values.reshape(self.config.batch_size, 1))

        loss = self.criterion(state_action_q_values, expected_state_action_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_clip_value)
        self.opt.step()
        return loss.item()

    def memorize(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor | None,
                 reward: torch.Tensor) -> None:
        # Assumes inputs are already correctly shaped tensors
        self.replay_memory.push(state, action, next_state, reward)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_eps(self):
        self.eps = max(self.config.eps_min, self.eps * self.config.eps_decay)