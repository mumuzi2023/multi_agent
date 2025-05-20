# 整合后的单个Python文件
# MARL with IQL (CNN) - Pursuit Environment - Single File

# %% md
# MARL with IQL (CNN) - Pursuit Environment
# %% md
# Independent Q-Learning adapted for the PettingZoo SISL Pursuit environment.
# This version uses a CNN for processing image-like observations from Pursuit.
# All necessary class definitions and the main script are in this single file.
# %%
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim  # Explicitly import optim
import numpy as np
import random
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, asdict, field
from typing import Callable, Union, Tuple, Optional, List, Dict  # Added List, Dict for clarity
from itertools import count
from pprint import pprint
DEVICE='cpu'
# Environment and Gym spaces
from pettingzoo.sisl import pursuit_v4
import gymnasium as gym

# Plotting imports
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


# ==============================================================================
# 1. DQN Network Definitions (dqn.py content)
# ==============================================================================
def _init_weights(layer: nn.Module, nonlinearity: str) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DQN_MLP(nn.Module):
    def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128):
        super(DQN_MLP, self).__init__()
        self.obs_dim = n_obs
        self.act_dim = n_act
        self.hidden_dim = n_hidden
        self.model = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.model:
            _init_weights(layer, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class DQN_CNN(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_act: int,
                 n_hidden_fc: int = 256):  # n_hidden_fc default changed
        super(DQN_CNN, self).__init__()
        self.obs_shape = obs_shape  # Expects (C, H, W)
        self.n_act = n_act

        c, h, w = obs_shape

        # Example CNN architecture - tune based on obs_shape (H, W)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1 if h >= 5 and w >= 5 else 0),
            # Adjust padding based on size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if h > 4 and w > 4 else nn.Identity(),  # H/2, W/2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1 if h >= 5 and w >= 5 else 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if h > 8 and w > 8 else nn.Identity()  # H/4, W/4
        )

        dummy_input = torch.zeros(1, c, h, w)
        conv_out_dim = self._get_conv_out_dim(dummy_input)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_dim, n_hidden_fc),
            nn.ReLU(),
            nn.Linear(n_hidden_fc, n_act)
        )
        self._reset_parameters()

    def _get_conv_out_dim(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = self.conv_layers(x)
        return int(np.prod(x.shape[1:]))

    def _reset_parameters(self) -> None:
        for block in [self.conv_layers, self.fc_layers]:
            for layer_or_module in block:
                if isinstance(layer_or_module, (nn.Linear, nn.Conv2d)):  # Only init these types
                    _init_weights(layer_or_module, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape == self.obs_shape:  # (C,H,W)
            x = x.unsqueeze(0)  # Add batch dim: (1, C, H, W)
        elif x.ndim == 2 and np.prod(x.shape[1:]) == np.prod(self.obs_shape):  # (N, C*H*W) - flattened image
            x = x.reshape(x.size(0), *self.obs_shape)  # Reshape to (N, C, H, W)
        # else: assume x is already (N, C, H, W)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)


# ==============================================================================
# 2. Replay Memory (memory.py content)
# ==============================================================================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, *args: Union[torch.Tensor, None, int]) -> None:  # Added type hints
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # type: ignore
        self.memory[self.position] = Transition(*args)  # type: ignore
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.memory) < batch_size:
            return []
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# ==============================================================================
# 3. Agent Definitions (agent.py content)
# ==============================================================================
@dataclass
class DqnAgentConfig:
    obs_dim: Optional[int] = None  # For MLP: flattened observation dim
    obs_shape: Optional[Tuple[int, ...]] = None  # For CNN: (C, H, W)
    act_dim: Optional[int] = None
    hidden_dim: int = 128  # For MLP hidden layers, or FC part of CNN
    use_cnn: bool = False  # Flag to use CNN
    batch_size: int = 128
    lr: float = 1e-4
    grad_clip_value: float = 100
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_decay: float = 0.95  # Original user value, may need tuning
    eps_min: float = 0.01
    mem_size: int = 10_000

    def validate(self) -> None:
        if self.use_cnn:
            assert self.obs_shape is not None, "obs_shape must be set for CNN mode."
        else:
            assert self.obs_dim is not None, "obs_dim must be set for MLP mode."
        assert self.act_dim is not None, "act_dim must be set."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CqlAgentConfig(DqnAgentConfig):  # Kept from user's original agent.py
    obs_dims: Optional[OrderedDict[str, int]] = None  # For MLP-based CQL on joint obs
    act_dims: Optional[OrderedDict[str, int]] = None

    # obs_shapes and use_cnn inherited. Using use_cnn=True with CqlAgent
    # would require significant changes to infer_joint_space and get_masked_joint_obs for CNNs.

    def validate(self) -> None:  # Overrides DqnAgentConfig.validate
        # This validate is for CqlAgent's specific multi-agent observation/action dicts
        if self.use_cnn:
            # If CqlAgent were to use CNNs for individual agent observations before combining,
            # this would check self.obs_shapes (a new field for dict of shapes)
            print("Warning: CqlAgentConfig with use_cnn=True requires careful review of joint space for CNNs.")
            # For now, let DqnAgentConfig.validate handle basic checks if super() is called
            # or if obs_dim/act_dim are set by infer_joint_space
        else:
            assert self.obs_dims is not None, "obs_dims must be set for CqlAgentConfig (MLP mode)."
        assert self.act_dims is not None, "act_dims must be set for CqlAgentConfig."
        # If obs_dim is set (e.g. by infer_joint_space), DqnAgentConfig's validation logic for obs_dim can pass.
        # super().validate() # This would call DqnAgentConfig.validate()

    def infer_joint_space(self) -> CqlAgentConfig:
        if self.use_cnn:
            # TODO: Proper joint observation space inference for CNNs is complex.
            # This might involve concatenating CNN feature outputs, not raw obs_shapes.
            # For now, if use_cnn is True, this method will likely produce an obs_dim
            # that is not directly usable by a single joint CNN without further design.
            # It's better to assume CqlAgent with use_cnn=True is for future extension.
            # Defaulting to MLP-like sum of flattened features if obs_dims is used.
            if self.obs_dims:  # If legacy obs_dims (flattened) are provided
                self.obs_dim = sum(self.obs_dims.values())
                print("Warning: CqlAgentConfig use_cnn=True but inferring joint obs_dim from obs_dims (MLP style).")
            else:  # This path is problematic for CNNs
                self.obs_dim = self.hidden_dim  # Placeholder, needs real logic
                print(f"Warning: CqlAgentConfig use_cnn=True, joint obs_dim set to placeholder {self.obs_dim}.")

        elif self.obs_dims is not None:
            self.obs_dim = sum(self.obs_dims.values())
        else:
            # This case should be prevented by validate or prior logic if obs_dim is needed.
            # If DqnAgentConfig.validate() is called, self.obs_dim needs to be not None.
            pass  # obs_dim might be set directly on the config

        if self.act_dims is None: raise ValueError("act_dims cannot be None for infer_joint_space")
        self.act_dim = 1
        for cur_act_dim in self.act_dims.values():
            self.act_dim *= cur_act_dim
        return self


class DqnAgent:
    def __init__(self, sid: str, config: Union[DqnAgentConfig, CqlAgentConfig], act_sampler: Callable,
                 device: Optional[torch.device] = None):
        self.sid = sid
        self.config = config

        if isinstance(config, CqlAgentConfig) and config.obs_dim is None:
            if config.use_cnn and config.obs_shapes:  # CQL + CNN (complex, placeholder)
                config.infer_joint_space()  # Will use placeholder logic for obs_dim
            elif not config.use_cnn and config.obs_dims:  # CQL + MLP
                config.infer_joint_space()

        self.config.validate()  # Validate after potential inference

        self.act_sampler = act_sampler
        self.replay_memory = ReplayMemory(capacity=self.config.mem_size)
        self.eps = self.config.eps_start
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        if self.config.use_cnn:
            assert self.config.obs_shape is not None, "obs_shape required for CNN in DqnAgent"
            self.policy_net = DQN_CNN(obs_shape=self.config.obs_shape, n_act=self.config.act_dim,
                                      n_hidden_fc=self.config.hidden_dim).to(self.device)
            self.target_net = DQN_CNN(obs_shape=self.config.obs_shape, n_act=self.config.act_dim,
                                      n_hidden_fc=self.config.hidden_dim).to(self.device)
        else:
            assert self.config.obs_dim is not None, "obs_dim required for MLP in DqnAgent"
            self.policy_net = DQN_MLP(n_obs=self.config.obs_dim, n_act=self.config.act_dim,
                                      n_hidden=self.config.hidden_dim).to(self.device)
            self.target_net = DQN_MLP(n_obs=self.config.obs_dim, n_act=self.config.act_dim,
                                      n_hidden=self.config.hidden_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.opt = optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)  # Use optim
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: torch.Tensor, **kwargs):  # type: ignore
        return self._select_action_eps(state, dqn=self.policy_net, **kwargs)

    def select_action_greedy(self, state: torch.Tensor, dqn: nn.Module, **kwargs):  # type: ignore
        return self._select_action_eps(state, dqn=dqn, eps=0, **kwargs)

    def _select_action_eps(self, state: torch.Tensor, dqn: nn.Module, eps: float = -1, **kwargs):  # type: ignore
        # Input state for CNN should be (1, C, H, W) or (C,H,W)
        # Input state for MLP should be (1, obs_dim) or (obs_dim)
        if self.config.use_cnn:
            if state.ndim == 3 and state.shape == self.config.obs_shape: state = state.unsqueeze(0)  # Add batch for CNN
        else:  # MLP
            if state.ndim == 1: state = state.unsqueeze(0)  # Add batch for MLP

        if eps == -1: eps = self.eps
        if random.random() < eps:
            return torch.tensor([[self.act_sampler()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = dqn(state)
                return q_values.argmax(dim=1).reshape(1, 1)

    def train(self) -> None:
        if len(self.replay_memory) < self.config.batch_size: return
        transitions = self.replay_memory.sample(self.config.batch_size)
        if not transitions: return
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            non_final_nxt_states = torch.cat(non_final_next_states_list)
        else:
            empty_shape = (0,) + self.config.obs_shape if self.config.use_cnn and self.config.obs_shape else (
            0, self.config.obs_dim if self.config.obs_dim else 0)
            if empty_shape[
                -1] == 0 and not self.config.use_cnn and self.config.obs_dim is None:  # Safety for unconfigured obs_dim for MLP
                raise ValueError("obs_dim is None for MLP mode in train when creating empty tensor.")
            non_final_nxt_states = torch.empty(empty_shape, device=self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_best_q_values = torch.zeros(self.config.batch_size, device=self.device)
        if non_final_next_states_list:
            with torch.no_grad():
                next_state_best_q_values[non_final_mask] = self.target_net(non_final_nxt_states).max(1).values

        expected_state_action_q_values = reward_batch + (
                    self.config.gamma * next_state_best_q_values.reshape(self.config.batch_size, 1))
        loss = self.criterion(state_action_q_values, expected_state_action_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_clip_value)
        self.opt.step()

    def memorize(self, *args: Union[torch.Tensor, None, int]) -> None:
        self.replay_memory.push(*args)  # type: ignore

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_eps(self) -> None:
        self.eps = max(self.config.eps_min, self.eps * self.config.eps_decay)


class CqlAgent(DqnAgent):  # User's original CqlAgent, with minimal changes for compatibility
    def __init__(self, sid: str, config: CqlAgentConfig, act_sampler: Callable, device: Optional[torch.device] = None):
        # CqlAgent typically works with joint observations/actions for an MLP-based central controller.
        # If use_cnn=True is passed in CqlAgentConfig, it implies individual agents might use CNNs,
        # and their feature outputs would then be combined for a central MLP.
        # The current infer_joint_space is primarily for MLP.
        if config.obs_dim is None and config.obs_dims is not None:  # If joint obs_dim not set but individual are
            config.infer_joint_space()  # This sets config.obs_dim for the DqnAgent base.
            # It also sets config.act_dim (joint action).

        # If CqlAgent directly uses a CNN (config.use_cnn=True), then config.obs_shape should be set.
        # But its methods below (decode/encode joint action) assume a single Q-network
        # outputting joint actions, typically an MLP fed with joint observations.
        super().__init__(sid, config, act_sampler,
                         device)  # DqnAgent.__init__ will use config.obs_dim or config.obs_shape

    def agent_keys(self) -> list[str]:  # type: ignore
        if self.config.act_dims is None: return []  # type: ignore [attr-defined]
        return list(self.config.act_dims.keys())  # type: ignore [attr-defined]

    def n_agents(self) -> int:
        return len(self.agent_keys())

    def get_masked_joint_obs(self, observations: Union[torch.Tensor, Dict[str, np.ndarray]],
                             done_agents: OrderedDict[str, bool]) -> torch.Tensor:  # type: ignore
        # This method is for creating a joint observation vector, typically for an MLP.
        # If individual agents used CNNs, 'observations' might be feature vectors.
        if isinstance(observations, torch.Tensor): return observations

        if self.config.obs_dims is None or done_agents is None:  # type: ignore [attr-defined]
            raise ValueError("obs_dims and done_agents must be provided for get_masked_joint_obs")

        # Assumes self.config is CqlAgentConfig
        obs_config_ref: OrderedDict[str, int] = self.config.obs_dims  # type: ignore

        joint_obs_list = []
        for cur_agent, cur_done in done_agents.items():
            if cur_agent not in obs_config_ref: continue  # Only process agents in config
            agent_obs_dim = obs_config_ref[cur_agent]
            if cur_done:
                joint_obs_list.append(torch.zeros(agent_obs_dim, device=self.device))
            else:
                # Ensure observations[cur_agent] is correctly preprocessed if it's raw
                # For MLP, this expects a 1D array/tensor.
                obs_data = observations[cur_agent]
                if isinstance(obs_data, np.ndarray):  # If raw numpy array
                    # Assuming it's already flattened and normalized if needed before this stage
                    # or that obs_config_ref[cur_agent] IS the flattened dim.
                    obs_data = torch.tensor(obs_data, dtype=torch.float32, device=self.device)

                if obs_data.shape[0] != agent_obs_dim:  # Check if it matches expected dim
                    obs_data = obs_data.flatten()  # Attempt to flatten
                if obs_data.shape[0] != agent_obs_dim:
                    raise ValueError(f"Obs dim mismatch for {cur_agent}: {obs_data.shape[0]} vs {agent_obs_dim}")
                joint_obs_list.append(obs_data)

        return torch.cat(joint_obs_list).reshape(1, -1)  # type: ignore

    def decode_joint_action(self, joint_action: int) -> Dict[str, Optional[int]]:
        actions: List[int] = []
        if self.config.act_dims is None: raise ValueError(
            "act_dims not set in CqlAgentConfig")  # type: ignore [attr-defined]

        act_config_ref: OrderedDict[str, int] = self.config.act_dims  # type: ignore
        for cur_act_dim in reversed(list(act_config_ref.values())):  # Ensure iteration
            actions.append(joint_action % cur_act_dim)
            joint_action //= cur_act_dim
        actions = list(reversed(actions))
        agent_keys_list = self.agent_keys()
        res = {agent_keys_list[a_i]: actions[a_i] for a_i in range(len(agent_keys_list))}
        return res

    def encode_joint_action(self, actions: Dict[str, Optional[int]]) -> int:
        res = 0
        multiplier = 1
        if self.config.act_dims is None: raise ValueError(
            "act_dims not set in CqlAgentConfig")  # type: ignore [attr-defined]

        act_config_ref: OrderedDict[str, int] = self.config.act_dims  # type: ignore
        for agent_key, cur_act_dim in reversed(list(act_config_ref.items())):  # Ensure iteration
            cur_action = actions.get(agent_key)  # Use .get for safety
            if cur_action is None:  # Handle None action
                cur_action = random.randint(0, cur_act_dim - 1)
            res += cur_action * multiplier
            multiplier *= cur_act_dim
        return res

    def get_masked_actions(self, joint_action: int, done_agents: Optional[OrderedDict[str, bool]] = None) -> Dict[
        str, Optional[int]]:
        decoded = self.decode_joint_action(joint_action)
        if done_agents:
            for agent_key_original in self.agent_keys():  # Iterate over known agent keys
                if done_agents.get(agent_key_original, False):  # Check if agent is done
                    decoded[agent_key_original] = None
        return decoded

    # _select_action_eps for CqlAgent needs to output a dict of actions if act_sampler returns a dict
    def _select_action_eps(self, state: torch.Tensor, dqn: nn.Module, eps: float = -1,
                           done_agents: Optional[OrderedDict[str, bool]] = None):  # type: ignore
        # state for CqlAgent is typically joint observation, fed to an MLP
        if state.ndim == 1: state = state.unsqueeze(0)
        if eps == -1: eps = self.eps

        if random.random() < eps:
            # act_sampler for CqlAgent should return a dictionary of actions for each sub-agent
            return self.act_sampler()
        else:
            with torch.no_grad():
                q_values = dqn(state)  # q_values for joint actions
                sel_joint_action = q_values.argmax(dim=1).item()
                return self.get_masked_actions(sel_joint_action, done_agents)

    # train method for CqlAgent (from user's original code)
    def train(self) -> None:
        if len(self.replay_memory) < self.config.batch_size: return
        transitions = self.replay_memory.sample(self.config.batch_size)
        if not transitions: return

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)  # Joint states
        # Actions in batch are expected to be dicts for CqlAgent
        action_batch = torch.tensor(
            [[self.encode_joint_action(cur_actions)] for cur_actions in batch.action],  # type: ignore
            device=self.device, dtype=torch.long
        )
        reward_batch = torch.cat(batch.reward)  # Joint rewards typically

        state_action_q_values = self.policy_net(state_batch).gather(1, action_batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            non_final_nxt_states = torch.cat(non_final_next_states_list)
        else:
            # CqlAgent's config.obs_dim is the joint observation dimension
            if self.config.obs_dim is None:  # Should be set by infer_joint_space
                raise ValueError(
                    "CqlAgent: config.obs_dim is None during train, infer_joint_space likely not called or failed.")
            non_final_nxt_states = torch.empty((0, self.config.obs_dim), device=self.device)

        next_state_best_q_values = torch.zeros(self.config.batch_size, device=self.device)
        if non_final_next_states_list:
            with torch.no_grad():
                next_state_best_q_values[non_final_mask] = self.target_net(non_final_nxt_states).max(1).values

        expected_state_action_q_values = reward_batch + (
                    self.config.gamma * next_state_best_q_values.reshape(self.config.batch_size, 1))
        loss = self.criterion(state_action_q_values, expected_state_action_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config.grad_clip_value)
        self.opt.step()


# ==============================================================================
# 4. MAIN SCRIPT LOGIC
# ==============================================================================

# Determine device (already defined globally, but good practice if this section was separate)
# DEVICE = torch.device(...) (using global DEVICE)

# %% md
### Environment Setup
# %%
MAX_CYCLES_ENV = 500
N_EVADERS_ENV = 5
N_PURSUERS_ENV = 4
OBS_RANGE_ENV = 7
N_CATCH_ENV = 2

env = pursuit_v4.parallel_env(
    max_cycles=MAX_CYCLES_ENV,
    n_evaders=N_EVADERS_ENV,
    n_pursuers=N_PURSUERS_ENV,
    obs_range=OBS_RANGE_ENV,
    n_catch=N_CATCH_ENV,
    render_mode='rgb_array'
)
env.reset()

ALL_AGENT_IDS_FROM_ENV = env.agents
CONTROLLED_AGENT_KEYS = sorted([agent_id for agent_id in ALL_AGENT_IDS_FROM_ENV if "pursuer" in agent_id])

if not CONTROLLED_AGENT_KEYS:
    raise ValueError("No pursuers found to control.")
num_controlled_agents = len(CONTROLLED_AGENT_KEYS)
print(f"Controlling {num_controlled_agents} pursuer(s): {CONTROLLED_AGENT_KEYS}")

obs_shapes_dict: Dict[str, Tuple[int, int, int]] = {}  # Stores (C, H, W)
act_dims_dict: Dict[str, int] = {}

for agent_key in CONTROLLED_AGENT_KEYS:
    obs_space = env.observation_space(agent_key)
    act_space = env.action_space(agent_key)
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 3:
        raise ValueError(f"Expected Box obs space (H,W,C) for {agent_key}, got {obs_space}")
    h, w, c = obs_space.shape
    obs_shapes_dict[agent_key] = (c, h, w)  # CHW
    act_dims_dict[agent_key] = int(act_space.n)

print(f"Observation Shapes (C, H, W) for controlled agents: {obs_shapes_dict}")
print(f"Action Dimensions for controlled agents: {act_dims_dict}")

# %% md
### Agent Initialization
# %%
USE_CNN_FLAG = True  # Switch to True to use CNN

AGENT_CONFIGS_DICT: Dict[str, DqnAgentConfig] = {}
for agent_key in CONTROLLED_AGENT_KEYS:
    AGENT_CONFIGS_DICT[agent_key] = DqnAgentConfig(
        obs_shape=obs_shapes_dict[agent_key] if USE_CNN_FLAG else None,
        obs_dim=int(np.prod(obs_shapes_dict[agent_key])) if not USE_CNN_FLAG else None,
        act_dim=act_dims_dict[agent_key],
        hidden_dim=256,  # FC hidden units for CNN, or MLP hidden units
        use_cnn=USE_CNN_FLAG,
        batch_size=32,  # Smaller batch for potentially larger CNN data
        lr=1e-4,  # Common LR for CNNs
        grad_clip_value=1.0,
        gamma=0.99,
        eps_start=1.0,  # Start with full exploration
        eps_decay=0.999,  # Slower decay for complex image tasks
        eps_min=0.05,
        mem_size=50000  # Larger memory for image data
    )

current_agents_dict: Dict[str, DqnAgent] = {
    agent_key: DqnAgent(
        sid=agent_key,
        config=AGENT_CONFIGS_DICT[agent_key],
        act_sampler=env.action_space(agent_key).sample,
        device=DEVICE
    )
    for agent_key in CONTROLLED_AGENT_KEYS
}

if current_agents_dict:
    print(f"Created {len(current_agents_dict)} DqnAgent(s). Network type: {'CNN' if USE_CNN_FLAG else 'MLP'}")
    print("First agent's policy network:", list(current_agents_dict.values())[0].policy_net)
else:
    print("No agents created.")


# %% md
### Helper Function for State Preprocessing (for CNN)
# %%
def preprocess_cnn_observation(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    if obs_hwc is None: raise ValueError("Received None observation.")
    obs_normalized = np.array(obs_hwc, dtype=np.float32) / 255.0
    obs_tensor_chw = torch.tensor(obs_normalized, device=device).permute(2, 0, 1)
    return obs_tensor_chw.unsqueeze(0)  # NCHW


# %% md
### Evaluation Function
# %%
def eval_agents_fn(  # Renamed to avoid conflict if user runs cells out of order
        eval_env_creator: Callable,  # Lambda to create env
        agents_to_eval: Dict[str, DqnAgent],
        networks_to_use_for_eval: Dict[str, nn.Module],
        num_eval_episodes: int = 1,
        eval_render_flag: bool = False
) -> Dict[str, List[float]]:
    if not agents_to_eval: return {}

    ep_rewards_per_agent: Dict[str, List[float]] = {agent_key: [] for agent_key in agents_to_eval.keys()}
    eval_device_internal = list(agents_to_eval.values())[0].device

    for _ in range(num_eval_episodes):
        eval_env_instance = eval_env_creator(render_human=eval_render_flag)
        raw_states_dict_eval, _ = eval_env_instance.reset()

        current_proc_states_eval: Dict[str, torch.Tensor] = {}
        for ag_key in agents_to_eval.keys():
            if ag_key in raw_states_dict_eval:
                # Assuming all agents in agents_to_eval use same config type (CNN/MLP)
                agent_conf_for_eval = agents_to_eval[ag_key].config
                if agent_conf_for_eval.use_cnn:
                    current_proc_states_eval[ag_key] = preprocess_cnn_observation(raw_states_dict_eval[ag_key],
                                                                                  eval_device_internal)
                else:  # MLP with flattened
                    current_proc_states_eval[ag_key] = preprocess_observation_flattened(  # Define this if using MLP
                        raw_states_dict_eval[ag_key], agent_conf_for_eval.obs_dim, eval_device_internal  # type: ignore
                    )

        episode_dones_eval = {agent_key: False for agent_key in agents_to_eval.keys()}
        current_ep_rewards = {agent_key: 0.0 for agent_key in agents_to_eval.keys()}

        for _ in range(MAX_CYCLES_ENV):  # Max steps for safety
            if eval_render_flag and hasattr(eval_env_instance, 'render'):
                eval_env_instance.render()

            actions_to_env_eval = {}
            for ag_key, ag_instance in agents_to_eval.items():
                if ag_key in current_proc_states_eval and not episode_dones_eval[ag_key]:
                    action_t = ag_instance.select_action_greedy(
                        current_proc_states_eval[ag_key],
                        networks_to_use_for_eval[ag_key]
                    )
                    actions_to_env_eval[ag_key] = action_t.item()

            if not actions_to_env_eval: break

            next_raw_obs_eval, rewards_eval, terminations_eval, truncations_eval, _ = eval_env_instance.step(
                actions_to_env_eval)
            next_proc_states_temp_eval: Dict[str, torch.Tensor] = {}

            for ag_key in agents_to_eval.keys():
                if episode_dones_eval[ag_key]:
                    if ag_key in current_proc_states_eval:
                        next_proc_states_temp_eval[ag_key] = current_proc_states_eval[ag_key]
                    continue

                if ag_key in actions_to_env_eval:  # Reward if action was taken
                    current_ep_rewards[ag_key] += rewards_eval.get(ag_key, 0)

                episode_dones_eval[ag_key] = terminations_eval.get(ag_key, False) or truncations_eval.get(ag_key, False)

                if not episode_dones_eval[ag_key] and ag_key in next_raw_obs_eval:
                    agent_conf_for_eval = agents_to_eval[ag_key].config
                    if agent_conf_for_eval.use_cnn:
                        next_proc_states_temp_eval[ag_key] = preprocess_cnn_observation(next_raw_obs_eval[ag_key],
                                                                                        eval_device_internal)
                    else:
                        next_proc_states_temp_eval[ag_key] = preprocess_observation_flattened(
                            next_raw_obs_eval[ag_key], agent_conf_for_eval.obs_dim, eval_device_internal  # type: ignore
                        )
                elif ag_key in current_proc_states_eval:
                    next_proc_states_temp_eval[ag_key] = current_proc_states_eval[ag_key]

            current_proc_states_eval = next_proc_states_temp_eval
            if all(episode_dones_eval.values()): break

        if eval_render_flag and hasattr(eval_env_instance, 'close'):
            eval_env_instance.close()
        for ag_key in agents_to_eval.keys():
            ep_rewards_per_agent[ag_key].append(current_ep_rewards[ag_key])

    return ep_rewards_per_agent


def get_avg_rewards_fn(rewards_dict: Dict[str, List[float]]) -> Dict[str, float]:  # Renamed
    return {
        key: sum(val_list) / len(val_list) if val_list else 0.0
        for key, val_list in rewards_dict.items()
    }


# %% md
### Baseline Evaluation
# %%
EVAL_ENV_CREATOR_LAMBDA = lambda render_human=False: pursuit_v4.parallel_env(
    max_cycles=MAX_CYCLES_ENV, n_evaders=N_EVADERS_ENV, n_pursuers=num_controlled_agents,
    obs_range=OBS_RANGE_ENV, n_catch=N_CATCH_ENV,
    render_mode='human' if render_human else 'rgb_array'
)

baseline_overall_avg_reward = float('-inf')
if current_agents_dict:
    print("Running baseline evaluation...")
    baseline_rewards = eval_agents_fn(
        EVAL_ENV_CREATOR_LAMBDA,
        current_agents_dict,
        networks_to_use_for_eval={agent.sid: agent.target_net for agent in current_agents_dict.values()},
        num_eval_episodes=10
    )
    avg_baseline_agent_rewards = get_avg_rewards_fn(baseline_rewards)
    pprint(f"Avg baseline rewards per agent: {avg_baseline_agent_rewards}")
    if avg_baseline_agent_rewards:
        baseline_overall_avg_reward = sum(avg_baseline_agent_rewards.values()) / len(avg_baseline_agent_rewards)
    print(f"Overall average baseline reward: {baseline_overall_avg_reward}")
else:
    print("No agents for baseline eval.")


# %% md
### Plotting Module
# %%
def plot_training_returns(returns_log: List[float], plot_title_suffix: str = ''):  # Renamed
    if not returns_log: return
    plt.figure(1);
    plt.clf()
    returns_tensor = torch.tensor(returns_log, dtype=torch.float)
    plt.title(f'Training Progress {plot_title_suffix}');
    plt.xlabel('Episode');
    plt.ylabel('Overall Avg. Return')
    plt.plot(returns_tensor.numpy())
    if len(returns_tensor) >= 10:
        rolling_means = returns_tensor.unfold(0, 10, 1).mean(1).view(-1)
        padding = torch.full((min(9, len(returns_tensor) - 1),), float('nan')) if len(
            returns_tensor) > 0 else torch.tensor([])
        if padding.numel() > 0 or rolling_means.numel() > 0:
            padded_means = torch.cat((padding, rolling_means))
            if padded_means.numel() > 0 and not torch.all(torch.isnan(padded_means)):
                plt.plot(padded_means.numpy(), label="10-ep Avg")
                plt.legend()
    plt.pause(0.001)
    if is_ipython: display.display(plt.gcf()); display.clear_output(wait=True)


# %% md
### Training Setup
# %%
NUM_TRAIN_EPISODES = 500
MAX_STEPS_PER_TRAIN_EPISODE = MAX_CYCLES_ENV
TARGET_NET_UPDATE_FREQ_GLOBAL_STEPS = 200

# Main training environment instance (re-init for safety, though lambda is better for loops)
main_training_env = EVAL_ENV_CREATOR_LAMBDA()


# %% md
### Target Network Update Logic
# %%
def update_targets_if_better_fn(  # Renamed
        agents_for_update: Dict[str, DqnAgent],
        env_creator_fn: Callable,
        best_reward_so_far: float,
        eval_eps: int = 5
) -> float:
    if not agents_for_update: return best_reward_so_far
    print(f"Eval policy nets for target update (current best: {best_reward_so_far:.2f})...")
    with torch.no_grad():
        policy_rewards = eval_agents_fn(
            env_creator_fn,
            agents_for_update,
            networks_to_use_for_eval={agent.sid: agent.policy_net for agent in agents_for_update.values()},
            num_eval_episodes=eval_eps
        )
    avg_policy_agent_rewards = get_avg_rewards_fn(policy_rewards)
    current_overall_avg_reward = sum(avg_policy_agent_rewards.values()) / len(
        avg_policy_agent_rewards) if avg_policy_agent_rewards else float('-inf')

    print(f"Policy net eval: {current_overall_avg_reward:.2f}")
    if current_overall_avg_reward > best_reward_so_far:
        print(f"Improvement! {current_overall_avg_reward:.2f} > {best_reward_so_far:.2f}. Updating targets.")
        best_reward_so_far = current_overall_avg_reward
        for agent_instance in agents_for_update.values(): agent_instance.update_target_network()
    else:
        print("No improvement, targets not updated.")
    return best_reward_so_far


# %% md
### Training Loop
# %%
if not current_agents_dict:
    print("No agents for training.")
else:
    global_training_steps = 0
    current_best_overall_reward = float(baseline_overall_avg_reward)  # Start from baseline
    training_returns_log: List[float] = [current_best_overall_reward] if current_best_overall_reward > float(
        '-inf') else []

    print(f"Starting training for {NUM_TRAIN_EPISODES} episodes...")
    for i_ep in range(NUM_TRAIN_EPISODES):
        raw_ep_states_dict, _ = main_training_env.reset()

        # Preprocess initial states
        processed_ep_states: Dict[str, torch.Tensor] = {}
        for ag_k_ctrl in CONTROLLED_AGENT_KEYS:
            if ag_k_ctrl in raw_ep_states_dict:
                agent_conf = current_agents_dict[ag_k_ctrl].config
                if agent_conf.use_cnn:
                    processed_ep_states[ag_k_ctrl] = preprocess_cnn_observation(raw_ep_states_dict[ag_k_ctrl], DEVICE)
                else:  # MLP
                    processed_ep_states[ag_k_ctrl] = preprocess_observation_flattened(  # Need this if MLP is an option
                        raw_ep_states_dict[ag_k_ctrl], agent_conf.obs_dim, DEVICE  # type: ignore
                    )

        ep_dones_status = {ag_k_ctrl: False for ag_k_ctrl in CONTROLLED_AGENT_KEYS}
        ep_total_reward_for_controlled_agents = 0.0

        for t_s in range(MAX_STEPS_PER_TRAIN_EPISODE):
            actions_to_env_training: Dict[str, int] = {}
            action_tensors_for_mem: Dict[str, torch.Tensor] = {}

            for ag_k_ctrl in CONTROLLED_AGENT_KEYS:
                if ag_k_ctrl in processed_ep_states and not ep_dones_status[ag_k_ctrl]:
                    ag_instance = current_agents_dict[ag_k_ctrl]
                    action_t = ag_instance.select_action(processed_ep_states[ag_k_ctrl])
                    actions_to_env_training[ag_k_ctrl] = action_t.item()
                    action_tensors_for_mem[ag_k_ctrl] = action_t

            if not actions_to_env_training: break

            next_raw_obs_tr, rewards_tr, terminations_tr, truncations_tr, _ = main_training_env.step(
                actions_to_env_training)
            next_proc_states_temp_tr: Dict[str, torch.Tensor] = {}

            for ag_k_ctrl in CONTROLLED_AGENT_KEYS:
                ag_instance = current_agents_dict[ag_k_ctrl]
                reward_val_tr = rewards_tr.get(ag_k_ctrl, 0.0)

                if ag_k_ctrl in actions_to_env_training:
                    ep_total_reward_for_controlled_agents += reward_val_tr

                if ep_dones_status[ag_k_ctrl]:  # Already done
                    if ag_k_ctrl in action_tensors_for_mem:  # Memorize last action if taken
                        ag_instance.memorize(
                            processed_ep_states[ag_k_ctrl], action_tensors_for_mem[ag_k_ctrl], None,
                            torch.tensor([[reward_val_tr]], device=DEVICE, dtype=torch.float32)
                        )
                    if ag_k_ctrl in processed_ep_states:
                        next_proc_states_temp_tr[ag_k_ctrl] = processed_ep_states[ag_k_ctrl]
                    continue

                is_done_now = terminations_tr.get(ag_k_ctrl, False) or truncations_tr.get(ag_k_ctrl, False)

                next_state_mem = None
                if not is_done_now and ag_k_ctrl in next_raw_obs_tr:
                    agent_c = ag_instance.config
                    if agent_c.use_cnn:
                        proc_next_obs = preprocess_cnn_observation(next_raw_obs_tr[ag_k_ctrl], DEVICE)
                    else:
                        proc_next_obs = preprocess_observation_flattened(next_raw_obs_tr[ag_k_ctrl], agent_c.obs_dim,
                                                                         DEVICE)  # type: ignore
                    next_proc_states_temp_tr[ag_k_ctrl] = proc_next_obs
                    next_state_mem = proc_next_obs
                elif ag_k_ctrl in processed_ep_states:
                    next_proc_states_temp_tr[ag_k_ctrl] = processed_ep_states[ag_k_ctrl]

                if ag_k_ctrl in action_tensors_for_mem:
                    ag_instance.memorize(
                        processed_ep_states[ag_k_ctrl], action_tensors_for_mem[ag_k_ctrl], next_state_mem,
                        torch.tensor([[reward_val_tr]], device=DEVICE, dtype=torch.float32)
                    )

                ag_instance.train()
                ep_dones_status[ag_k_ctrl] = is_done_now

            processed_ep_states = next_proc_states_temp_tr

            if global_training_steps > 0 and global_training_steps % TARGET_NET_UPDATE_FREQ_GLOBAL_STEPS == 0:
                current_best_overall_reward = update_targets_if_better_fn(
                    current_agents_dict, EVAL_ENV_CREATOR_LAMBDA, current_best_overall_reward
                )

            for ag_val in current_agents_dict.values(): ag_val.update_eps()
            global_training_steps += 1

            if all(ep_dones_status.values()): break

        current_best_overall_reward = update_targets_if_better_fn(
            current_agents_dict, EVAL_ENV_CREATOR_LAMBDA, current_best_overall_reward
        )
        avg_ep_return_overall = ep_total_reward_for_controlled_agents / num_controlled_agents if num_controlled_agents > 0 else 0.0
        training_returns_log.append(avg_ep_return_overall)
        plot_training_returns(training_returns_log)

        first_agent_eps_val = list(current_agents_dict.values())[0].eps if current_agents_dict else 0.0
        print(
            f"Ep {i_ep + 1}/{NUM_TRAIN_EPISODES}. Avg Return: {avg_ep_return_overall:.2f}. Eps: {first_agent_eps_val:.3f}. Steps: {global_training_steps}")

    print("Training finished.")
    if training_returns_log:
        plot_training_returns(training_returns_log, plot_title_suffix=" - Final")
    plt.ioff();
    plt.show()

# %% md
### Post-Training Evaluation
# %%
if current_agents_dict:
    print("Running post-training evaluation...")
    final_eval_rewards = eval_agents_fn(
        EVAL_ENV_CREATOR_LAMBDA,
        current_agents_dict,
        networks_to_use_for_eval={agent.sid: agent.target_net for agent in current_agents_dict.values()},
        num_eval_episodes=20
    )
    avg_final_agent_rewards = get_avg_rewards_fn(final_eval_rewards)
    pprint(f"Avg post-training rewards per agent: {avg_final_agent_rewards}")
    overall_avg_final_reward = sum(avg_final_agent_rewards.values()) / len(
        avg_final_agent_rewards) if avg_final_agent_rewards else 0.0
    print(f"Overall average post-training reward: {overall_avg_final_reward}")
else:
    print("No agents for post-training eval.")

# %% md
### Rendered Evaluation (Optional)
# %%
# if current_agents_dict:
#     print("Running final rendered evaluation...")
#     eval_agents_fn(
#         EVAL_ENV_CREATOR_LAMBDA,
#         current_agents_dict,
#         networks_to_use_for_eval={agent.sid: agent.target_net for agent in current_agents_dict.values()},
#         num_eval_episodes=1,
#         eval_render_flag=True
#     )
#     print("Rendered evaluation finished.")

# %%
# Close the main training environment instance if it's still open
if 'main_training_env' in locals() and hasattr(main_training_env, 'close'):
    main_training_env.close()