# memory.py (as it would be structured)
from __future__ import annotations  # Required if type hinting Transition in older Python
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# Your DQN.py content
import torch
import torch.nn as nn


def _init_weights(layer: nn.Module, nonlinearity: str) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DQN(nn.Module):
    def __init__(self, n_obs: int, n_act: int, n_hidden: int = 128):
        super(DQN, self).__init__()
        self.obs_dim = n_obs
        self.act_dim = n_act
        self.hidden_dim = n_hidden
        # model
        self.model = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # re-initialize learnable parameters
        for layer_idx, layer in enumerate(self.model):  # Iterate with index to get nonlinearity for previous layer
            if isinstance(layer, nn.Linear):
                nonlinearity = 'relu'  # Default for hidden layers
                if layer_idx == len(self.model) - 1:  # Last linear layer
                    nonlinearity = 'linear'  # Or some other appropriate for output Q-values if not followed by ReLU
                nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            # Note: The original _init_weights only accepted 'relu'.
            # Kaiming normal usually wants the nonlinearity of the *current* layer if it's like Conv2d->ReLU.
            # For Linear->ReLU->Linear->ReLU->Linear, the 'relu' makes sense for the first two.
            # For the last linear layer, 'linear' or 'leaky_relu' (if that was used) might be more standard for kaiming_normal's `nonlinearity` param.
            # However, using 'relu' for all as in the original simplified _reset_parameters is also a common approach.
            # I'll stick to a slightly more robust _reset_parameters matching the original intent for `_init_weights`.
        for model_layer in self.model:  # Simpler loop, assumes 'relu' for init if Linear
            _init_weights(model_layer, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.model(x)
        return res


# Your agent.py content (DqnAgent and DqnAgentConfig parts)
# from __future__ import annotations # Already at the top of memory.py
import random
from collections import OrderedDict  # Not strictly needed for DqnAgent here, but CqlAgent used it
from dataclasses import dataclass, asdict
from typing import Callable, Union


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
        assert (self.obs_dim is not None), "obs_dim must be set in DqnAgentConfig"
        assert (self.act_dim is not None), "act_dim must be set in DqnAgentConfig"

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

    def train(self) -> float | None:  # Returns loss value or None
        if len(self.replay_memory) < self.config.batch_size:
            return None  # Indicating no training was done

        transitions = self.replay_memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        non_final_nxt_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_nxt_states_list) > 0:
            non_final_nxt_states = torch.cat(non_final_nxt_states_list)
        else:
            non_final_nxt_states = torch.empty((0, self.config.obs_dim), device=self.device, dtype=torch.float32)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values_batch: torch.Tensor = self.policy_net(state_batch)
        state_action_q_values = q_values_batch.gather(1, action_batch)

        next_state_best_q_values = torch.zeros(self.config.batch_size, device=self.device)
        if non_final_nxt_states.nelement() > 0:
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


# %%
# Main script for IQL on Pursuit
import gymnasium  # PettingZoo uses Gymnasium spaces
import numpy as np
# import torch # already imported
# import torch.nn as nn # already imported
# import torch.optim as optim # DqnAgent handles this
# import random # already imported
# from collections import deque, namedtuple # ReplayMemory handles this
import matplotlib
import matplotlib.pyplot as plt
# from IPython import display # Handled by is_ipython check
from itertools import count
from pprint import pprint
import time  # For render delay

from pettingzoo.sisl import pursuit_v4


# --- Helper function to preprocess observations ---
def preprocess_observation(obs_numpy: np.ndarray, device: torch.device) -> torch.Tensor:
    """Flattens and converts numpy observation to a tensor of shape (1, obs_dim)."""
    flat_obs = obs_numpy.flatten().astype(np.float32)
    return torch.tensor(flat_obs, device=device).unsqueeze(0)


# %%
# Environment and Agent Configuration
MAX_CYCLES = 200
N_PURSERS = 8
N_EVADERS = 4  # Keep it simple for IQL demo
ENV_RENDER_MODE = None  # "human" or None

env = pursuit_v4.parallel_env(
    n_pursuers=N_PURSERS,
    n_evaders=N_EVADERS,
    max_cycles=MAX_CYCLES,
    # obs_type='grid', # <--- REMOVED THIS ARGUMENT
    render_mode=ENV_RENDER_MODE
)
init_obs_dict, init_infos = env.reset(seed=42)  # Use a seed for reproducibility

pursuer_ids = [agent_id for agent_id in env.agents if "pursuer" in agent_id]
evader_ids = [agent_id for agent_id in env.agents if "evader" in agent_id]
print(f"Pursuers: {pursuer_ids}")
print(f"Evaders: {evader_ids}")

# Determine obs and action dimensions for pursuers
sample_pursuer_id = pursuer_ids[0]
raw_obs_space_shape = env.observation_space(sample_pursuer_id).shape
obs_dim_pursuer = np.prod(raw_obs_space_shape)  # Flattened
act_dim_pursuer = env.action_space(sample_pursuer_id).n
print(f"Pursuer Obs Dim (flat): {obs_dim_pursuer}")
print(f"Pursuer Act Dim: {act_dim_pursuer}")

# Create DQN agents for pursuers
# Use a common device for all agents and tensors
training_device = torch.device('cuda' if torch.cuda.is_available() else \
                                   'mps' if hasattr(torch.backends,
                                                    'mps') and torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {training_device}")

cur_agents_pursuers = {
    agent_id: DqnAgent(
        sid=agent_id,
        config=DqnAgentConfig(
            obs_dim=obs_dim_pursuer,
            act_dim=act_dim_pursuer,
            hidden_dim=128,  # Can be tuned
            batch_size=64,  # Can be tuned
            lr=1e-4,  # Can be tuned
            gamma=0.99,
            eps_start=1.0,
            eps_decay=0.999,  # Slower decay for more exploration
            eps_min=0.05,  # Minimum exploration
            mem_size=30000,  # Memory size
            grad_clip_value=1.0  # Clip gradients to prevent explosion
        ),
        act_sampler=lambda ag_id=agent_id: env.action_space(ag_id).sample(),  # Lambda to ensure correct agent_id scope
        device=training_device
    )
    for agent_id in pursuer_ids
}
print(f"Initialized {len(cur_agents_pursuers)} pursuer agents.")


# %% md
# ## Evaluation Function (Adapted for new DqnAgent)
# %%
def eval_pursuit_agents_new(
        dqn_pursuer_agents: dict[str, DqnAgent],
        dqns_to_use_type: str,  # "policy" or "target"
        n_episodes: int = 1,
        render: bool = False,
        eval_device: torch.device = torch.device('cpu')  # Device for evaluation
) -> dict[str, list[float]]:
    eval_env_params = dict(
        n_pursuers=N_PURSERS,
        n_evaders=N_EVADERS,
        max_cycles=MAX_CYCLES,
        # obs_type='grid', # <--- REMOVED THIS ARGUMENT
        render_mode=('human' if render else None)
    )
    eval_env = pursuit_v4.parallel_env(**eval_env_params)

    cumulative_rewards_pursuers = {pursuer_id: [] for pursuer_id in dqn_pursuer_agents.keys()}

    for i in range(n_episodes):
        states_dict_numpy, infos_dict = eval_env.reset(seed=42 + i)
        current_episode_rewards_pursuers = {pursuer_id: 0.0 for pursuer_id in dqn_pursuer_agents.keys()}

        # active_agents_in_env = set(eval_env.agents) # PettingZoo's env.agents updates dynamically

        for t_step in count():
            if render and eval_env.render_mode == "human":
                eval_env.render()
                time.sleep(0.05)

            actions_to_env = {}
            # Pursuers select actions greedily
            for pursuer_id, agent_obj in dqn_pursuer_agents.items():
                if pursuer_id in eval_env.agents and pursuer_id in states_dict_numpy:
                    state_tensor = preprocess_observation(states_dict_numpy[pursuer_id], eval_device)
                    network = agent_obj.policy_net if dqns_to_use_type == "policy" else agent_obj.target_net
                    action_tensor = agent_obj.select_action_greedy(state_tensor, network)
                    actions_to_env[pursuer_id] = action_tensor.item()

            # Evaders take random actions
            for evader_id in evader_ids:  # Assuming evader_ids are fixed for the scenario
                if evader_id in eval_env.agents and evader_id in states_dict_numpy:
                    actions_to_env[evader_id] = eval_env.action_space(evader_id).sample()

            filtered_actions_to_env = {agent_id: act for agent_id, act in actions_to_env.items() if
                                       agent_id in eval_env.agents}

            if not eval_env.agents:
                break
            # If filtered_actions_to_env is empty but env.agents is not, it means no actions were generated for live agents.
            # This could happen if e.g. all pursuers are done but evaders remain.
            # PettingZoo step expects actions only for live agents if you pass a filtered dict.
            # If you pass actions for all original agents, it internally filters.
            # For simplicity, we build actions for agents we know are live and active this turn.
            if not filtered_actions_to_env and eval_env.agents:
                # print(f"Warning: No actions in filtered_actions_to_env but agents {eval_env.agents} are live.")
                # This can happen if all learning agents (pursuers) are done. Let PettingZoo handle it if needed.
                # Or, ensure evaders always have an action if they are live.
                for ev_id_check in evader_ids:  # Ensure evaders get random if they are the only ones left
                    if ev_id_check in eval_env.agents and ev_id_check not in filtered_actions_to_env and ev_id_check in states_dict_numpy:
                        filtered_actions_to_env[ev_id_check] = eval_env.action_space(ev_id_check).sample()

            next_states_dict_numpy, rewards_dict, terminations_dict, truncations_dict, infos_dict = eval_env.step(
                actions_to_env)  # Use original actions_to_env; PZ filters

            for pursuer_id in dqn_pursuer_agents.keys():
                if pursuer_id in rewards_dict:
                    current_episode_rewards_pursuers[pursuer_id] += rewards_dict[pursuer_id]

            if not eval_env.agents or t_step >= MAX_CYCLES - 1:  # Check if all agents are done or max cycles
                break
            states_dict_numpy = next_states_dict_numpy

        for pursuer_id in dqn_pursuer_agents.keys():
            cumulative_rewards_pursuers[pursuer_id].append(current_episode_rewards_pursuers[pursuer_id])

        if render and eval_env.render_mode == "human": eval_env.render()

    eval_env.close()
    return cumulative_rewards_pursuers


# get_agent_wise_cumulative_rewards (same as before)
def get_agent_wise_cumulative_rewards(cumulative_rewards: dict[str, list[float]]) -> dict[str, float]:
    return {
        agent_key: sum(agent_episode_rewards) / len(agent_episode_rewards) if agent_episode_rewards else 0.0
        for agent_key, agent_episode_rewards in cumulative_rewards.items()
    }


# %% md
# Baseline Evaluation
# %%
print("Evaluating untrained agents (using policy_net, which is same as target_net initially)...")
baseline_eval_res = eval_pursuit_agents_new(cur_agents_pursuers, dqns_to_use_type="policy", n_episodes=5,
                                            eval_device=training_device)
avg_baseline_res = get_agent_wise_cumulative_rewards(baseline_eval_res)
print("> Average cumulative rewards (baseline) per pursuer:")
pprint(avg_baseline_res)
all_avg_baseline_res = sum(avg_baseline_res.values()) / len(avg_baseline_res) if avg_baseline_res and len(
    avg_baseline_res) > 0 else -float('inf')
print(f"> Overall average cumulative reward (baseline): {all_avg_baseline_res:.2f}")

# %% md
# ## Training
# %%
# Plotting
is_ipython = 'inline' in matplotlib.get_backend() if ('matplotlib' in globals() and 'plt' in globals()) else False
if is_ipython: from IPython import display


def plot_episodes_rewards(episode_mean_rewards_all_agents: list[float], title='Training... Pursuer Avg. Return'):
    if not plt: return  # Matplotlib not available
    plt.figure(1)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Return (Pursuers)')
    rewards_t = torch.tensor(episode_mean_rewards_all_agents, dtype=torch.float)
    plt.plot(rewards_t.numpy(), label='Episode Avg Return')


    if len(rewards_t) >= 10:
        means = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
        padding_size = min(9, len(rewards_t) - 1) if len(rewards_t) > 1 else 0
        padding = torch.full((padding_size,), rewards_t[0] if len(rewards_t) > 0 else 0.0)

        plot_means = torch.cat((padding, means))
        if len(plot_means) == len(rewards_t):
            plt.plot(plot_means.numpy(), label='10-ep Mov Avg')
        elif len(means) > 0:
            plt.plot(range(len(rewards_t) - len(means), len(rewards_t)), means.numpy(), label='10-ep Mov Avg (part)')

    plt.legend()
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        plt.draw()
        plt.show(block=False)


N_TRAIN_EPISODES = 100
TARGET_NET_UPDATE_STRATEGY_FREQ = 20  # Episodes


def update_all_target_networks_if_better_new(
        current_pursuer_agents: dict[str, DqnAgent],
        current_best_mean_reward: float,
        eval_episodes: int = 5,
        device_for_eval: torch.device = torch.device('cpu')
) -> float:
    print("Evaluating current policy networks to consider target update...")
    eval_res = eval_pursuit_agents_new(current_pursuer_agents, dqns_to_use_type="policy", n_episodes=eval_episodes,
                                       eval_device=device_for_eval)
    avg_eval_res_per_agent = get_agent_wise_cumulative_rewards(eval_res)
    if not avg_eval_res_per_agent: return current_best_mean_reward  # No results

    current_overall_mean_reward = sum(avg_eval_res_per_agent.values()) / len(avg_eval_res_per_agent)
    print(
        f"  Evaluation of policy_nets: Overall mean reward = {current_overall_mean_reward:.3f} (vs best: {current_best_mean_reward:.3f})")
    if current_overall_mean_reward > current_best_mean_reward:
        print(f"  New best performance! Updating target networks for all pursuers.")
        for agent in current_pursuer_agents.values():
            agent.update_target_network()
        return current_overall_mean_reward
    else:
        print(f"  No improvement. Target networks remain unchanged.")
        return current_best_mean_reward


# %%
# Main Training Loop
print(f"\nStarting training for {N_TRAIN_EPISODES} episodes...")
current_best_overall_reward = all_avg_baseline_res if all_avg_baseline_res != -float('inf') else -float('inf')
episode_mean_rewards_history = [all_avg_baseline_res] if all_avg_baseline_res != -float('inf') else []

for i_episode in range(N_TRAIN_EPISODES):
    states_dict_numpy, infos_dict = env.reset(seed=1000 + i_episode)

    episode_total_loss_pursuers = {p_id: 0.0 for p_id in pursuer_ids}
    episode_steps_with_loss_pursuers = {p_id: 0 for p_id in pursuer_ids}  # Track steps where loss was computed

    for t_step in count():
        actions_to_env = {}
        current_pursuer_actions_for_memory = {}

        active_pursuers_this_step = [pid for pid in pursuer_ids if pid in env.agents and pid in states_dict_numpy]
        for pursuer_id in active_pursuers_this_step:
            agent = cur_agents_pursuers[pursuer_id]
            state_tensor = preprocess_observation(states_dict_numpy[pursuer_id], training_device)
            action_tensor = agent.select_action(state_tensor)
            actions_to_env[pursuer_id] = action_tensor.item()
            current_pursuer_actions_for_memory[pursuer_id] = action_tensor

        active_evaders_this_step = [eid for eid in evader_ids if eid in env.agents and eid in states_dict_numpy]
        for evader_id in active_evaders_this_step:
            actions_to_env[evader_id] = env.action_space(evader_id).sample()

        if not env.agents: break  # All agents are done from PZ perspective

        next_states_dict_numpy, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(actions_to_env)

        for pursuer_id in active_pursuers_this_step:
            agent = cur_agents_pursuers[pursuer_id]
            reward_val = rewards_dict.get(pursuer_id, 0.0)
            reward_tensor = torch.tensor([[reward_val]], device=training_device, dtype=torch.float32)

            current_state_tensor = preprocess_observation(states_dict_numpy[pursuer_id], training_device)
            action_taken_tensor = current_pursuer_actions_for_memory[pursuer_id]

            next_state_tensor_for_memory = None
            agent_terminated = terminations_dict.get(pursuer_id, False)
            agent_truncated = truncations_dict.get(pursuer_id, False)

            if not (agent_terminated or agent_truncated):
                if pursuer_id in next_states_dict_numpy:
                    next_state_tensor_for_memory = preprocess_observation(next_states_dict_numpy[pursuer_id],
                                                                          training_device)

            agent.memorize(current_state_tensor, action_taken_tensor, next_state_tensor_for_memory, reward_tensor)

            loss = agent.train()
            if loss is not None:
                episode_total_loss_pursuers[pursuer_id] += loss
                episode_steps_with_loss_pursuers[pursuer_id] += 1

        if not env.agents or t_step >= MAX_CYCLES - 1:  # PZ: env.agents is empty if all done
            break
        states_dict_numpy = next_states_dict_numpy

    for pursuer_id in pursuer_ids:
        cur_agents_pursuers[pursuer_id].update_eps()

    avg_loss_strings = []
    for p_id in pursuer_ids:
        if episode_steps_with_loss_pursuers[p_id] > 0:
            avg_loss = episode_total_loss_pursuers[p_id] / episode_steps_with_loss_pursuers[p_id]
            avg_loss_strings.append(f"{p_id.split('_')[-1]}: {avg_loss:.3f}")
        else:
            avg_loss_strings.append(f"{p_id.split('_')[-1]}: N/A")
    avg_episode_loss_str = ", ".join(avg_loss_strings)

    print(
        f"Ep {i_episode + 1}/{N_TRAIN_EPISODES} ({t_step + 1} steps). Avg Loss: [{avg_episode_loss_str}] Eps: {cur_agents_pursuers[pursuer_ids[0]].eps:.3f}")

    if (i_episode + 1) % TARGET_NET_UPDATE_STRATEGY_FREQ == 0:
        current_best_overall_reward = update_all_target_networks_if_better_new(
            cur_agents_pursuers, current_best_overall_reward, eval_episodes=3, device_for_eval=training_device)

    if (i_episode + 1) % 10 == 0 or i_episode == N_TRAIN_EPISODES - 1:
        if N_TRAIN_EPISODES > 1:
            print(f"Plotting: Evaluating policy_nets for episode {i_episode + 1} performance...")
            current_perf_res = eval_pursuit_agents_new(cur_agents_pursuers, dqns_to_use_type="policy", n_episodes=3,
                                                       eval_device=training_device)
            avg_current_perf_per_agent = get_agent_wise_cumulative_rewards(current_perf_res)
            if avg_current_perf_per_agent:
                current_overall_perf = sum(avg_current_perf_per_agent.values()) / len(avg_current_perf_per_agent)
                episode_mean_rewards_history.append(current_overall_perf)
                print(f"Plotting: Current policy_net overall performance: {current_overall_perf:.3f}")
                if 'plt' in globals() and 'display' in globals() and is_ipython: plot_episodes_rewards(
                    episode_mean_rewards_history)
            else:
                print("Plotting: No performance data to plot.")

if 'plt' in globals() and 'display' in globals() and is_ipython and episode_mean_rewards_history:
    plt.ioff()
    plot_episodes_rewards(episode_mean_rewards_history)
    plt.show()
elif 'plt' in globals() and plt:  # Check if plt was imported and figure exists
    try:
        if plt.get_fignums():  # Check if any figures are open
            plt.close('all')
    except Exception:  # Catch any errors if backend doesn't support get_fignums well
        pass

print("Training completed.")

# %% md
# ## Final Evaluation
# %%
print("\nEvaluating trained agents (using final target_net)...")
final_eval_res = eval_pursuit_agents_new(cur_agents_pursuers, dqns_to_use_type="target", n_episodes=10,
                                         eval_device=training_device)
avg_final_res = get_agent_wise_cumulative_rewards(final_eval_res)
if avg_final_res:
    print("> Average cumulative rewards (trained) per pursuer:")
    pprint(avg_final_res)
    all_avg_final_res = sum(avg_final_res.values()) / len(avg_final_res)
    print(f"> Overall average cumulative reward (trained): {all_avg_final_res:.2f}")
else:
    print("> No final evaluation results.")

# %% md
# ## Render a few episodes with trained agents
# %%
# Ensure ENV_RENDER_MODE was None during training to avoid issues with multiple envs trying to render to same display
# For rendering, it's cleaner to explicitly set render_mode="human" in the eval call.
if ENV_RENDER_MODE is None:
    print("\nRendering a few episodes with trained agents (target_net)...")
    eval_pursuit_agents_new(cur_agents_pursuers, dqns_to_use_type="target", n_episodes=3, render=True,
                            eval_device=training_device)

env.close()  # Close the main training environment instance
print("Demo finished.")