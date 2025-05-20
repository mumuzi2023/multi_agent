import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pettingzoo.sisl import pursuit_v4
import random
from collections import deque, namedtuple


# --- Q-Network Definition ---
class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        # The observation in pursuit_v4 is often a 3D array (channels, height, width)
        # We might need a CNN or flatten it for an MLP
        # Assuming obs_size is already flattened or a 1D representation
        # If obs is image-like (e.g., (3, 7, 7)), use nn.Conv2d layers first
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # If x is a 3D observation, flatten it first: x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- IQL Agent ---
class IQLAgent:
    def __init__(self, agent_id, obs_space_shape, action_space_n, lr=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update_freq=10):
        self.agent_id = agent_id
        self.action_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Assuming observation is a 3D grid (channels, height, width)
        # Flatten it for the MLP, or use a CNN
        obs_size = np.prod(obs_space_shape)  # Flatten

        self.policy_net = QNetwork(obs_size, action_space_n)
        self.target_net = QNetwork(obs_size, action_space_n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.train_step_counter = 0

    def select_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_n - 1)
        else:
            with torch.no_grad():
                # Preprocess observation (e.g., flatten, to tensor)
                obs_tensor = torch.FloatTensor(observation.flatten()).unsqueeze(0)
                q_values = self.policy_net(obs_tensor)
                return q_values.max(1)[1].item()  # Get action with max Q-value

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Process batch data
        state_batch = torch.FloatTensor(np.array([s.flatten() for s in batch.state]))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)

        non_final_next_states_list = [s_next.flatten() for s_next, done in zip(batch.next_state, batch.done) if
                                      not done]
        non_final_mask = torch.tensor([not d for d in batch.done], dtype=torch.bool)

        next_state_values = torch.zeros(self.batch_size)
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.FloatTensor(np.array(non_final_next_states_list))
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Optional gradient clipping
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.train_step_counter += 1


# --- Main Training Loop ---
if __name__ == "__main__":
    # Environment setup
    env_config = {
        "max_cycles": 500,
        "x_size": 16,
        "y_size": 16,
        "n_evaders": 5,  # Reduced for simplicity in example
        "n_pursuers": 2,  # Reduced for simplicity
        "obs_range": 7,
        "shared_reward": True,  # IQL often works better with individual rewards if possible,
        # but shared_reward is a common setting in MARL.
        # If False, you'd need to ensure rewards dict is handled per agent.
        "n_catch": 2,  # Number of pursuers needed to catch an evader
    }
    env = pursuit_v4.parallel_env(render_mode="human" , **env_config)  # "human" for rendering
    observations, infos = env.reset()

    num_agents = len(env.possible_agents)
    agents_ids = env.possible_agents
    action_spaces = {agent_id: env.action_space(agent_id) for agent_id in agents_ids}
    observation_spaces = {agent_id: env.observation_space(agent_id) for agent_id in agents_ids}

    # Initialize IQL agents
    # Note: observation_spaces[agent_id].shape is crucial.
    # For pursuit, it's typically (obs_range, obs_range, 3).
    # PettingZoo often returns Box spaces with shape like (H, W, C).
    # PyTorch Conv2d expects (C, H, W). You'll need to handle this transposition if using CNNs.
    # For MLP, ensure flattening is consistent.
    # The raw observation from PettingZoo might need np.transpose(obs, (2,0,1)) if using CNN.

    iql_agents = {
        agent_id: IQLAgent(
            agent_id,
            observation_spaces[agent_id].shape,  # e.g., (7, 7, 3)
            action_spaces[agent_id].n,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.999,
            buffer_size=50000,
            batch_size=128,
            target_update_freq=20
        )
        for agent_id in agents_ids
    }

    num_episodes = 1000
    max_steps_per_episode = env_config["max_cycles"]

    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in agents_ids}
        terminated = {agent_id: False for agent_id in agents_ids}
        truncated = {agent_id: False for agent_id in agents_ids}

        current_observations = observations

        for step in range(max_steps_per_episode):
            if not env.agents:  # All agents are done
                break

            actions_to_env = {}
            current_actions_dict = {}  # To store actions chosen by agents for learning

            for agent_id in env.agents:  # Iterate through active agents
                obs_for_agent = current_observations[agent_id]
                action = iql_agents[agent_id].select_action(obs_for_agent)
                actions_to_env[agent_id] = action
                current_actions_dict[agent_id] = action  # Store for learning

            next_observations, rewards, terminations, truncations, infos = env.step(actions_to_env)

            # Store experience for each agent that acted
            for agent_id in current_observations.keys():  # Use keys from observations before step
                if agent_id in actions_to_env:  # If agent took an action this step
                    obs = current_observations[agent_id]
                    action = current_actions_dict[agent_id]
                    rew = rewards.get(agent_id, 0)  # Handle if agent is done and not in rewards
                    next_obs = next_observations.get(agent_id)
                    done = terminations.get(agent_id, False) or truncations.get(agent_id, False)

                    episode_rewards[agent_id] += rew

                    if next_obs is not None:  # Only store if next_obs exists
                        iql_agents[agent_id].memory.push(obs, action, next_obs, rew, done)

                    # Perform learning step for each agent
                    iql_agents[agent_id].learn()

            current_observations = next_observations

            # Check if all agents are done
            if not env.agents or all(terminations.values()) or all(truncations.values()):
                break

        total_episode_reward = sum(episode_rewards.values())
        avg_epsilon = np.mean([iql_agents[aid].epsilon for aid in iql_agents])
        print(f"Episode {episode + 1}: Total Reward: {total_episode_reward:.2f}, Avg Epsilon: {avg_epsilon:.3f}")

    env.close()
    print("Training finished.")