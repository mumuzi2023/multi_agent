import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, n_features, n_actions, n_hidden1=50, n_hidden2=50):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden1)
        self.fc_hidden = nn.Linear(n_hidden1, n_hidden2)
        self.fc2 = nn.Linear(n_hidden2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_hidden(x))
        x = self.fc2(x)
        return x


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,  # Initial epsilon for choosing greedy action
            replace_target_iter=100,  # Paper uses 10000, but for smaller problems less is fine
            memory_size=5000,  # Increased from original
            batch_size=32,
            e_greedy_increment=None,  # If None, epsilon is fixed. If a value, epsilon is annealed.
            device='cpu'
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.device = device

        self.learn_step_counter = 0
        self.memory_counter = 0

        # Initialize memory [s, a, r, s', done]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))  # s, s' (n_features each), a, r, done (1 each)

        self.eval_net = QNetwork(n_features, n_actions).to(self.device)
        self.target_net = QNetwork(n_features, n_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.cost_his = []

    def store_transition(self, s, a, r, s_prime, done):
        if self.n_features == 0:  # Should not happen if initialized correctly
            print("Warning: n_features is 0. Cannot store transition.")
            return

        transition = np.hstack((s, [a, r], s_prime, [1 if done else 0]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, action_mask, execution=False):
        current_epsilon = 1.0 if execution else self.epsilon  # Full exploitation if execution=True

        if np.random.uniform() < current_epsilon:
            # Exploit: choose the best action based on Q-values
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                actions_value = self.eval_net(observation_tensor)

            # Apply action mask: set Q-values of invalid actions to a very low number
            masked_actions_value = actions_value.clone()  # ทำงานบน CPU ก่อน
            for i in range(self.n_actions):
                if action_mask[i] == 0:  # Invalid action
                    masked_actions_value[0, i] = -float('inf')
            action = torch.argmax(masked_actions_value).item()
        else:
            # Explore: choose a random valid action
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:  # Should not happen in pursuit if game is not over
                action = np.random.randint(0, self.n_actions)  # Fallback, though not ideal
        return action

    def _update_target_net(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print(f'Target network updated at step {self.learn_step_counter}')

    def learn(self):
        if self.memory_counter < self.batch_size:
            return  # Not enough samples in memory yet

        self._update_target_net()

        # Sample a batch of transitions
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]

        # Convert to PyTorch tensors
        # s: columns 0 to n_features-1
        # a: column n_features
        # r: column n_features+1
        # s': columns n_features+2 to 2*n_features+1
        # done: column 2*n_features+2

        states = torch.FloatTensor(batch_memory[:, :self.n_features]).to(self.device)
        actions = torch.LongTensor(batch_memory[:, self.n_features].astype(int)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch_memory[:, self.n_features + 1]).to(self.device)
        next_states = torch.FloatTensor(batch_memory[:, self.n_features + 2: self.n_features * 2 + 2]).to(self.device)
        dones = torch.FloatTensor(batch_memory[:, self.n_features * 2 + 2]).to(self.device)

        # Get Q-values for current states and chosen actions
        q_eval = self.eval_net(states).gather(1, actions).squeeze(1)

        # Get max Q-values for next states from target network
        q_next_all = self.target_net(next_states).detach()  # detach: no gradients for target network
        q_next_max = q_next_all.max(1)[0]

        # Compute target Q-values
        q_target = rewards + self.gamma * q_next_max * (1 - dones)  # (1-dones) ensures Q_target=reward if done

        # Compute loss
        loss = F.mse_loss(q_eval, q_target)
        self.cost_his.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (exploration rate)
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)

        self.learn_step_counter += 1

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)
        print(f"Model saved in path: {path}")

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.eval_net.state_dict())  # Sync target net
        self.eval_net.eval()  # Set to evaluation mode if only for inference
        self.target_net.eval()
        print(f"Model restored from path: {path}")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.title('DQN Training Cost')
        plt.show()