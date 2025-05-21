import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, n_features, n_actions,
                 n_hidden1=50,
                 n_hidden2=50):
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
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
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

        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        self.eval_net = QNetwork(n_features, n_actions).to(self.device)
        self.target_net = QNetwork(n_features, n_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.cost_his = []

    def store_transition(self, s, a, r, s_prime, done):
        transition = np.hstack((s, [a, r], s_prime, [1 if done else 0]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, action_mask, execution=False):
        current_epsilon = 1.0 if execution else self.epsilon
        if np.random.uniform() < current_epsilon:
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                actions_value = self.eval_net(observation_tensor)
            masked_actions_value = actions_value.clone()
            for i in range(self.n_actions):
                if action_mask[i] == 0:
                    masked_actions_value[0, i] = -float('inf')
            action = torch.argmax(masked_actions_value).item()
        else:
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = np.random.randint(0, self.n_actions)
        return action

    def _update_target_net(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self._update_target_net()

        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]

        # S_joint
        states = torch.FloatTensor(batch_memory[:, :self.n_features]).to(self.device)
        # agent i's action a_i
        actions_from_data = torch.LongTensor(batch_memory[:, self.n_features].astype(int)).unsqueeze(1).to(self.device)
        # agent i's award r_i
        rewards = torch.FloatTensor(batch_memory[:, self.n_features + 1]).to(self.device)
        next_states = torch.FloatTensor(batch_memory[:, self.n_features + 2: self.n_features * 2 + 2]).to(self.device)
        dones = torch.FloatTensor(batch_memory[:, self.n_features * 2 + 2]).to(self.device)

        #Bellman Loss
        q_eval_all_actions = self.eval_net(states)
        q_eval_data_actions = q_eval_all_actions.gather(1, actions_from_data).squeeze(1)

        with torch.no_grad():
            q_next_all_actions_target = self.target_net(next_states)
            q_next_max_target = q_next_all_actions_target.max(1)[0]
            q_target_bellman = rewards + self.gamma * q_next_max_target * (1 - dones)

        bellman_loss = F.mse_loss(q_eval_data_actions, q_target_bellman)

        self.cost_his.append(bellman_loss.item())

        self.optimizer.zero_grad()
        bellman_loss.backward()
        self.optimizer.step()

        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)

        self.learn_step_counter += 1

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)
        print(f"Model saved in path: {path}")

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.eval_net.eval()
        self.target_net.eval()
        print(f"Model restored from path: {path}")