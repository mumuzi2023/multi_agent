import gymnasium as gym
from pettingzoo.sisl import pursuit_v4
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import pickle  # 用于保存/加载Q表 (可选)
import time
import warnings  # 用于处理一些PettingZoo可能发出的警告

# --- 主要可调整超参数 ---

# --- 环境与评估设置 ---
MAX_CYCLES_TRAIN = 100  # 训练时每回合最大步数。较小的值可以加快训练迭代速度，但可能导致智能体无法完成复杂任务。
# 如果任务需要更多时间探索或完成，应适当增大。
MAX_CYCLES_EVAL = 70  # 评估时每回合最大步数。通常应设置为足以让智能体展示其学到策略的长度。
NUM_EPISODES_TRAIN = 1000  # 总训练回合数。对于复杂问题或期望获得更好性能，需要显著增加此值。
# 注意：对于表格型方法，过多的回合数配合巨大的状态空间会导致内存问题。
EVAL_INTERVAL = 10  # 每训练 EVAL_INTERVAL 个回合后进行一次评估。用于观察学习进度。
NUM_EVAL_EPISODES = 10  # 每次评估时运行的回合数。用于获得更稳定的平均奖励。
NUM_SEEDS = 1  # 用于多次运行实验并平均结果的随机种子数量。
# 理想情况下应为 3-10 以获得更可靠的性能曲线，但会增加总训练时间。
# 为快速演示，这里设为1。

# --- 算法特定的环境参数 ---
# IQL (Independent Q-Learning) 设置
N_PURSUERS_IQL = 30  # IQL训练时的追捕者数量。可以适当多一些，因为IQL的复杂度随智能体数量线性增长（近似）。
N_EVADERS_IQL = 8  # IQL训练时的逃跑者数量。

# CQL (Centralized Q-Learning) 设置
N_PURSUERS_CQL = 2  # CQL训练时的追捕者数量。对于表格型CQL，此值【必须】保持很小（通常2，最多3）。
# 因为联合动作空间随追捕者数量指数级增长 ($M^{N_{pursuers}}$)。
N_EVADERS_CQL = 1  # CQL训练时的逃跑者数量。

# --- Q学习算法参数 ---
ALPHA = 0.5  # 学习率 (Learning Rate)。控制每次经验更新Q值的幅度。
# 典型值范围：0.01 - 0.5。太大会导致学习不稳定，太小则学习缓慢。
GAMMA = 0.99  # 折扣因子 (Discount Factor)。衡量未来奖励的重要性。
# 典型值范围：0.9 - 0.999。越接近1，智能体越重视长期回报。
EPSILON_START = 1.0  # 探索率 (Epsilon) 的初始值。开始时完全随机探索。
EPSILON_END = 0.05  # 探索率的最终值。在训练后期仍保持少量探索。
# 典型值范围：0.01 - 0.1。
# 探索率衰减设置：目标是在训练的大约80%时达到EPSILON_END
# 您也可以选择固定的衰减率，例如 EPSILON_DECAY_FACTOR = 0.999 (用于每步衰减) 或 0.99 (用于每回合衰减)
# 下面的计算方式是动态的，也可以替换为固定的衰减因子。
# 如果 NUM_EPISODES_TRAIN 为0，则衰减率为0，epsilon保持为 EPSILON_START
EPSILON_DECAY_EPISODES_TARGET_PERCENTAGE = 0.8  # 在训练回合数的这个百分比时，epsilon衰减到EPSILON_END

# IQL的Epsilon衰减是基于【每智能体每步】的，因为每个智能体独立决策和更新
# 估算IQL总决策步数：NUM_EPISODES_TRAIN * (MAX_CYCLES_TRAIN / 2 平均步数) * N_PURSUERS_IQL
# 为简化，我们让IQL的epsilon也按回合衰减，但实际应用中更精细的逐（智能体）步衰减可能更好
# 这里统一为按回合衰减，便于比较和实现。
# 如果要改为每步衰减，需要在训练循环的智能体决策步骤后调用 decay_epsilon()。
# 此处我们将IQL的epsilon衰减也设置为按回合衰减（在每个episode结束时调用decay_epsilon_episode）。
# 如果要改为每（联合）步衰减，请在训练循环的每个step之后调用decay_epsilon_step。
# 为简化，两个算法的epsilon衰减都设计为在回合结束时发生。
_num_decay_episodes = NUM_EPISODES_TRAIN * EPSILON_DECAY_EPISODES_TARGET_PERCENTAGE
if _num_decay_episodes > 0 and EPSILON_START > EPSILON_END:
    # 每回合衰减率 = (最终值 / 初始值)^(1 / 达到最终值所需的回合数)
    EPSILON_DECAY_RATE_PER_EPISODE = (EPSILON_END / EPSILON_START) ** (1 / _num_decay_episodes)
else:
    EPSILON_DECAY_RATE_PER_EPISODE = 1.0  # 不衰减或衰减已完成/无效


# --- Helper Functions ---
def obs_to_state_key(observation):
    """Converts a NumPy observation array or dict of arrays to a hashable key."""
    if isinstance(observation, dict):
        # 对字典中的每个观测值进行处理，并保持顺序（基于agent_id排序）
        # 注意：为了CQL的全局状态一致性，需要保证pursuer_ids的顺序
        # 这里假设CQL的get_global_state_key会处理好顺序
        return tuple(obs.tobytes() for obs in observation.values())
    return observation.tobytes()


def create_pursuit_env(n_pursuers, n_evaders, max_cycles, render_mode=None):
    """Creates a parallel Pursuit environment instance."""
    env = pursuit_v4.parallel_env(
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        max_cycles=max_cycles,
        obs_range=7,
        n_catch=2,
        render_mode=render_mode
    )
    return env


# --- Independent Q-Learning (IQL) ---
class IQLTrainer:
    def __init__(self, n_pursuers, n_evaders, hyperparams):
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.alpha = hyperparams['alpha']
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']  # 保留初始值用于重置
        self.epsilon = self.epsilon_start
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate_per_episode']

        temp_env = create_pursuit_env(n_pursuers, n_evaders, 1)
        self.action_spaces = {
            agent: temp_env.action_space(agent)
            for agent in temp_env.possible_agents if "pursuer" in agent
        }
        temp_env.close()

        if not self.action_spaces:
            raise ValueError("IQL: No pursuer agents found. Check agent naming ('pursuer_X') or environment setup.")

        self.q_tables = {
            agent: defaultdict(lambda: np.zeros(self.action_spaces[agent].n))
            for agent in self.action_spaces.keys()
        }
        self.agent_ids = list(self.action_spaces.keys())

        if len(self.agent_ids) != self.n_pursuers:
            warnings.warn(f"IQL: Expected {self.n_pursuers} pursuers based on N_PURSUERS_IQL, "
                          f"but found {len(self.agent_ids)} pursuers based on 'pursuer' in agent names: {self.agent_ids}. "
                          f"Using {len(self.agent_ids)} pursuers found.")
            self.n_pursuers = len(self.agent_ids)  # 更新实际使用的追捕者数量

    def choose_action(self, agent_id, observation_key, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return self.action_spaces[agent_id].sample()
        else:
            q_values = self.q_tables[agent_id][observation_key]
            # 如果q_values全为0或存在多个最大值，随机选择一个，避免固定偏好
            max_q = np.max(q_values)
            return np.random.choice(np.where(q_values == max_q)[0])

    def update_q_table(self, agent_id, state_key, action, reward, next_state_key, done):
        old_q_value = self.q_tables[agent_id][state_key][action]
        if done or next_state_key is None:  # next_state_key is None if agent is done
            target_q_value = reward
        else:
            next_max_q = np.max(self.q_tables[agent_id][next_state_key]) if next_state_key in self.q_tables[
                agent_id] else 0.0
            target_q_value = reward + self.gamma * next_max_q

        self.q_tables[agent_id][state_key][action] = old_q_value + self.alpha * (target_q_value - old_q_value)

    def decay_epsilon_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)

    def train(self, num_episodes_train, max_cycles_train, eval_interval, num_eval_episodes, max_cycles_eval, seed=None):
        self.epsilon = self.epsilon_start  # 重置epsilon为初始值
        training_rewards_over_time = []
        env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles_train)

        for episode in range(num_episodes_train):
            observations, infos = env.reset(seed=seed + episode if seed is not None else None)

            # 确保只处理当前活跃的追捕者
            current_pursuers_in_episode = [p_id for p_id in self.agent_ids if p_id in observations]
            if not current_pursuers_in_episode:  # 如果没有追捕者了（不太可能在reset时发生，但作为防御）
                # print(f"IQL Episode {episode+1}: No pursuers active at reset. Skipping.")
                if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:  # 保证评估点数量正确
                    training_rewards_over_time.append(0)  # 记录0奖励
                continue

            current_observations_keys = {
                agent: obs_to_state_key(observations[agent])
                for agent in current_pursuers_in_episode
            }

            episode_rewards_sum = 0

            for step_num in range(max_cycles_train):
                if not env.agents or not any("pursuer" in agent for agent in env.agents):  # 如果环境中没有智能体或没有追捕者了
                    break

                actions_to_take = {}
                active_pursuers_in_step = [p_id for p_id in current_pursuers_in_episode if
                                           p_id in env.agents and p_id in current_observations_keys]

                for agent_id in active_pursuers_in_step:
                    obs_key = current_observations_keys[agent_id]
                    actions_to_take[agent_id] = self.choose_action(agent_id, obs_key, explore=True)

                for agent_id in env.agents:  # 其他智能体（如逃跑者）随机行动
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = env.action_space(agent_id).sample()

                if not actions_to_take: break  # 如果没有动作可执行 (例如所有pursuer都已结束)

                next_observations, rewards, terminations, truncations, infos = env.step(actions_to_take)

                step_pursuer_reward = 0
                next_obs_keys_this_step = {}

                for agent_id in active_pursuers_in_step:  # 只为那些采取了行动的pursuer更新
                    if agent_id not in actions_to_take: continue  # 理论上不应发生

                    state_key = current_observations_keys[agent_id]
                    action = actions_to_take[agent_id]
                    reward = rewards.get(agent_id, 0)
                    step_pursuer_reward += reward


                    is_done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    next_state_key_agent = None
                    if agent_id in next_observations:
                        next_state_key_agent = obs_to_state_key(next_observations[agent_id])
                        next_obs_keys_this_step[agent_id] = next_state_key_agent  # 存储下一个状态的key

                    self.update_q_table(agent_id, state_key, action, reward, next_state_key_agent, is_done)

                episode_rewards_sum += step_pursuer_reward
                current_observations_keys = next_obs_keys_this_step  # 更新观察字典
                if step_pursuer_reward > 0:
                    # seed, episode, step_num 都是在此上下文中可用的变量
                    # episode 和 step_num 从0开始计数，所以打印时 +1 更符合通常的理解
                    print(
                        f"IQL - Seed {seed if seed is not None else 'N/A'} - 回合 {episode + 1} - 步数 {step_num + 1}: "
                        f"智能体获得正奖励: {step_pursuer_reward:.2f}")
                # 如果所有pursuer都结束了，或者环境说没有agents了，就结束这一回合
                if not any(p_id in env.agents for p_id in current_pursuers_in_episode) or not env.agents:
                    break

            self.decay_epsilon_episode()

            if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:
                avg_eval_reward = self.evaluate(num_eval_episodes, max_cycles_eval, seed=seed)
                training_rewards_over_time.append(avg_eval_reward)
                print(
                    f"IQL - Seed {seed if seed is not None else 0} - Eval after Episode {episode + 1}: Avg Reward = {avg_eval_reward:.2f}, Eps: {self.epsilon:.3f}")

        env.close()
        return training_rewards_over_time

    def evaluate(self, num_episodes, max_cycles, seed=None):
        total_rewards_eval = 0
        eval_env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles)

        for episode in range(num_episodes):
            observations, _ = eval_env.reset(seed=seed + episode if seed is not None else None)
            current_pursuers_in_eval = [p_id for p_id in self.agent_ids if p_id in observations]
            if not current_pursuers_in_eval: continue

            current_observations_keys_eval = {
                agent: obs_to_state_key(observations[agent])
                for agent in current_pursuers_in_eval
            }
            episode_reward_sum = 0

            for step_num in range(max_cycles):
                if not eval_env.agents or not any("pursuer" in agent for agent in eval_env.agents): break

                actions_to_take = {}
                active_pursuers_in_step = [p_id for p_id in current_pursuers_in_eval if
                                           p_id in eval_env.agents and p_id in current_observations_keys_eval]

                for agent_id in active_pursuers_in_step:
                    obs_key = current_observations_keys_eval[agent_id]
                    actions_to_take[agent_id] = self.choose_action(agent_id, obs_key, explore=False)  # Greedy

                for agent_id in eval_env.agents:
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = eval_env.action_space(agent_id).sample()

                if not actions_to_take: break

                next_observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take)

                step_pursuer_reward = 0
                next_obs_keys_this_step_eval = {}
                for agent_id in active_pursuers_in_step:
                    if agent_id not in rewards: continue  # 如果pursuer已经结束，可能没有它的reward
                    step_pursuer_reward += rewards.get(agent_id, 0)
                    if agent_id in next_observations:
                        next_obs_keys_this_step_eval[agent_id] = obs_to_state_key(next_observations[agent_id])

                episode_reward_sum += step_pursuer_reward
                current_observations_keys_eval = next_obs_keys_this_step_eval

                if not any(p_id in eval_env.agents for p_id in current_pursuers_in_eval) or not eval_env.agents:
                    break
            total_rewards_eval += episode_reward_sum

        eval_env.close()
        return total_rewards_eval / num_episodes if num_episodes > 0 else 0


# --- Centralized Q-Learning (CQL) ---
class CQLTrainer:
    def __init__(self, n_pursuers, n_evaders, hyperparams):
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.alpha = hyperparams['alpha']
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']  # 保留初始值
        self.epsilon = self.epsilon_start
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate_per_episode']

        temp_env = create_pursuit_env(n_pursuers, n_evaders, 1)
        self.pursuer_ids = sorted([agent for agent in temp_env.possible_agents if "pursuer" in agent])  # 排序以保证顺序

        if not self.pursuer_ids:
            raise ValueError("CQL: No pursuer agents found. Check agent naming or environment setup.")
        if len(self.pursuer_ids) != self.n_pursuers:
            warnings.warn(f"CQL: Expected {self.n_pursuers} pursuers based on N_PURSUERS_CQL, "
                          f"but found {len(self.pursuer_ids)} pursuers: {self.pursuer_ids}. "
                          f"Using {len(self.pursuer_ids)} pursuers found.")
            self.n_pursuers = len(self.pursuer_ids)  # 更新实际使用的追捕者数量
            if self.n_pursuers == 0:  # 如果没有追捕者，无法继续
                raise ValueError("CQL: No pursuers available to train after filtering by name.")

        self.action_n_per_pursuer = [temp_env.action_space(p_id).n for p_id in self.pursuer_ids]
        # 生成所有可能的联合动作
        # 例如，如果 pursuer_ids = ['pursuer_0', 'pursuer_1'] 且每个都有5个动作
        # all_individual_actions = [range(5), range(5)]
        # self.joint_action_space = [(0,0), (0,1), ..., (4,4)]
        all_individual_actions = [range(n) for n in self.action_n_per_pursuer]
        self.joint_action_space = list(itertools.product(*all_individual_actions))
        self.num_joint_actions = len(self.joint_action_space)
        temp_env.close()

        self.q_table = defaultdict(lambda: np.zeros(self.num_joint_actions))

    def get_global_state_key(self, observations):
        # 从observations字典中提取所有pursuer_ids对应的观察，并确保是按self.pursuer_ids的固定顺序
        # 以确保全局状态键的一致性
        pursuer_obs_tuple = []
        for p_id in self.pursuer_ids:
            if p_id not in observations:  # 如果某个预期的pursuer不在当前观察中
                return None  # 表示无法形成完整的全局状态
            pursuer_obs_tuple.append(observations[p_id].tobytes())
        return tuple(pursuer_obs_tuple)

    def choose_joint_action_idx(self, global_state_key, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_joint_actions)
        else:
            q_values = self.q_table[global_state_key]
            max_q = np.max(q_values)
            return np.random.choice(np.where(q_values == max_q)[0])

    def update_q_table(self, global_state_key, joint_action_idx, total_reward, next_global_state_key, done):
        old_q_value = self.q_table[global_state_key][joint_action_idx]
        if done or next_global_state_key is None:
            target_q_value = total_reward
        else:
            # 如果next_global_state_key不在q_table中（即第一次遇到），其Q值为0
            next_max_q = np.max(self.q_table[next_global_state_key]) if next_global_state_key in self.q_table else 0.0
            target_q_value = total_reward + self.gamma * next_max_q

        self.q_table[global_state_key][joint_action_idx] = old_q_value + self.alpha * (target_q_value - old_q_value)

    def decay_epsilon_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)

    def train(self, num_episodes_train, max_cycles_train, eval_interval, num_eval_episodes, max_cycles_eval, seed=None):
        self.epsilon = self.epsilon_start  # 重置epsilon
        training_rewards_over_time = []
        env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles_train)

        for episode in range(num_episodes_train):
            observations, _ = env.reset(seed=seed + episode if seed is not None else None)
            episode_reward_sum = 0

            for step_num in range(max_cycles_train):
                if not env.agents or not any(p_id in env.agents for p_id in self.pursuer_ids):
                    break  # 如果环境中没有智能体了，或者所有我们关心的pursuer都不在了

                # 检查是否所有预期的pursuer都存在于当前观察中
                active_pursuers_in_obs = [p_id for p_id in self.pursuer_ids if p_id in observations]
                if len(active_pursuers_in_obs) != len(self.pursuer_ids):
                    # 如果不是所有预期的pursuer都在，可能意味着一些pursuer已经结束
                    # 此时无法形成完整的全局状态，也无法选择联合动作
                    # 可以选择让剩余的智能体随机行动，或者直接结束回合/跳过这个状态
                    # print(f"CQL Train Ep {episode+1} Step {step_num+1}: Not all expected pursuers active. Skipping Q-update.")
                    # 为简化，让所有当前智能体随机行动并进入下一步
                    actions_to_take_fallback = {agent_id: env.action_space(agent_id).sample() for agent_id in
                                                env.agents}
                    if not actions_to_take_fallback: break
                    observations, rewards, terminations, truncations, _ = env.step(actions_to_take_fallback)
                    # episode_reward_sum += sum(rewards.get(p_id,0) for p_id in self.pursuer_ids if p_id in rewards) # 累加奖励
                    if not any(p_id in env.agents for p_id in self.pursuer_ids): break  # 如果没有pursuer了
                    continue  # 继续到下一个step

                global_state_key = self.get_global_state_key(observations)
                if global_state_key is None:  # 应该不会发生，因为上面检查过了
                    # print(f"CQL Train Ep {episode+1} Step {step_num+1}: Global state key is None. This shouldn't happen.")
                    break

                joint_action_idx = self.choose_joint_action_idx(global_state_key, explore=True)
                joint_action_tuple = self.joint_action_space[joint_action_idx]

                actions_to_take = {p_id: joint_action_tuple[i] for i, p_id in enumerate(self.pursuer_ids)}

                for agent_id in env.agents:  # 其他智能体（如逃跑者）随机行动
                    if agent_id not in actions_to_take:  # 即非pursuer的智能体
                        actions_to_take[agent_id] = env.action_space(agent_id).sample()

                next_observations, rewards, terminations, truncations, _ = env.step(actions_to_take)

                total_reward_this_step = sum(rewards.get(p_id, 0) for p_id in self.pursuer_ids)
                episode_reward_sum += total_reward_this_step

                next_global_state_key = None
                if any(p_id in next_observations for p_id in self.pursuer_ids):  # 确保下一个观察中至少有一个pursuer
                    active_next_pursuers = [p_id for p_id in self.pursuer_ids if p_id in next_observations]
                    if len(active_next_pursuers) == len(self.pursuer_ids):  # 所有预期的pursuer都在下一个观察中
                        next_global_state_key = self.get_global_state_key(next_observations)

                # 回合结束条件：所有pursuer都结束，或者环境本身结束，或者达到最大步数
                done_this_step = (
                        all(terminations.get(p_id, False) or truncations.get(p_id, False) for p_id in
                            self.pursuer_ids) or
                        not env.agents or
                        step_num == max_cycles_train - 1
                )

                self.update_q_table(global_state_key, joint_action_idx, total_reward_this_step, next_global_state_key,
                                    done_this_step)

                observations = next_observations
                if done_this_step:
                    break

            self.decay_epsilon_episode()

            if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:
                avg_eval_reward = self.evaluate(num_eval_episodes, max_cycles_eval, seed)
                training_rewards_over_time.append(avg_eval_reward)
                print(
                    f"CQL - Seed {seed if seed is not None else 0} - Eval after Episode {episode + 1}: Avg Reward = {avg_eval_reward:.2f}, Eps: {self.epsilon:.3f}")

        env.close()
        return training_rewards_over_time

    def evaluate(self, num_episodes, max_cycles, seed=None):
        total_rewards_eval = 0
        eval_env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles)

        for episode in range(num_episodes):
            observations, _ = eval_env.reset(seed=seed + episode if seed is not None else None)
            episode_reward_sum = 0

            for step_num in range(max_cycles):
                if not eval_env.agents or not any(p_id in eval_env.agents for p_id in self.pursuer_ids): break

                active_pursuers_in_obs = [p_id for p_id in self.pursuer_ids if p_id in observations]
                if len(active_pursuers_in_obs) != len(self.pursuer_ids):
                    # print(f"CQL Eval Ep {episode+1} Step {step_num+1}: Not all expected pursuers active. Fallback random.")
                    actions_to_take_fallback = {agent_id: eval_env.action_space(agent_id).sample() for agent_id in
                                                eval_env.agents}
                    if not actions_to_take_fallback: break
                    observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take_fallback)
                    # episode_reward_sum += sum(rewards.get(p_id,0) for p_id in self.pursuer_ids if p_id in rewards)
                    if not any(p_id in eval_env.agents for p_id in self.pursuer_ids): break
                    continue

                global_state_key = self.get_global_state_key(observations)
                if global_state_key is None:  # Should not happen if above check works
                    # print(f"CQL Eval Ep {episode+1} Step {step_num+1}: Global state key is None. Breaking.")
                    break

                joint_action_idx = self.choose_joint_action_idx(global_state_key, explore=False)  # Greedy
                joint_action_tuple = self.joint_action_space[joint_action_idx]

                actions_to_take = {p_id: joint_action_tuple[i] for i, p_id in enumerate(self.pursuer_ids)}

                for agent_id in eval_env.agents:
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = eval_env.action_space(agent_id).sample()

                next_observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take)

                total_reward_this_step = sum(rewards.get(p_id, 0) for p_id in self.pursuer_ids)
                episode_reward_sum += total_reward_this_step

                observations = next_observations
                done_this_step = (
                        all(terminations.get(p_id, False) or truncations.get(p_id, False) for p_id in
                            self.pursuer_ids) or
                        not eval_env.agents or
                        step_num == max_cycles - 1
                )
                if done_this_step:
                    break
            total_rewards_eval += episode_reward_sum

        eval_env.close()
        return total_rewards_eval / num_episodes if num_episodes > 0 else 0


# --- Plotting ---
def plot_results(results_iql, results_cql, eval_interval, num_episodes_train_total, n_seeds):
    plt.figure(figsize=(12, 7))

    num_eval_points = len(results_iql[0])  # Assuming all seeds have same number of eval points
    eval_episodes_x_axis = np.arange(1, num_eval_points + 1) * eval_interval

    if results_iql:
        avg_iql_rewards = np.mean(results_iql, axis=0)
        std_iql_rewards = np.std(results_iql, axis=0)
        plt.plot(eval_episodes_x_axis, avg_iql_rewards, label=f'IQL (N_Pursuers={N_PURSUERS_IQL}) Rewards',
                 color='deepskyblue')
        plt.fill_between(eval_episodes_x_axis, avg_iql_rewards - std_iql_rewards, avg_iql_rewards + std_iql_rewards,
                         alpha=0.2, color='deepskyblue')

    if results_cql:
        avg_cql_rewards = np.mean(results_cql, axis=0)
        std_cql_rewards = np.std(results_cql, axis=0)
        plt.plot(eval_episodes_x_axis, avg_cql_rewards, label=f'CQL (N_Pursuers={N_PURSUERS_CQL}) Rewards',
                 color='salmon')
        plt.fill_between(eval_episodes_x_axis, avg_cql_rewards - std_cql_rewards, avg_cql_rewards + std_cql_rewards,
                         alpha=0.2, color='salmon')

    plt.xlabel(f'Training Episodes')
    plt.ylabel('Average Cumulative Reward during Evaluation')
    plt.title(f'IQL vs CQL Performance in Pursuit Environment (Avg over {n_seeds} seed(s))')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图表到文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"marl_pursuit_performance_{timestamp}.png"
    plt.savefig(filename)
    print(f"结果图表已保存为: {filename}")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    all_iql_rewards_over_seeds = []
    all_cql_rewards_over_seeds = []

    # 根据上面定义的超参数准备算法的参数字典
    hyperparams_algo = {
        'alpha': ALPHA,
        'gamma': GAMMA,
        'epsilon_start': EPSILON_START,
        'epsilon_end': EPSILON_END,
        'epsilon_decay_rate_per_episode': EPSILON_DECAY_RATE_PER_EPISODE
    }

    # --- IQL Training ---
    print("Starting IQL Training...")
    for seed_run in range(NUM_SEEDS):
        current_seed = seed_run  # 或者使用一个固定的种子列表
        print(f"\n--- IQL: Running Seed {current_seed + 1}/{NUM_SEEDS} ---")
        start_time_seed = time.time()

        iql_trainer = IQLTrainer(N_PURSUERS_IQL, N_EVADERS_IQL, hyperparams_algo)
        iql_rewards = iql_trainer.train(NUM_EPISODES_TRAIN, MAX_CYCLES_TRAIN, EVAL_INTERVAL, NUM_EVAL_EPISODES,
                                        MAX_CYCLES_EVAL, seed=current_seed)
        all_iql_rewards_over_seeds.append(iql_rewards)

        end_time_seed = time.time()
        print(f"IQL Seed {current_seed + 1} finished in {(end_time_seed - start_time_seed) / 60:.2f} minutes.")
        # 可选: 保存Q表
        # with open(f'iql_q_tables_seed_{current_seed}.pkl', 'wb') as f:
        #     pickle.dump({agent: dict(q_table) for agent, q_table in iql_trainer.q_tables.items()}, f)

    # --- CQL Training ---
    print("\n\nStarting CQL Training...")
    if N_PURSUERS_CQL > 2:  # 对于表格型CQL，超过2个pursuer通常不现实
        warnings.warn(f"WARNING: CQL N_PURSUERS_CQL is {N_PURSUERS_CQL}. Tabular CQL with >2 pursuers "
                      "can be extremely slow or run out of memory due to exponential joint action space. "
                      "Consider reducing N_PURSUERS_CQL or using function approximation (e.g., DQN).")

    for seed_run in range(NUM_SEEDS):
        current_seed = seed_run
        print(f"\n--- CQL: Running Seed {current_seed + 1}/{NUM_SEEDS} ---")
        start_time_seed = time.time()

        # 确保CQL的pursuer数量不会导致无法处理的联合动作空间
        if (5 ** N_PURSUERS_CQL) > 100000 and N_PURSUERS_CQL > 0:  # 5是动作数，这是一个粗略的警告阈值
            warnings.warn(
                f"CQL joint action space size is 5^{N_PURSUERS_CQL} = {5 ** N_PURSUERS_CQL}. This is very large!")

        cql_trainer = CQLTrainer(N_PURSUERS_CQL, N_EVADERS_CQL, hyperparams_algo)
        cql_rewards = cql_trainer.train(NUM_EPISODES_TRAIN, MAX_CYCLES_TRAIN, EVAL_INTERVAL, NUM_EVAL_EPISODES,
                                        MAX_CYCLES_EVAL, seed=current_seed)
        all_cql_rewards_over_seeds.append(cql_rewards)

        end_time_seed = time.time()
        print(f"CQL Seed {current_seed + 1} finished in {(end_time_seed - start_time_seed) / 60:.2f} minutes.")
        # 可选: 保存Q表
        # with open(f'cql_q_table_seed_{current_seed}.pkl', 'wb') as f:
        #     pickle.dump(dict(cql_trainer.q_table), f)

    # --- Plotting Results ---
    # 确保即使某个算法没有结果（例如NUM_SEEDS=0，或某个训练被跳过），绘图函数也能处理
    plot_iql_results = [res for res in all_iql_rewards_over_seeds if res]  # 过滤空结果
    plot_cql_results = [res for res in all_cql_rewards_over_seeds if res]

    if not plot_iql_results and not plot_cql_results:
        print(
            "No results to plot. Ensure training ran for at least one evaluation interval for at least one algorithm.")
    else:
        # 检查结果列表的长度是否一致，如果不一致，可能需要特殊处理或发出警告
        # 为简单起见，这里假设如果列表非空，则内部结果长度一致
        plot_results(plot_iql_results, plot_cql_results, EVAL_INTERVAL, NUM_EPISODES_TRAIN, NUM_SEEDS)

    print("\n--- Experiment Finished ---")