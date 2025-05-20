# 导入必要的库
import gymnasium as gym  # PettingZoo 使用 Gymnasium 作为其核心 API 和空间定义的基础
from pettingzoo.sisl import pursuit_v4  # 导入特定的 PettingZoo 环境：SISL 实验室的 Pursuit v4
import numpy as np  # 用于数值计算，特别是处理观察、奖励和 Q 表
import matplotlib.pyplot as plt  # 用于绘制训练结果图表
from collections import defaultdict  # 用于创建默认值为特定类型的字典，方便 Q 表的实现
import itertools  # 用于创建迭代器，例如生成所有可能的联合动作
import pickle  # (可选) 用于序列化和反序列化 Python 对象，例如保存和加载训练好的 Q 表
import time  # 用于计时，例如记录训练时长
import warnings  # 用于处理 Python 的警告信息，例如某些库的弃用警告

# --- 主要可调整超参数 ---
# 这些参数会显著影响训练的效果和时长，需要根据具体问题和计算资源进行调整

# --- 环境与评估设置 ---
MAX_CYCLES_TRAIN = 75  # 训练时每回合最大步数 (Max steps per training episode)。
# 智能体在一回合内与环境交互的最大次数。
# 较小的值可以加快训练迭代，但可能使智能体没足够时间完成任务。
MAX_CYCLES_EVAL = 100  # 评估时每回合最大步数 (Max steps per evaluation episode)。
# 评估时通常允许更多步数，以充分展示智能体学到的策略水平。
NUM_EPISODES_TRAIN = 1000  # 总训练回合数 (Total training episodes)。
# 智能体与环境交互的总回合数量。这是最重要的参数之一，直接影响学习的充分性。
# 对于复杂问题，通常需要更多的回合数。
EVAL_INTERVAL = 100  # 评估间隔 (Evaluation interval)。
# 每训练 EVAL_INTERVAL 个回合后，进行一次策略评估。用于监控学习进度。
NUM_EVAL_EPISODES = 10  # 每次评估时运行的回合数 (Number of episodes for each evaluation run)。
# 多次评估并取平均可以得到更稳定和可靠的性能指标。
NUM_SEEDS = 1  # 随机种子运行次数 (Number of random seeds for averaging results)。
# 为了结果的鲁棒性，通常会用多个不同的随机种子运行整个训练过程，然后对结果取平均。
# 理想情况下为3-10次，但会成倍增加总训练时间。此处设为1以便快速演示。

# --- 算法特定的环境参数 ---
# 这些参数定义了在不同算法训练时，环境中追捕者和逃跑者的数量

# IQL (Independent Q-Learning) 设置
N_PURSUERS_IQL = 3  # IQL训练时的追捕者数量。
# IQL中，每个智能体独立学习，计算复杂度大致随智能体数量线性增加。
N_EVADERS_IQL = 1  # IQL训练时的逃跑者数量。

# CQL (Centralized Q-Learning) 设置
N_PURSUERS_CQL = 2  # CQL训练时的追捕者数量。
# 【警告】对于表格型CQL，此值【必须】保持很小（通常2，最多3）。
# 因为其Q表的状态-动作空间大小随智能体数量指数级增长。
N_EVADERS_CQL = 1  # CQL训练时的逃跑者数量。

# --- Q学习算法参数 ---
ALPHA = 0.1  # 学习率 (Learning Rate)，符号 α。
# 控制Q值更新的步长。值越大，Q值变化越快。
# 太大可能导致学习不稳定、Q值震荡；太小则学习速度过慢。典型值在 [0.01, 0.5]。
GAMMA = 0.99  # 折扣因子 (Discount Factor)，符号 γ。
# 用于衡量未来奖励相对于当前奖励的重要性。值越接近1，智能体越有远见。
# 典型值在 [0.9, 0.999]。
EPSILON_START = 1.0  # 探索率 (Epsilon) 的初始值，用于 ε-greedy 策略。
# 训练开始时，设置为1.0意味着完全随机探索，以发现更多状态和动作。
EPSILON_END = 0.05  # 探索率的最终值。
# 训练后期，即使策略已经较优，仍保持少量探索（例如5%）以避免陷入局部最优。
# 典型值在 [0.01, 0.1]。
EPSILON_DECAY_EPISODES_TARGET_PERCENTAGE = 0.8  # Epsilon衰减目标：在总训练回合数的这个百分比时，epsilon衰减到EPSILON_END。
# 例如，80%的训练回合后，探索率降至最终值。
_num_decay_episodes = NUM_EPISODES_TRAIN * EPSILON_DECAY_EPISODES_TARGET_PERCENTAGE  # 计算实际用于衰减的回合数
if _num_decay_episodes > 0 and EPSILON_START > EPSILON_END:  # 确保衰减有意义
    # 每回合探索率的乘性衰减因子。
    # 计算公式: decay_rate = (end_epsilon / start_epsilon)^(1 / num_decay_steps)
    # 这样可以确保在 _num_decay_episodes次回合后，epsilon从EPSILON_START衰减到EPSILON_END。
    EPSILON_DECAY_RATE_PER_EPISODE = (EPSILON_END / EPSILON_START) ** (1 / _num_decay_episodes)
else:
    EPSILON_DECAY_RATE_PER_EPISODE = 1.0  # 如果不满足衰减条件（例如总回合数为0，或初始epsilon已小于等于最终epsilon），则不衰减。


# --- 辅助函数 ---

def obs_to_state_key(observation):
    """
    将环境返回的观察（observation）转换为可用作Q表键（key）的可哈希格式。
    表格型Q学习需要将状态映射到一个离散的、可哈希的表示。
    - 如果观察是一个NumPy数组（例如IQL的局部观察），将其转换为字节串。
    - 如果观察是一个字典（例如CQL中组合的全局观察，其中键是智能体ID，值是其局部观察数组），
      则将字典中每个智能体的观察数组转换为字节串，然后将这些字节串组成一个元组。
      注意：为了保证全局状态键的一致性，字典的值（即各智能体的观察）在转换为元组前需要按固定的智能体ID顺序排列。
            （在CQL的get_global_state_key中已通过sorted(self.pursuer_ids)来保证顺序）
    """
    if isinstance(observation, dict):  # 通常用于CQL的全局状态
        # 确保从字典中取值时，如果依赖于固定的智能体顺序，调用者需要保证
        # 这里简单地按值迭代，但CQL的get_global_state_key会按排序后的pursuer_ids取值
        return tuple(obs.tobytes() for obs in observation.values())
    return observation.tobytes()  # 用于IQL的局部观察


def create_pursuit_env(n_pursuers, n_evaders, max_cycles, render_mode=None):
    """
    创建并返回一个配置好的PettingZoo SISL Pursuit并行环境实例。
    参数:
        n_pursuers (int): 追捕者数量。
        n_evaders (int): 逃跑者数量。
        max_cycles (int): 每回合最大步数。
        render_mode (str, optional): 渲染模式 ('human', 'rgb_array', None)。默认为None（不渲染）。
    """
    env = pursuit_v4.parallel_env(  # 使用并行API创建环境
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        max_cycles=max_cycles,
        obs_range=7,  # 追捕者的视野范围 (例如7x7的局部网格)
        n_catch=2,  # 捕获一个逃跑者所需要的追捕者数量
        render_mode=render_mode
        # 其他可选参数如: field_size=(width, height) 用于设置网格世界大小
    )
    return env


# --- 独立Q学习 (Independent Q-Learning, IQL) 训练器类 ---
class IQLTrainer:
    """
    IQL训练器类，封装了IQL算法的实现细节。
    每个智能体独立学习自己的Q函数，将其他智能体视为环境的一部分。
    """

    def __init__(self, n_pursuers, n_evaders, hyperparams):
        """
        初始化IQL训练器。
        参数:
            n_pursuers (int): 环境中配置的追捕者数量。
            n_evaders (int): 环境中配置的逃跑者数量。
            hyperparams (dict): 包含算法超参数的字典 (alpha, gamma, epsilon等)。
        """
        self.n_pursuers = n_pursuers  # 期望的追捕者数量 (来自配置)
        self.n_evaders = n_evaders  # 期望的逃跑者数量 (来自配置)

        # 从超参数字典中获取Q学习参数
        self.alpha = hyperparams['alpha']
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']  # 保存初始epsilon，用于在每次新种子运行开始时重置
        self.epsilon = self.epsilon_start  # 当前的探索率
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate_per_episode']  # 每回合的衰减率

        # 创建一个临时环境以获取智能体信息和动作空间
        # 这比在主训练循环外创建和关闭环境更清晰
        temp_env = create_pursuit_env(n_pursuers, n_evaders, 1)  # max_cycles=1 足够获取信息

        # 获取所有名字中包含 "pursuer" 的潜在智能体的动作空间
        # 使用 temp_env.possible_agents 是因为这是在 reset() 之前获取所有潜在智能体ID的正确方法
        self.action_spaces = {
            agent: temp_env.action_space(agent)
            for agent in temp_env.possible_agents if "pursuer" in agent  # 只关心追捕者
        }
        temp_env.close()  # 关闭临时环境

        if not self.action_spaces:  # 如果没有找到任何追捕者智能体
            raise ValueError("IQL: 未找到追捕者智能体。请检查智能体命名（应为 'pursuer_X'）或环境设置。")

        # 为每个追捕者智能体创建独立的Q表
        # Q表是一个字典，键是状态（局部观察的哈希值），值是一个NumPy数组，表示该状态下每个动作的Q值。
        # defaultdict 使得在遇到新状态时，会自动创建一个全零的Q值数组。
        self.q_tables = {
            agent: defaultdict(lambda: np.zeros(self.action_spaces[agent].n))
            for agent in self.action_spaces.keys()  # agent是 'pursuer_0', 'pursuer_1', ...
        }
        self.agent_ids = list(self.action_spaces.keys())  # 保存追捕者智能体的ID列表

        # 核对配置的n_pursuers和实际找到的pursuer智能体数量是否一致
        if len(self.agent_ids) != self.n_pursuers:
            warnings.warn(f"IQL: 期望的追捕者数量 (N_PURSUERS_IQL={self.n_pursuers}) "
                          f"与根据名称'pursuer'找到的数量 ({len(self.agent_ids)}: {self.agent_ids}) 不符。"
                          f"将使用实际找到的 {len(self.agent_ids)} 个追捕者进行训练。")
            self.n_pursuers = len(self.agent_ids)  # 更新实际使用的追捕者数量为找到的数量

    def choose_action(self, agent_id, observation_key, explore=True):
        """
        根据ε-greedy策略为指定的智能体选择动作。
        参数:
            agent_id (str): 需要选择动作的智能体ID。
            observation_key (hashable): 当前智能体的局部观察（已转换为可哈希的键）。
            explore (bool): 是否启用探索。评估时应设为False。
        返回:
            int: 选择的动作。
        """
        if explore and np.random.rand() < self.epsilon:  # 以 epsilon 的概率随机选择动作
            return self.action_spaces[agent_id].sample()
        else:  # 以 1-epsilon 的概率选择当前最优动作 (greedy)
            q_values = self.q_tables[agent_id][observation_key]  # 获取当前状态下所有动作的Q值
            max_q = np.max(q_values)  # 找到最大的Q值
            # 如果有多个动作具有相同的最大Q值（例如初始时都为0），从中随机选择一个。
            # 这有助于打破平局，避免智能体总是选择具有相同Q值的动作中索引最小的那个。
            best_actions = np.where(q_values == max_q)[0]  # 找到所有具有最大Q值的动作的索引
            return np.random.choice(best_actions)  # 从最优动作中随机选择一个

    def update_q_table(self, agent_id, state_key, action, reward, next_state_key, done):
        """
        根据Q学习的更新规则更新指定智能体的Q表。
        Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        参数:
            agent_id (str): 要更新Q表的智能体ID。
            state_key (hashable): 当前状态的键。
            action (int): 执行的动作。
            reward (float): 执行动作后获得的奖励。
            next_state_key (hashable): 下一个状态的键。如果回合结束，可能为None。
            done (bool): 指示回合是否因为该智能体或全局原因结束。
        """
        old_q_value = self.q_tables[agent_id][state_key][action]  # 获取旧的Q(s,a)值

        if done or next_state_key is None:  # 如果是终止状态或者没有下一个状态
            target_q_value = reward  # 目标Q值就是即时奖励
        else:
            # 获取下一个状态的最大Q值。如果next_state_key是第一次遇到，其Q值默认为0。
            next_max_q = np.max(self.q_tables[agent_id][next_state_key]) if next_state_key in self.q_tables[
                agent_id] else 0.0
            target_q_value = reward + self.gamma * next_max_q  # 根据贝尔曼方程计算目标Q值

        # 更新Q值
        self.q_tables[agent_id][state_key][action] = old_q_value + self.alpha * (target_q_value - old_q_value)

    def decay_epsilon_episode(self):
        """在每回合结束时按设定的衰减率衰减探索率epsilon。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)

    def train(self, num_episodes_train, max_cycles_train, eval_interval, num_eval_episodes, max_cycles_eval, seed=None):
        """
        执行IQL训练过程。
        参数:
            num_episodes_train (int): 总训练回合数。
            max_cycles_train (int): 训练时每回合最大步数。
            eval_interval (int): 评估间隔。
            num_eval_episodes (int): 每次评估的回合数。
            max_cycles_eval (int): 评估时每回合最大步数。
            seed (int, optional): 用于环境重置的随机种子，以保证可复现性。
        返回:
            list: 包含每次评估时的平均累积奖励的列表。
        """
        self.epsilon = self.epsilon_start  # 每次调用train时（例如每个seed的训练开始时）重置epsilon
        training_rewards_over_time = []  # 存储每个评估点的平均奖励

        # 创建训练用的环境实例
        env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles_train)

        # 主训练循环，共进行 num_episodes_train 个回合
        for episode in range(num_episodes_train):  # `episode` 是当前回合的编号 (从0开始)
            # 重置环境，获取初始观察。为每个回合设置不同的种子（如果主种子提供了）以增加多样性。
            observations, infos = env.reset(seed=seed + episode if seed is not None else None)

            # 获取在回合开始时活跃的、且是我们IQL追踪的追捕者
            current_pursuers_in_episode = [p_id for p_id in self.agent_ids if p_id in observations]
            if not current_pursuers_in_episode:  # 如果没有活跃的追踪的追捕者（在reset时不太可能，但作为防御性编程）
                if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:  # 确保评估点数量正确
                    training_rewards_over_time.append(0)  # 如果回合无法开始，记录0奖励
                continue  # 跳过此回合

            # 将初始观察转换为Q表的键
            current_observations_keys = {
                agent: obs_to_state_key(observations[agent])
                for agent in current_pursuers_in_episode
            }

            episode_rewards_sum_for_all_pursuers = 0  # 用于累加本回合所有我们追踪的追捕者的总奖励

            # 单个回合内的步数循环，最多 max_cycles_train 步
            for step_num in range(max_cycles_train):  # `step_num` 是当前回合内的步数 (从0开始)
                # 如果环境中没有智能体了，或者没有存活的追捕者了，则提前结束本回合
                if not env.agents or not any("pursuer" in agent for agent in env.agents):
                    break

                actions_to_take = {}  # 存储本步骤所有智能体要执行的动作

                # 确定当前步骤仍然活跃且在我们追踪列表中的追捕者
                active_pursuers_in_step = [
                    p_id for p_id in current_pursuers_in_episode
                    if p_id in env.agents and p_id in current_observations_keys  # 确保智能体仍活跃且有当前观察
                ]
                if not active_pursuers_in_step:  # 如果没有我们追踪的活跃追捕者了
                    break  # 结束本回合

                # 为每个活跃的、被追踪的追捕者选择动作
                for agent_id in active_pursuers_in_step:
                    obs_key = current_observations_keys[agent_id]  # 获取该智能体的当前状态键
                    actions_to_take[agent_id] = self.choose_action(agent_id, obs_key, explore=True)  # ε-greedy选择动作

                # 为环境中其他智能体（例如逃跑者，或者未被IQL追踪的pursuer）选择随机动作
                for agent_id in env.agents:
                    if agent_id not in actions_to_take:  # 如果该智能体的动作尚未被决定
                        actions_to_take[agent_id] = env.action_space(agent_id).sample()

                if not actions_to_take: break  # 如果没有动作可以执行 (例如所有pursuer都已结束)

                # 在环境中执行联合动作，获取下一步的反馈
                next_observations, rewards, terminations, truncations, infos = env.step(actions_to_take)

                current_step_total_pursuer_reward = 0  # 当前步骤所有被追踪追捕者的奖励和
                next_obs_keys_this_step = {}  # 存储下一步观察的键

                # 为每个执行了动作的、被追踪的追捕者更新Q表
                for agent_id in active_pursuers_in_step:
                    if agent_id not in actions_to_take: continue  # 理论上不应发生，因为active_pursuers_in_step里的都应已决策

                    state_key = current_observations_keys[agent_id]  # 当前状态键
                    action = actions_to_take[agent_id]  # 执行的动作
                    reward = rewards.get(agent_id, 0)  # 从环境获取该智能体的奖励，如果不存在则为0
                    current_step_total_pursuer_reward += reward  # 累加到当前步骤的总奖励中

                    # 检查单个追捕者的奖励是否为正，如果是，则打印信息
                    if reward > 0:
                        print(
                            f"IQL - Seed {seed if seed is not None else 'N/A'} - 回合 {episode + 1} - 步数 {step_num + 1}: "
                            f"智能体 {agent_id} 获得正奖励: {reward:.2f}")

                    # 判断该智能体是否在本步骤结束 (terminated 或 truncated)
                    is_done = terminations.get(agent_id, False) or truncations.get(agent_id, False)

                    next_state_key_agent = None  # 初始化下一个状态的键
                    if agent_id in next_observations:  # 如果该智能体有下一个观察 (即它没有在本步结束)
                        next_state_key_agent = obs_to_state_key(next_observations[agent_id])
                        next_obs_keys_this_step[agent_id] = next_state_key_agent  # 存储下一个状态的键，用于下一轮迭代

                    # 更新Q表
                    self.update_q_table(agent_id, state_key, action, reward, next_state_key_agent, is_done)

                episode_rewards_sum_for_all_pursuers += current_step_total_pursuer_reward  # 累加到本回合总奖励
                current_observations_keys = next_obs_keys_this_step  # 更新当前观察键，为下一步做准备

                # 如果所有我们追踪的追捕者都已结束，或者环境中没有智能体了，则结束本回合
                if not any(p_id in env.agents for p_id in current_pursuers_in_episode) or not env.agents:
                    break

            # 每回合结束后，衰减探索率epsilon
            self.decay_epsilon_episode()

            # 如果达到评估间隔，或者这是最后一个训练回合，则进行策略评估
            if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:
                # 传递给evaluate的seed可以基于主seed和当前episode，确保评估有一定变化性但整体可控
                avg_eval_reward = self.evaluate(num_eval_episodes, max_cycles_eval,
                                                seed=(seed + episode) if seed is not None else None)
                training_rewards_over_time.append(avg_eval_reward)
                # 打印评估信息时，使用传递给train方法的主seed
                print(
                    f"IQL - Seed {seed if seed is not None else 'N/A'} - Eval after Episode {episode + 1}: Avg Reward = {avg_eval_reward:.2f}, Eps: {self.epsilon:.3f}")

        env.close()  # 关闭训练环境
        return training_rewards_over_time  # 返回训练过程中记录的评估奖励

    def evaluate(self, num_episodes, max_cycles, seed=None):
        """
        评估当前IQL策略的性能。在评估期间，不进行学习（Q表不更新），且采用贪婪策略（epsilon=0）。
        参数:
            num_episodes (int): 评估运行的回合数。
            max_cycles (int): 评估时每回合最大步数。
            seed (int, optional): 用于环境重置的种子，确保评估的可复现性或多样性。
        返回:
            float: 在评估回合中获得的平均累积奖励。
        """
        total_rewards_eval = 0  # 累积所有评估回合的总奖励
        eval_env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles)  # 创建评估环境

        for episode_eval in range(num_episodes):  # 运行指定数量的评估回合
            # 为每个评估回合设置不同的种子（如果主种子提供了）
            observations, _ = eval_env.reset(seed=seed + episode_eval if seed is not None else None)

            # 获取评估回合开始时活跃的、且是我们IQL追踪的追捕者
            current_pursuers_in_eval = [p_id for p_id in self.agent_ids if p_id in observations]
            if not current_pursuers_in_eval: continue  # 如果没有，跳过此评估回合

            current_observations_keys_eval = {
                agent: obs_to_state_key(observations[agent])
                for agent in current_pursuers_in_eval
            }
            episode_reward_sum_eval = 0  # 当前评估回合的累积奖励

            for step_num_eval in range(max_cycles):  # 单个评估回合的步数循环
                if not eval_env.agents or not any("pursuer" in agent for agent in eval_env.agents): break  # 环境结束条件

                actions_to_take = {}
                active_pursuers_in_step = [
                    p_id for p_id in current_pursuers_in_eval
                    if p_id in eval_env.agents and p_id in current_observations_keys_eval
                ]
                if not active_pursuers_in_step: break

                for agent_id in active_pursuers_in_step:  # 为每个活跃追捕者选择动作
                    obs_key = current_observations_keys_eval[agent_id]
                    actions_to_take[agent_id] = self.choose_action(agent_id, obs_key, explore=False)  # 评估时使用贪婪策略

                for agent_id in eval_env.agents:  # 其他智能体随机行动
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = eval_env.action_space(agent_id).sample()

                if not actions_to_take: break  # 如果没有动作可执行

                next_observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take)

                step_pursuer_reward_eval = 0
                next_obs_keys_this_step_eval = {}
                for agent_id in active_pursuers_in_step:  # 累加我们追踪的追捕者的奖励
                    if agent_id not in rewards: continue  # 如果pursuer已结束，可能没有它的reward
                    step_pursuer_reward_eval += rewards.get(agent_id, 0)
                    if agent_id in next_observations:  # 如果智能体有下一状态
                        next_obs_keys_this_step_eval[agent_id] = obs_to_state_key(next_observations[agent_id])

                episode_reward_sum_eval += step_pursuer_reward_eval  # 累加到本回合评估奖励
                current_observations_keys_eval = next_obs_keys_this_step_eval  # 更新观察

                # 检查是否所有追踪的pursuer都结束了
                if not any(p_id in eval_env.agents for p_id in current_pursuers_in_eval) or not eval_env.agents:
                    break
            total_rewards_eval += episode_reward_sum_eval  # 累加到总评估奖励

        eval_env.close()  # 关闭评估环境
        return total_rewards_eval / num_episodes if num_episodes > 0 else 0  # 返回平均每回合的奖励


# --- 中心化Q学习 (Centralized Q-Learning, CQL) 训练器类 ---
class CQLTrainer:
    """
    CQL训练器类，实现了一个完全中心化的Q学习智能体。
    它维护一个单一的Q表，该Q表基于全局状态（所有智能体观察的组合）和联合动作（所有智能体动作的组合）。
    【警告】由于状态空间和特别是联合动作空间随智能体数量指数增长，此方法仅适用于智能体数量非常少的情况。
    """

    def __init__(self, n_pursuers, n_evaders, hyperparams):
        """
        初始化CQL训练器。
        参数:
            n_pursuers (int): 环境中配置的追捕者数量。
            n_evaders (int): 环境中配置的逃跑者数量。
            hyperparams (dict): 包含算法超参数的字典。
        """
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.alpha = hyperparams['alpha']
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon = self.epsilon_start
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate_per_episode']

        temp_env = create_pursuit_env(n_pursuers, n_evaders, 1)
        # 获取所有名字中包含 "pursuer" 的潜在智能体的ID，并排序以确保全局状态和联合动作的一致性
        self.pursuer_ids = sorted([agent for agent in temp_env.possible_agents if "pursuer" in agent])

        if not self.pursuer_ids:
            raise ValueError("CQL: 未找到追捕者智能体。")
        # 核对并更新实际使用的追捕者数量
        if len(self.pursuer_ids) != self.n_pursuers:
            warnings.warn(f"CQL: 期望的追捕者数量 (N_PURSUERS_CQL={self.n_pursuers}) "
                          f"与根据名称找到的数量 ({len(self.pursuer_ids)}: {self.pursuer_ids}) 不符。"
                          f"将使用实际找到的 {len(self.pursuer_ids)} 个追捕者。")
            self.n_pursuers = len(self.pursuer_ids)
            if self.n_pursuers == 0:  # 如果没有追捕者了，CQL无法运行
                raise ValueError("CQL: 筛选后没有可用的追捕者进行训练。")

        # 获取每个被CQL控制的追捕者的个体动作数量
        self.action_n_per_pursuer = [temp_env.action_space(p_id).n for p_id in self.pursuer_ids]

        # 生成所有可能的联合动作。例如，如果2个pursuer各有5个动作，则有5*5=25个联合动作。
        # all_individual_actions会是类似 [range(5), range(5)]
        # itertools.product会生成这些范围的笛卡尔积，即 [(0,0), (0,1), ..., (4,4)]
        all_individual_actions = [range(n) for n in self.action_n_per_pursuer]
        self.joint_action_space = list(itertools.product(*all_individual_actions))  # 存储所有联合动作元组
        self.num_joint_actions = len(self.joint_action_space)  # 联合动作的总数量
        temp_env.close()

        # 创建中心Q表。键是全局状态的哈希值，值是一个NumPy数组，表示该全局状态下每个联合动作的Q值。
        self.q_table = defaultdict(lambda: np.zeros(self.num_joint_actions))

    def get_global_state_key(self, observations):
        """
        从所有智能体的局部观察字典中构建全局状态的键。
        为了保证键的一致性，严格按照 self.pursuer_ids 中定义的顺序来组合观察。
        参数:
            observations (dict): 当前时间步所有活跃智能体的观察字典。
        返回:
            tuple or None: 如果所有预期的pursuer都有观察，则返回由其观察字节串组成的元组；否则返回None。
        """
        pursuer_obs_tuple_list = []
        for p_id in self.pursuer_ids:  # 严格按照预定义的、排序后的pursuer_ids顺序
            if p_id not in observations:  # 如果某个预期的pursuer不在当前观察中（可能已结束）
                return None  # 表示无法形成一个完整的、预期的全局状态
            pursuer_obs_tuple_list.append(observations[p_id].tobytes())  # 将观察转为字节串
        return tuple(pursuer_obs_tuple_list)  # 返回所有相关观察字节串的元组作为键

    def choose_joint_action_idx(self, global_state_key, explore=True):
        """
        根据ε-greedy策略选择一个联合动作的索引。
        参数:
            global_state_key (hashable): 当前全局状态的键。
            explore (bool): 是否启用探索。
        返回:
            int: 选择的联合动作在 self.joint_action_space 列表中的索引。
        """
        if explore and np.random.rand() < self.epsilon:  # 以 epsilon 概率随机选择联合动作
            return np.random.randint(self.num_joint_actions)
        else:  # 以 1-epsilon 概率选择当前最优联合动作
            q_values = self.q_table[global_state_key]  # 获取当前全局状态下所有联合动作的Q值
            max_q = np.max(q_values)
            best_actions_indices = np.where(q_values == max_q)[0]  # 找到所有最优联合动作的索引
            return np.random.choice(best_actions_indices)  # 从中随机选一个（处理平局）

    def update_q_table(self, global_state_key, joint_action_idx, total_reward, next_global_state_key, done):
        """
        更新中心Q表。
        Q_total(S, A) <- Q_total(S, A) + α * [R_total + γ * max_A' Q_total(S', A') - Q_total(S, A)]
        其中 S 是全局状态, A 是联合动作, R_total 是所有被控智能体的奖励总和。
        参数:
            global_state_key (hashable): 当前全局状态的键。
            joint_action_idx (int): 执行的联合动作的索引。
            total_reward (float): 所有被控追捕者在这一步获得的奖励总和。
            next_global_state_key (hashable or None): 下一个全局状态的键。如果回合结束或无法形成则为None。
            done (bool): 指示回合是否结束。
        """
        old_q_value = self.q_table[global_state_key][joint_action_idx]  # 获取旧的Q(S,A)

        if done or next_global_state_key is None:  # 如果是终止状态或没有有效的下一全局状态
            target_q_value = total_reward  # 目标Q值就是即时总奖励
        else:
            # 获取下一全局状态的最大Q值。如果next_global_state_key是新状态，Q值默认为0。
            next_max_q = np.max(self.q_table[next_global_state_key]) if next_global_state_key in self.q_table else 0.0
            target_q_value = total_reward + self.gamma * next_max_q  # 目标Q值

        # 更新Q值
        self.q_table[global_state_key][joint_action_idx] = old_q_value + self.alpha * (target_q_value - old_q_value)

    def decay_epsilon_episode(self):
        """在每回合结束时按设定的衰减率衰减探索率epsilon。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)

    def train(self, num_episodes_train, max_cycles_train, eval_interval, num_eval_episodes, max_cycles_eval, seed=None):
        """执行CQL训练过程。"""
        self.epsilon = self.epsilon_start  # 重置epsilon
        training_rewards_over_time = []  # 存储评估结果
        env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles_train)  # 创建训练环境

        for episode in range(num_episodes_train):  # 主训练循环
            observations, _ = env.reset(seed=seed + episode if seed is not None else None)
            episode_reward_sum = 0  # 本回合所有被控追捕者的奖励总和

            for step_num in range(max_cycles_train):  # 单回合步数循环
                # 如果环境中没有智能体了，或者所有我们CQL控制的pursuer都不在了，则结束本回合
                if not env.agents or not any(p_id in env.agents for p_id in self.pursuer_ids):
                    break

                    # 检查是否所有预期的、被CQL控制的pursuer都存在于当前观察中
                # 这是形成有效全局状态的前提
                active_controlled_pursuers = [p_id for p_id in self.pursuer_ids if p_id in observations]
                if len(active_controlled_pursuers) != len(self.pursuer_ids):
                    # 如果并非所有预期的pursuer都活跃（例如有的已经结束了），则无法形成完整的全局状态。
                    # 这种情况下，Q学习更新比较困难。
                    # 简化处理：让当前所有（包括非CQL控制的）智能体随机行动，然后进入下一步。
                    # 不进行Q表更新，因为没有有效的(S,A,R,S')转移。
                    # print(f"CQL Train Ep {episode+1} Step {step_num+1}: Not all expected pursuers active. Fallback random action.")
                    actions_to_take_fallback = {agent_id: env.action_space(agent_id).sample() for agent_id in
                                                env.agents}
                    if not actions_to_take_fallback: break  # 如果没动作可执行了
                    observations, rewards, terminations, truncations, _ = env.step(actions_to_take_fallback)
                    # 可以选择是否累加这一步的奖励到episode_reward_sum，但由于未更新Q表，这里可省略。
                    # episode_reward_sum += sum(rewards.get(p_id,0) for p_id in self.pursuer_ids if p_id in rewards)
                    if not any(p_id in env.agents for p_id in self.pursuer_ids): break  # 如果没有pursuer了
                    continue  # 跳过本step的Q学习部分，直接到下一个step

                # 获取当前全局状态的键
                global_state_key = self.get_global_state_key(observations)
                if global_state_key is None:  # 理论上，经过上面的检查，这里不应为None
                    # print(f"CQL Train Ep {episode+1} Step {step_num+1}: Global state key is None unexpectedly. Breaking.")
                    break

                    # 根据ε-greedy选择联合动作的索引
                joint_action_idx = self.choose_joint_action_idx(global_state_key, explore=True)
                joint_action_tuple = self.joint_action_space[joint_action_idx]  # 获取实际的联合动作元组

                # 构建传递给env.step()的动作字典
                actions_to_take = {p_id: joint_action_tuple[i] for i, p_id in enumerate(self.pursuer_ids)}

                # 其他非CQL控制的智能体（例如逃跑者）随机行动
                for agent_id in env.agents:
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = env.action_space(agent_id).sample()

                # 执行联合动作
                next_observations, rewards, terminations, truncations, _ = env.step(actions_to_take)

                # 计算所有被CQL控制的追捕者的总奖励
                total_reward_this_step = sum(rewards.get(p_id, 0) for p_id in self.pursuer_ids)
                episode_reward_sum += total_reward_this_step

                # (可选) 打印正的联合奖励信息
                if total_reward_this_step > 0 and any(
                        rewards.get(p_id, 0) > 0 for p_id in self.pursuer_ids if p_id in rewards):
                    print(
                        f"CQL - Seed {seed if seed is not None else 'N/A'} - 回合 {episode + 1} - 步数 {step_num + 1}: "
                        f"获得联合正奖励: {total_reward_this_step:.2f}")

                # 获取下一个全局状态的键
                next_global_state_key = None
                # 检查下一个观察中是否所有预期的pursuer都存在，以形成有效的下一全局状态
                if any(p_id in next_observations for p_id in self.pursuer_ids):
                    active_next_pursuers = [p_id for p_id in self.pursuer_ids if p_id in next_observations]
                    if len(active_next_pursuers) == len(self.pursuer_ids):  # 必须所有预期的pursuer都在
                        next_global_state_key = self.get_global_state_key(next_observations)

                # 判断回合是否结束的条件
                # 1. 所有被CQL控制的pursuer都结束了 (terminated or truncated)
                # 2. PettingZoo环境说没有智能体了 (env.agents is empty)
                # 3. 达到了本回合的最大步数
                done_this_step = (
                        all(terminations.get(p_id, False) or truncations.get(p_id, False) for p_id in
                            self.pursuer_ids) or
                        not env.agents or
                        step_num == max_cycles_train - 1  # 注意：step_num从0开始
                )

                # 更新中心Q表
                self.update_q_table(global_state_key, joint_action_idx, total_reward_this_step, next_global_state_key,
                                    done_this_step)

                observations = next_observations  # 更新观察，为下一步做准备
                if done_this_step:  # 如果回合结束
                    break

            # 每回合结束，衰减epsilon
            self.decay_epsilon_episode()

            # 定期评估
            if (episode + 1) % eval_interval == 0 or episode == num_episodes_train - 1:
                avg_eval_reward = self.evaluate(num_eval_episodes, max_cycles_eval,
                                                seed=(seed + episode) if seed is not None else None)  # 传递评估种子
                training_rewards_over_time.append(avg_eval_reward)
                print(
                    f"CQL - Seed {seed if seed is not None else 'N/A'} - Eval after Episode {episode + 1}: Avg Reward = {avg_eval_reward:.2f}, Eps: {self.epsilon:.3f}")

        env.close()  # 关闭训练环境
        return training_rewards_over_time

    def evaluate(self, num_episodes, max_cycles, seed=None):
        """评估CQL策略性能。"""
        total_rewards_eval = 0
        eval_env = create_pursuit_env(self.n_pursuers, self.n_evaders, max_cycles)  # 创建评估环境
        for episode_eval in range(num_episodes):  # 运行评估回合
            observations, _ = eval_env.reset(seed=seed + episode_eval if seed is not None else None)
            episode_reward_sum = 0
            for step_num_eval in range(max_cycles):  # 单回合步数循环
                if not eval_env.agents or not any(p_id in eval_env.agents for p_id in self.pursuer_ids): break

                # 检查是否所有预期的pursuer都活跃，以形成全局状态
                active_controlled_pursuers_eval = [p_id for p_id in self.pursuer_ids if p_id in observations]
                if len(active_controlled_pursuers_eval) != len(self.pursuer_ids):
                    # print(f"CQL Eval Ep {episode_eval+1} Step {step_num_eval+1}: Not all pursuers active. Fallback random.")
                    actions_to_take_fallback = {agent_id: eval_env.action_space(agent_id).sample() for agent_id in
                                                eval_env.agents}
                    if not actions_to_take_fallback: break
                    observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take_fallback)
                    # episode_reward_sum += sum(rewards.get(p_id,0) for p_id in self.pursuer_ids if p_id in rewards)
                    if not any(p_id in eval_env.agents for p_id in self.pursuer_ids): break
                    continue

                global_state_key = self.get_global_state_key(observations)
                if global_state_key is None:  # 如果无法获取有效全局状态
                    # print(f"CQL Eval Ep {episode_eval+1} Step {step_num_eval+1}: Global state key is None. Breaking.")
                    break

                # 评估时使用贪婪策略 (explore=False)
                joint_action_idx = self.choose_joint_action_idx(global_state_key, explore=False)
                joint_action_tuple = self.joint_action_space[joint_action_idx]

                actions_to_take = {p_id: joint_action_tuple[i] for i, p_id in enumerate(self.pursuer_ids)}
                for agent_id in eval_env.agents:  # 其他智能体随机行动
                    if agent_id not in actions_to_take:
                        actions_to_take[agent_id] = eval_env.action_space(agent_id).sample()

                next_observations, rewards, terminations, truncations, _ = eval_env.step(actions_to_take)

                total_reward_this_step = sum(rewards.get(p_id, 0) for p_id in self.pursuer_ids)
                episode_reward_sum += total_reward_this_step
                observations = next_observations  # 更新观察

                done_this_step = (
                        all(terminations.get(p_id, False) or truncations.get(p_id, False) for p_id in
                            self.pursuer_ids) or
                        not eval_env.agents or
                        step_num_eval == max_cycles - 1
                )
                if done_this_step:  # 如果回合结束
                    break
            total_rewards_eval += episode_reward_sum  # 累加本评估回合的奖励

        eval_env.close()  # 关闭评估环境
        return total_rewards_eval / num_episodes if num_episodes > 0 else 0  # 返回平均奖励


# --- 绘图函数 ---
def plot_results(results_iql, results_cql, eval_interval, num_episodes_train_total, n_seeds):
    """
    绘制IQL和CQL算法在训练过程中的评估奖励曲线。
    参数:
        results_iql (list of lists): IQL的评估奖励列表，外层列表对应不同种子，内层列表对应评估点。
        results_cql (list of lists): CQL的评估奖励列表。
        eval_interval (int): 评估间隔。
        num_episodes_train_total (int): 总训练回合数 (用于确定x轴范围，但实际点数由results列表长度决定)。
        n_seeds (int): 运行的随机种子数量。
    """
    plt.figure(figsize=(12, 7))  # 设置图表大小

    num_eval_points = 0  # 初始化评估点的数量
    # 尝试从IQL或CQL的结果中确定评估点的数量（它们应该是一致的）
    if results_iql and results_iql[0]:  # 如果IQL有结果且第一个种子的结果非空
        num_eval_points = len(results_iql[0])
    elif results_cql and results_cql[0]:  # 否则，尝试从CQL获取
        num_eval_points = len(results_cql[0])

    if num_eval_points == 0:  # 如果没有有效的评估点数据
        print("没有评估数据点可以绘制。")
        # 可以选择创建一个空的图表或直接返回
        plt.title(f'无有效评估数据 (Avg over {n_seeds} seed(s))')
        plt.xlabel(f'Training Episodes (Evaluated every {eval_interval} episodes)')
        plt.ylabel('Average Cumulative Reward during Evaluation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    # 计算x轴的刻度，代表每个评估点对应的训练回合数
    eval_episodes_x_axis = np.arange(1, num_eval_points + 1) * eval_interval

    # 绘制IQL的结果曲线（如果存在）
    if results_iql and all(len(r) == num_eval_points for r in results_iql):  # 确保所有种子的结果长度一致
        avg_iql_rewards = np.mean(results_iql, axis=0)  # 计算多个种子下每个评估点的平均奖励
        std_iql_rewards = np.std(results_iql, axis=0)  # 计算标准差，用于表示结果的波动范围
        plt.plot(eval_episodes_x_axis, avg_iql_rewards, label=f'IQL (N_Pursuers={N_PURSUERS_IQL}) Rewards',
                 color='deepskyblue')
        # 填充平均奖励上下一个标准差的区域，以可视化不确定性/方差
        plt.fill_between(eval_episodes_x_axis, avg_iql_rewards - std_iql_rewards, avg_iql_rewards + std_iql_rewards,
                         alpha=0.2, color='deepskyblue')

    # 绘制CQL的结果曲线（如果存在）
    if results_cql and all(len(r) == num_eval_points for r in results_cql):  # 确保所有种子的结果长度一致
        avg_cql_rewards = np.mean(results_cql, axis=0)
        std_cql_rewards = np.std(results_cql, axis=0)
        plt.plot(eval_episodes_x_axis, avg_cql_rewards, label=f'CQL (N_Pursuers={N_PURSUERS_CQL}) Rewards',
                 color='salmon')
        plt.fill_between(eval_episodes_x_axis, avg_cql_rewards - std_cql_rewards, avg_cql_rewards + std_cql_rewards,
                         alpha=0.2, color='salmon')

    plt.xlabel(f'Training Episodes')  # x轴标签
    plt.ylabel('Average Cumulative Reward during Evaluation')  # y轴标签
    plt.title(f'IQL vs CQL Performance in Pursuit Environment (Avg over {n_seeds} seed(s))')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    # 保存图表到文件，文件名包含时间戳以避免覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"marl_pursuit_performance_{timestamp}.png"
    try:
        plt.savefig(filename)  # 尝试保存
        print(f"结果图表已保存为: {filename}")
    except Exception as e:
        print(f"保存图表失败: {e}")
    plt.show()  # 显示图表


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 用于存储每个算法在不同随机种子下的评估奖励序列
    all_iql_rewards_over_seeds = []
    all_cql_rewards_over_seeds = []

    # 将之前定义的Q学习超参数打包成字典，方便传递给训练器
    hyperparams_algo = {
        'alpha': ALPHA,
        'gamma': GAMMA,
        'epsilon_start': EPSILON_START,
        'epsilon_end': EPSILON_END,
        'epsilon_decay_rate_per_episode': EPSILON_DECAY_RATE_PER_EPISODE  # 使用计算好的每回合衰减率
    }

    # --- IQL 训练 ---
    print("开始 IQL 训练...")
    for seed_run in range(NUM_SEEDS):  # 针对每个随机种子进行一次完整的训练和评估流程
        current_seed = seed_run  # 当前使用的种子（也可以是一个预定义的种子列表中的值）
        print(f"\n--- IQL: 运行随机种子 {current_seed + 1}/{NUM_SEEDS} ---")
        start_time_seed = time.time()  # 记录当前种子运行的开始时间

        # 创建IQL训练器实例
        iql_trainer = IQLTrainer(N_PURSUERS_IQL, N_EVADERS_IQL, hyperparams_algo)
        # 调用训练方法，传入相关参数
        iql_rewards_for_this_seed = iql_trainer.train(
            NUM_EPISODES_TRAIN, MAX_CYCLES_TRAIN,
            EVAL_INTERVAL, NUM_EVAL_EPISODES, MAX_CYCLES_EVAL,
            seed=current_seed  # 传递当前种子
        )
        if iql_rewards_for_this_seed:  # 如果训练产生了有效的评估结果
            all_iql_rewards_over_seeds.append(iql_rewards_for_this_seed)

        end_time_seed = time.time()  # 记录当前种子运行的结束时间
        print(f"IQL 随机种子 {current_seed + 1} 训练完成，用时 {(end_time_seed - start_time_seed) / 60:.2f} 分钟.")

        # (可选) 保存训练好的Q表。defaultdict需要转换为普通dict才能pickle。
        # with open(f'iql_q_tables_seed_{current_seed}.pkl', 'wb') as f:
        #     pickle.dump({agent: dict(q_table) for agent, q_table in iql_trainer.q_tables.items()}, f)

    # --- CQL 训练 ---
    print("\n\n开始 CQL 训练...")
    # 对表格型CQL的智能体数量发出警告，因为其复杂度非常高
    if N_PURSUERS_CQL > 2:
        warnings.warn(f"警告: CQL的追捕者数量 N_PURSUERS_CQL 设置为 {N_PURSUERS_CQL}。"
                      "对于表格型CQL，超过2个追捕者通常会导致联合动作空间过大，"
                      "训练将极其缓慢或耗尽内存。请考虑减少数量或使用函数逼近方法（如DQN）。")
    # 对联合动作空间大小发出警告
    if N_PURSUERS_CQL > 0 and (5 ** N_PURSUERS_CQL) > 100000:  # 5是pursuit环境的默认动作数
        warnings.warn(f"警告: CQL的联合动作空间大小为 5^{N_PURSUERS_CQL} = {5 ** N_PURSUERS_CQL}。"
                      "这是一个非常大的离散动作空间！")

    for seed_run in range(NUM_SEEDS):  # 针对每个随机种子进行训练
        current_seed = seed_run
        print(f"\n--- CQL: 运行随机种子 {current_seed + 1}/{NUM_SEEDS} ---")
        start_time_seed = time.time()

        if N_PURSUERS_CQL > 0:  # 只有在配置了至少一个追捕者时才运行CQL训练
            # 创建CQL训练器实例
            cql_trainer = CQLTrainer(N_PURSUERS_CQL, N_EVADERS_CQL, hyperparams_algo)
            # 调用训练方法
            cql_rewards_for_this_seed = cql_trainer.train(
                NUM_EPISODES_TRAIN, MAX_CYCLES_TRAIN,
                EVAL_INTERVAL, NUM_EVAL_EPISODES, MAX_CYCLES_EVAL,
                seed=current_seed  # 传递当前种子
            )
            if cql_rewards_for_this_seed:  # 如果训练产生了有效的评估结果
                all_cql_rewards_over_seeds.append(cql_rewards_for_this_seed)
        else:
            print(f"CQL: 由于配置的追捕者数量 N_PURSUERS_CQL 为 {N_PURSUERS_CQL}，跳过CQL训练。")
            # 如果跳过训练，为了绘图时数据结构一致性，可以考虑添加空的或占位的数据，
            # 但更简单的做法是在绘图函数中检查列表是否为空。

        end_time_seed = time.time()
        print(
            f"CQL 随机种子 {current_seed + 1} 训练完成（如果运行），用时 {(end_time_seed - start_time_seed) / 60:.2f} 分钟。")

        # (可选) 保存训练好的Q表
        # if N_PURSUERS_CQL > 0 and 'cql_trainer' in locals(): # 确保cql_trainer已创建
        #     with open(f'cql_q_table_seed_{current_seed}.pkl', 'wb') as f:
        #         pickle.dump(dict(cql_trainer.q_table), f)

    # --- 绘制并显示结果 ---
    # 过滤掉可能因跳过训练而产生的空结果列表（例如NUM_SEEDS=0或某个算法的N_PURSUERS=0）
    plot_iql_results = [res for res in all_iql_rewards_over_seeds if res]
    plot_cql_results = [res for res in all_cql_rewards_over_seeds if res]

    if not plot_iql_results and not plot_cql_results:  # 如果两个算法都没有结果
        print("没有可供绘制的结果。请确保至少有一个算法成功运行并产生了评估数据。")
    else:
        # 调用绘图函数
        plot_results(plot_iql_results, plot_cql_results, EVAL_INTERVAL, NUM_EPISODES_TRAIN, NUM_SEEDS)

    print("\n--- 实验结束 ---")