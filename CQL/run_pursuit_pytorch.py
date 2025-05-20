from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from RL_brain_DQN_torch import DeepQNetwork  # 确保导入的是不含CQL的DQN版本
import csv
import numpy as np
import torch
import random
import gymnasium
import os

# --- 可配置参数和设置 ---
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'pettingzoo_pursuit_CentralizedIQL_torch.csv'  # 更新日志文件名
MODEL_SAVE_PATH = "./tmp/centralized_iql_pursuit_torch.ckpt"  # 更新模型保存路径

# DQN 及学习过程的超参数
LEARNING_RATE = 0.001
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9
# E_GREEDY_INCREMENT = 0.0001

REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 500
SAVE_MODEL_FREQUENCY = 10

# 环境配置参数
NUM_PURSUERS = 8
NUM_EVADERS = 10


# --- 辅助函数 ---
def set_seeds(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def change_observation(observation_data, is_dict_observation_space):
    if is_dict_observation_space:
        pixel_observation = observation_data['observation']
    else:
        pixel_observation = observation_data
    return pixel_observation.flatten().astype(np.float32)


def get_joint_state(observations_dict, ordered_agent_ids, single_agent_n_features,
                    is_dict_obs_space, default_obs_val=0.0):
    joint_state_list = []
    for agent_id in ordered_agent_ids:
        if agent_id in observations_dict:
            raw_obs = observations_dict[agent_id]
            processed_obs = change_observation(raw_obs, is_dict_obs_space)
        else:
            processed_obs = np.full(single_agent_n_features, default_obs_val, dtype=np.float32)
        joint_state_list.append(processed_obs)
    return np.concatenate(joint_state_list)


# --- 主执行函数 ---
def run_pursuit_centralized_iql():  # 函数名更新
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if MODEL_SAVE_PATH:
        model_dir = os.path.dirname(MODEL_SAVE_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"已创建模型保存目录: {os.path.abspath(model_dir)}")
        elif model_dir and os.path.exists(model_dir):
            print(f"模型保存目录已存在: {os.path.abspath(model_dir)}")
        elif not model_dir:
            print(f"模型将保存在当前工作目录。")

    aec_env = pursuit_v4.env(
        n_pursuers=NUM_PURSUERS,
        n_evaders=NUM_EVADERS,
        max_cycles=MAX_STEPS_PER_EPISODE
    )
    env = aec_to_parallel_wrapper(aec_env)

    ORDERED_POSSIBLE_AGENTS = sorted(list(env.possible_agents))
    print(f"所有可能的智能体 (固定顺序): {ORDERED_POSSIBLE_AGENTS}")

    first_agent_id_for_space = ORDERED_POSSIBLE_AGENTS[0]
    agent_raw_obs_space = env.observation_space(first_agent_id_for_space)
    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    if is_dict_observation_space:
        pixel_obs_space = agent_raw_obs_space['observation']
    elif isinstance(agent_raw_obs_space, gymnasium.spaces.Box):
        pixel_obs_space = agent_raw_obs_space
    else:
        env.close()
        raise TypeError(f"未预期的观测空间类型: {type(agent_raw_obs_space)}")

    single_agent_n_features = np.prod(pixel_obs_space.shape)
    TOTAL_N_FEATURES = len(ORDERED_POSSIBLE_AGENTS) * single_agent_n_features
    N_ACTIONS = env.action_space(first_agent_id_for_space).n

    print(f"单个智能体特征数: {single_agent_n_features}")
    print(f"中心化状态总特征数 (N_FEATURES for DQN): {TOTAL_N_FEATURES}")
    print(f"动作数量 (N_ACTIONS for DQN): {N_ACTIONS}")

    RL = DeepQNetwork(  # 不再传递 alpha_cql, cql_temp
        n_actions=N_ACTIONS,
        n_features=TOTAL_N_FEATURES,
        learning_rate=LEARNING_RATE,
        reward_decay=REWARD_DECAY,
        e_greedy=E_GREEDY_INITIAL,
        replace_target_iter=REPLACE_TARGET_ITER,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        device=device
    )
    if TOTAL_N_FEATURES == 0:
        print("错误: TOTAL_N_FEATURES 为 0。DQN无法正确初始化。")
        env.close()
        return

    with open(LOG_FILE, 'w+', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(["Episode", "AverageReward_CentralizedIQL"])  # 更新列名

    global_step_counter = 0

    print("开始训练 (Centralized IQL with Parallel API)...")  # 更新打印信息
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)
        current_joint_state = get_joint_state(
            current_observations_dict,
            ORDERED_POSSIBLE_AGENTS,
            single_agent_n_features,
            is_dict_observation_space
        )
        episode_rewards_sum = {agent: 0.0 for agent in env.possible_agents}

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            if not env.agents:
                break

            actions_to_take_dict = {}
            for agent_id in env.agents:
                if agent_id not in current_observations_dict:
                    continue
                raw_obs_for_agent = current_observations_dict[agent_id]
                info_for_agent = current_infos_dict.get(agent_id, {})
                action_mask = np.ones(N_ACTIONS, dtype=np.int8)
                if is_dict_observation_space:
                    action_mask = raw_obs_for_agent['action_mask']
                else:
                    action_mask_from_info = info_for_agent.get('action_mask')
                    if action_mask_from_info is not None:
                        if isinstance(action_mask_from_info, np.ndarray) and action_mask_from_info.shape == (
                        N_ACTIONS,):
                            action_mask = action_mask_from_info
                        else:
                            if step_in_episode < 2 and episode == 0:
                                print(f"警告: Agent {agent_id} 的 info 字典中 action_mask 格式不正确。使用默认掩码。")
                    elif step_in_episode < 2 and episode == 0:
                        print(f"警告: Agent {agent_id} 的 info 字典中未找到 action_mask。使用默认掩码。")
                actions_to_take_dict[agent_id] = RL.choose_action(current_joint_state, action_mask)

            if not actions_to_take_dict:
                if env.agents:
                    pass
                break

            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)
            next_joint_state = get_joint_state(
                next_observations_dict,
                ORDERED_POSSIBLE_AGENTS,
                single_agent_n_features,
                is_dict_observation_space
            )

            for agent_id_acted in actions_to_take_dict.keys():
                s = current_joint_state
                a = actions_to_take_dict[agent_id_acted]
                r = rewards_dict.get(agent_id_acted, 0.0)
                terminated = terminated_dict.get(agent_id_acted, False)
                s_prime = next_joint_state
                RL.store_transition(s, a, r, s_prime, terminated)
                if agent_id_acted in episode_rewards_sum:
                    episode_rewards_sum[agent_id_acted] += r

            current_joint_state = next_joint_state
            current_observations_dict = next_observations_dict
            current_infos_dict = next_infos_dict

            global_step_counter += 1
            if global_step_counter > WARMUP_STEPS and \
                    global_step_counter % LEARN_EVERY_STEPS == 0 and \
                    RL.memory_counter >= RL.batch_size:
                RL.learn()

        num_total_possible_agents = len(env.possible_agents)
        avg_episode_reward = sum(
            episode_rewards_sum.values()) / num_total_possible_agents if num_total_possible_agents > 0 else 0.0

        with open(LOG_FILE, 'a', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([episode + 1, avg_episode_reward])

        if (episode + 1) % SAVE_MODEL_FREQUENCY == 0:
            print(
                f"回合: {episode + 1}/{NUM_EPISODES}, 平均奖励: {avg_episode_reward:.2f}, Epsilon: {RL.epsilon:.2f}, 全局步数: {global_step_counter}, 缓冲池大小: {RL.memory_counter}")
            if RL.cost_his:
                print(f"上次训练损失: {RL.cost_his[-1]:.4f}")
            RL.save_model(MODEL_SAVE_PATH)

    env.close()
    RL.save_model(MODEL_SAVE_PATH)
    print("训练结束 (Centralized IQL with Parallel API)。")  # 更新打印信息
    print(f"模型已保存到 {MODEL_SAVE_PATH}")
    print(f"结果已记录到 {LOG_FILE}")


if __name__ == "__main__":
    run_pursuit_centralized_iql()