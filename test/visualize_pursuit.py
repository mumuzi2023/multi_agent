import torch
import numpy as np
import time
import gymnasium  # PettingZoo 使用 Gymnasium 的 spaces
import os
import argparse  # 用于从命令行读取模型路径
import random  # set_seeds 函数需要它

from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

# 方案1: 从 RL_brain_DQN_torch.py 导入 (推荐)
# 请确保 RL_brain_DQN_torch.py 与 visualize_pursuit.py 在同一目录，
# 或者 RL_brain_DQN_torch.py 所在的路径在 PYTHONPATH 中。
# DeepQNetwork 类内部会创建 QNetwork 实例，所以 QNetwork 也需要能被访问到。
# 如果 RL_brain_DQN_torch.py 中 QNetwork 是 DeepQNetwork 的内部类或未单独定义，
# 则可能需要调整导入或复制 QNetwork 定义。
# 假设 RL_brain_DQN_torch.py 中 DeepQNetwork 和 QNetwork 均可导入：
try:
    from RL_brain_DQN_torch import DeepQNetwork, QNetwork  # 尝试导入
except ImportError:
    print("错误: 无法从 RL_brain_DQN_torch.py 导入 DeepQNetwork 或 QNetwork。")
    print("请确保该文件存在且与此脚本在同一目录，或者将其类定义复制到此文件中。")
    print("为了继续，将尝试使用下面预留的类定义占位符（如果取消注释）。")
    # 方案2: 如果无法导入，可以将 QNetwork 和 DeepQNetwork 的类定义直接复制到这里
    # (如果这样做，请取消下面行的注释，并粘贴类定义)
    # class QNetwork(torch.nn.Module): ...
    # class DeepQNetwork: ...
    # if not ('DeepQNetwork' in globals() and 'QNetwork' in globals()): # 检查是否真的需要定义
    #     raise ImportError("DQN 类定义缺失，请从 RL_brain_DQN_torch.py 导入或复制到此文件。")
    pass  # 允许脚本继续，但如果类未定义，后续会出错


# --- 辅助函数 (从训练脚本复制) ---
def set_seeds(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def change_observation(observation_data, is_dict_observation_space):
    """
    处理来自环境的观测数据，如果是字典则提取像素部分，然后扁平化。
    """
    if is_dict_observation_space:
        pixel_observation = observation_data['observation']
    else:  # observation_data 已经是像素数组
        pixel_observation = observation_data
    return pixel_observation.flatten().astype(np.float32)


# --- 可视化函数 ---
def visualize_trained_agent(
        model_path: str,
        num_episodes_to_run: int = 3,
        max_steps_per_episode: int = 500,
        seed: int = 42,
        render_fps: int = 10
):
    print(f"\n--- 开始可视化训练后的智能体 (模型: {model_path}) ---")
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"可视化时使用设备: {device}")
    NUM_PURSUERS = 8  # 可调参数: 追捕者数量 (pursuit_v4 默认是 8)
    NUM_EVADERS = 10  # 可调参数: 逃跑者数量 (pursuit_v4 默认是 10)
    # 1. 初始化环境用于渲染 (AEC -> Parallel)
    aec_env_for_eval = pursuit_v4.env(
        render_mode='human',
        n_pursuers=NUM_PURSUERS,  # 使用配置的数量
        n_evaders=NUM_EVADERS,  # 使用配置的数量
        max_cycles=max_steps_per_episode)
    env = aec_to_parallel_wrapper(aec_env_for_eval)

    # 2. 获取环境的观测和动作空间信息
    first_agent_for_space_check = env.possible_agents[0]
    agent_raw_obs_space = env.observation_space(first_agent_for_space_check)
    is_dict_observation_space_eval = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    if is_dict_observation_space_eval:
        pixel_obs_space = agent_raw_obs_space['observation']
    elif isinstance(agent_raw_obs_space, gymnasium.spaces.Box):
        pixel_obs_space = agent_raw_obs_space
        print("可视化警告: 观测空间是 Box 类型。如果智能体策略需要动作掩码，期望其在 info 字典中。")
    else:
        env.close()
        raise TypeError(f"可视化时遇到未预期的观测空间类型: {type(agent_raw_obs_space)}")

    obs_shape = pixel_obs_space.shape
    N_FEATURES_EVAL = np.prod(obs_shape)
    N_ACTIONS_EVAL = env.action_space(first_agent_for_space_check).n
    print(f"可视化 - 特征数量: {N_FEATURES_EVAL}, 动作数量: {N_ACTIONS_EVAL}")

    # 3. 加载训练好的 DQN 模型
    # 确保 DeepQNetwork 类已定义或已导入
    if 'DeepQNetwork' not in globals():
        print("错误: DeepQNetwork 类未定义。请从 RL_brain_DQN_torch.py 导入或复制定义到此文件。")
        env.close()
        return

    agent_policy = DeepQNetwork(
        n_actions=N_ACTIONS_EVAL,
        n_features=N_FEATURES_EVAL,
        device=device,
        e_greedy=1.0,  # 执行模式，epsilon设为1（贪婪选择概率为1）
        e_greedy_increment=None  # 执行时不需要epsilon退火
    )
    try:
        agent_policy.load_model(model_path)
        print(f"模型从 {model_path} 加载成功。")
    except FileNotFoundError:
        print(f"错误: 模型文件未在路径 {model_path} 找到。")
        env.close()
        return
    except Exception as e:
        print(f"从 {model_path} 加载模型失败: {e}")
        env.close()
        return

    agent_policy.eval_net.eval()  # 将网络设置到评估模式

    render_delay = 1.0 / render_fps

    # 4. 运行可视化循环
    for episode in range(num_episodes_to_run):
        print(f"\n开始可视化回合 {episode + 1}/{num_episodes_to_run}")
        current_observations_dict, current_infos_dict = env.reset(seed=seed + episode)

        current_processed_observations = {}
        for agent_id, raw_obs in current_observations_dict.items():
            current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space_eval)

        episode_total_reward = {agent: 0.0 for agent in env.possible_agents}

        for step_num in range(max_steps_per_episode):
            env.render()

            if not env.agents:
                print(f"所有智能体在步骤 {step_num} 完成。")
                break

            actions_to_take_dict = {}
            for agent_id in env.agents:
                raw_obs_for_agent = current_observations_dict[agent_id]
                processed_obs_for_agent = current_processed_observations[agent_id]
                info_for_agent = current_infos_dict.get(agent_id, {})

                action_mask_for_agent = np.ones(N_ACTIONS_EVAL, dtype=np.int8)
                if is_dict_observation_space_eval:
                    action_mask_for_agent = raw_obs_for_agent['action_mask']
                elif info_for_agent:
                    action_mask_from_info = info_for_agent.get('action_mask')
                    if action_mask_from_info is not None:
                        action_mask_for_agent = action_mask_from_info
                    else:  # info 中没有 action_mask，继续使用全1掩码
                        if step_num < 2 and episode == 0:  # 仅在开始时少量打印此警告避免刷屏
                            print(
                                f"可视化警告 (回合{episode + 1}, 步骤{step_num}): Agent {agent_id} 的 info 字典中无 action_mask，使用默认掩码。Info: {info_for_agent}")
                elif step_num < 2 and episode == 0:  # info 为空
                    print(
                        f"可视化警告 (回合{episode + 1}, 步骤{step_num}): Agent {agent_id} 的 info 字典为空，使用默认掩码。")

                actions_to_take_dict[agent_id] = agent_policy.choose_action(
                    processed_obs_for_agent,
                    action_mask_for_agent,
                    execution=True  # 确保使用学习到的策略
                )

            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)

            for agent_id_rewarded in rewards_dict:
                if agent_id_rewarded in episode_total_reward:
                    episode_total_reward[agent_id_rewarded] += rewards_dict[agent_id_rewarded]

            current_observations_dict = next_observations_dict
            current_infos_dict = next_infos_dict
            current_processed_observations.clear()
            for agent_id, raw_obs in current_observations_dict.items():
                current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space_eval)

            time.sleep(render_delay)

            if any(truncated_dict.values()):
                print(f"回合在步骤 {step_num} 被截断。")
                break

        current_episode_total_reward_sum = sum(episode_total_reward.values())
        num_possible_agents = len(env.possible_agents)
        avg_reward = current_episode_total_reward_sum / num_possible_agents if num_possible_agents > 0 else 0
        print(
            f"回合 {episode + 1} 结束。总奖励: {current_episode_total_reward_sum:.2f}, 平均每智能体奖励: {avg_reward:.2f}")
        if num_episodes_to_run > 1 and episode < num_episodes_to_run - 1:
            time.sleep(1.5)

    env.close()
    print("--- 可视化结束 ---")


# --- 主执行模块 ---
if __name__ == "__main__":
    # --- 在代码中直接定义参数 ---
    # !!! 重要: 请确保将 MODEL_FILE_PATH 修改为您实际训练好的模型文件的路径 !!!
    MODEL_FILE_PATH = "./tmp/dqn_pursuit_parallel_torch.ckpt"  # <--- 修改这里! 例如: "your_trained_model.ckpt"

    VIS_EPISODES = 3  # 要可视化的回合数
    VIS_FPS = 8  # 渲染的每秒帧数
    VIS_SEED = 42  # 可视化时使用的随机种子
    VIS_MAX_STEPS = 500  # 每个可视化回合的最大步数
    # --- 参数定义结束 ---

    print(f"--- 开始可视化 ---")
    print(f"模型路径: {MODEL_FILE_PATH}")
    print(f"可视化回合数: {VIS_EPISODES}")
    print(f"FPS: {VIS_FPS}")
    print(f"随机种子: {VIS_SEED}")
    print(f"每回合最大步数: {VIS_MAX_STEPS}")

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"错误: 模型文件在路径 '{MODEL_FILE_PATH}' 未找到。请检查路径是否正确。")
    else:
        # 调用可视化函数，使用上面定义的参数
        visualize_trained_agent(
            model_path=MODEL_FILE_PATH,
            num_episodes_to_run=VIS_EPISODES,
            max_steps_per_episode=VIS_MAX_STEPS,
            seed=VIS_SEED,
            render_fps=VIS_FPS
        )