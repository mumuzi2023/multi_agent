import gymnasium
import pettingzoo
from pettingzoo.mpe import simple_spread_v3
import time
import random

# 参数
N_AGENTS = 3  # 智能体的数量 (应与目标点数量相同或相关)
MAX_CYCLES = 100 # 每个回合的最大步数
RENDER_MODE = "human" # "human" 会显示一个窗口, None 则不显示

def run_random_demo():
    """运行一个 simple_spread_v3 的随机动作演示"""

    # 1. 初始化环境
    # 注意：在 simple_spread_v3 中，默认情况下，智能体的数量 N 也是目标点的数量 N
    env = simple_spread_v3.parallel_env(
        N=N_AGENTS,
        max_cycles=MAX_CYCLES,
        continuous_actions=False, # 我们使用离散动作
        render_mode=RENDER_MODE
    )

    # 2. 重置环境获取初始观测
    observations, infos = env.reset()
    print(f"初始观测 (agent_0): {observations['agent_0']}")
    print(f"智能体列表: {env.agents}") # 打印智能体ID列表，例如 ['agent_0', 'agent_1', 'agent_2']

    total_rewards_all_agents = {agent_id: 0.0 for agent_id in env.agents}

    # 3. 运行一个回合
    for step in range(MAX_CYCLES):
        if RENDER_MODE == "human":
            env.render() # 渲染环境 (如果 render_mode="human")
            time.sleep(0.1) # 等待0.1秒，方便观察

        # 为每个智能体选择一个随机动作
        actions = {}
        for agent_id in env.agents:
            action_space = env.action_space(agent_id) # 获取该智能体的动作空间
            action = action_space.sample() # 从动作空间中随机采样一个动作
            actions[agent_id] = action
            # print(f"智能体 {agent_id} 的动作空间: {action_space}, 选择动作: {action}")

        # 4. 执行动作
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # 打印一些信息
        # print(f"步骤: {step + 1}")
        # print(f"  奖励: {rewards}")
        # print(f"  终止状态: {terminations}")
        # print(f"  截断状态: {truncations}")

        for agent_id in env.agents:
            total_rewards_all_agents[agent_id] += rewards[agent_id]

        # 检查回合是否结束
        # 在 parallel_env 中，当任何一个 agent 的 termination 或 truncation 为 True 时，
        # 通常意味着整个回合对于该 agent 结束了。
        # 回合的真正结束通常是所有 agent 都 terminated/truncated，或者达到了 max_cycles。
        if any(terminations.values()) or any(truncations.values()):
            print(f"回合在步骤 {step + 1} 结束。")
            break

        # 更新观测值
        observations = next_observations

    print("\n回合结束后的总结:")
    for agent_id, total_reward in total_rewards_all_agents.items():
        print(f"智能体 {agent_id} 的总奖励: {total_reward:.2f}")

    # 5. 关闭环境
    env.close()
    print("环境已关闭。")

if __name__ == "__main__":
    print("开始运行 simple_spread_v3 随机动作演示...")
    run_random_demo()