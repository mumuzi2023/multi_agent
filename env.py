import time
from pettingzoo.sisl import pursuit_v4
import pygame  # Pygame is used for rendering


def run_random_pursuit_with_display():
    """
    运行PettingZoo SISL Pursuit环境，智能体随机行动，并显示界面。
    """
    # -------------------- 环境参数 --------------------
    num_pursuers = 99  # 追捕者数量 (可以调整)
    num_evaders = 2  # 逃跑者数量 (可以调整)
    max_cycles_per_episode = 100  # 每回合最大步数

    # -------------------- 初始化环境 --------------------
    # render_mode='human' 会创建一个窗口来显示环境
    # 注意：render_mode='human' 会显著减慢执行速度，主要用于可视化
    env = pursuit_v4.parallel_env(
        n_pursuers=num_pursuers,
        n_evaders=num_evaders,
        max_cycles=max_cycles_per_episode,
        render_mode='human'  # 关键：设置为 'human' 以显示界面
    )

    print(f"环境创建成功：{num_pursuers} 个追捕者, {num_evaders} 个逃跑者。")
    print("智能体将随机行动。按 Ctrl+C 停止。")

    try:
        for episode in range(3):  # 运行3个回合作为演示
            print(f"\n--- 开始回合 {episode + 1} ---")
            # 重置环境，获取初始观察
            observations, infos = env.reset()

            total_reward_this_episode = 0
            current_step = 0

            # -------------------- 回合运行循环 --------------------
            while env.agents:  # 当环境中还有智能体时继续
                # 1. 渲染环境 (对于 'human' 模式，这会在env.step()内部或外部自动处理)
                #    调用 env.render() 是确保渲染的明确方式，尽管在 parallel_env 和 human 模式下，
                #    step() 通常会触发渲染。
                env.render()

                # 2. 为每个当前活跃的智能体选择随机动作
                actions = {}
                for agent_id in env.agents:
                    action_space = env.action_space(agent_id)
                    action = action_space.sample()  # 随机选择一个动作
                    actions[agent_id] = action

                # 3. 执行动作
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                print(infos)

                # 4. 更新观察和累积奖励
                observations = next_observations

                # 计算并打印当前步骤的奖励信息 (可选)
                step_reward = sum(rewards.values())
                total_reward_this_episode += step_reward
                # print(f"回合 {episode + 1}, 步骤 {current_step + 1}: 奖励 = {step_reward:.2f}, 智能体: {list(env.agents)}")

                # 5. 检查回合是否结束
                # 在 parallel_env 中，当所有智能体都完成 (terminated 或 truncated) 时，env.agents 列表会变空
                # 或者当达到 max_cycles 时，环境会自动处理截断（truncation）。

                # 6. 短暂暂停以便观察
                time.sleep(0.1)  # 调整这个值可以改变显示速度 (例如0.05到0.5秒)
                current_step += 1
                if current_step >= max_cycles_per_episode:
                    print(f"达到最大步数 {max_cycles_per_episode}。")
                    break

            print(f"--- 回合 {episode + 1} 结束 --- 总奖励: {total_reward_this_episode:.2f}")

    except KeyboardInterrupt:
        print("\n用户中断了程序。")
    finally:
        # -------------------- 关闭环境 --------------------
        env.close()
        print("环境已关闭。")


if __name__ == "__main__":
    run_random_pursuit_with_display()