# 最简测试脚本示例
from pettingzoo.sisl import pursuit_v4
import gymnasium # PettingZoo 使用 Gymnasium 的 spaces

try:
    env = pursuit_v4.env()
    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    print(f"Observation space type: {type(obs_space)}")
    print(f"Observation space: {obs_space}")

    env.reset()
    # 对于 AECEnv，reset() 后，第一个智能体的观测通过 env.last() 获取
    first_agent_selected = env.agent_selection
    first_obs, _, _, _, info = env.last() # info 通常是 agent-specific

    print(f"\nFor agent: {first_agent_selected}")
    print(f"First observation type: {type(first_obs)}")
    # 如果 obs_space 是 Dict，那么 first_obs 也应该是 dict
    if isinstance(first_obs, dict):
        print(f"First observation keys: {first_obs.keys()}")
        if 'action_mask' in first_obs:
            print(f"Action mask in first_obs: {first_obs['action_mask']}")
        else:
            print("Action mask NOT in first_obs dictionary.")
    else:
        print(f"First observation (not a dict): {first_obs}")

    print(f"Info dict from first last(): {info}")
    if 'action_mask' in info:
        print(f"Action mask in info: {info['action_mask']}")
    else:
        print("Action mask NOT in info dictionary.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'env' in locals():
        env.close()