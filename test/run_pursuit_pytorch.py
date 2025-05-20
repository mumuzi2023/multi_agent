from pettingzoo.sisl import pursuit_v4
from RL_brain_DQN_torch import DeepQNetwork  # Assuming the class is in this file
import csv
import numpy as np
import torch
import random
import gymnasium  # For type checking with gymnasium.spaces

# --- Configuration ---
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'pettingzoosislpursuitDQN_torch.csv'
MODEL_SAVE_PATH = "./tmp/dqn_pursuit_torch.ckpt"

# Learning hyperparameters
LEARNING_RATE = 0.001
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4


# --- Helper Functions ---
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


# --- Main Execution ---
def run_pursuit():
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = pursuit_v4.env(max_cycles=500)

    first_agent_for_space_check = env.possible_agents[0]  # Used to determine space structure
    agent_raw_obs_space = env.observation_space(first_agent_for_space_check)

    print(f"Detected observation space type for agent '{first_agent_for_space_check}': {type(agent_raw_obs_space)}")
    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    if is_dict_observation_space:
        print("Observation space is a Dict. Extracting 'observation' for pixel data.")
        pixel_obs_space = agent_raw_obs_space['observation']
    elif isinstance(agent_raw_obs_space, gymnasium.spaces.Box):
        print("Warning: Observation space is a Box, not a Dict. Assuming direct pixel data.")
        print("Action mask will be expected in the 'info' dictionary from env.last().")
        pixel_obs_space = agent_raw_obs_space
    else:
        env.close()  # Close env before raising error
        raise TypeError(f"Unexpected observation space type: {type(agent_raw_obs_space)}")

    obs_shape = pixel_obs_space.shape
    N_FEATURES = np.prod(obs_shape)
    N_ACTIONS = env.action_space(first_agent_for_space_check).n  # Action space usually same for all pursuers

    print(f"Number of features (flattened observation): {N_FEATURES}")
    print(f"Number of actions: {N_ACTIONS}")

    RL = DeepQNetwork(
        n_actions=N_ACTIONS,
        n_features=N_FEATURES,
        learning_rate=LEARNING_RATE,
        reward_decay=REWARD_DECAY,
        e_greedy=E_GREEDY_INITIAL,
        replace_target_iter=REPLACE_TARGET_ITER,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        device=device
    )
    if N_FEATURES == 0:
        print("Error: N_FEATURES is 0. DQN cannot be initialized properly.")
        env.close()
        return

    with open(LOG_FILE, 'w+', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(["Episode", "AverageReward_DQN"])

    global_step_counter = 0

    print("Starting Training...")
    for episode in range(NUM_EPISODES):
        env.reset(seed=SEED + episode)  # env.reset() returns None. Sets up internal state for first agent.

        # Buffers for storing (s,a) for which reward and s' are pending for each agent
        s_for_last_action = {agent: None for agent in env.possible_agents}
        last_action_taken = {agent: None for agent in env.possible_agents}

        episode_rewards_sum = {agent: 0.0 for agent in env.possible_agents}

        for agent_to_act in env.agent_iter():  # agent_iter will yield agents one by one
            current_observation_full, reward, terminated, truncated, info = env.last()
            done = terminated or truncated

            episode_rewards_sum[agent_to_act] += reward

            if s_for_last_action[agent_to_act] is not None and \
                    last_action_taken[agent_to_act] is not None:
                s_prime_flat = change_observation(current_observation_full, is_dict_observation_space)

                RL.store_transition(
                    s_for_last_action[agent_to_act],
                    last_action_taken[agent_to_act],
                    reward,
                    s_prime_flat,
                    terminated
                )

            if done:
                action_to_take = None
                s_for_last_action[agent_to_act] = None
                last_action_taken[agent_to_act] = None
            else:
                flat_obs_for_action = change_observation(current_observation_full, is_dict_observation_space)

                if is_dict_observation_space:  # 这个分支在您的情况下不会被执行
                    action_mask = current_observation_full['action_mask']
                else:  # Observation is raw pixel data (Box space), action_mask should be in info
                    action_mask = info.get('action_mask')
                    if action_mask is None:
                        print(
                            f"Warning: Action mask not found in 'info' dict for agent {agent_to_act} (obs type: Box). Defaulting to all actions valid.")
                        # ---- START DEBUG PRINT ----
                        # 打印 info 字典的内容，看看里面有什么
                        print(f"DEBUG: Info dict for agent {agent_to_act} when action_mask is None: {info}")
                        # ---- END DEBUG PRINT ----
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)  # Fallback
                    elif not isinstance(action_mask, np.ndarray) or action_mask.shape != (N_ACTIONS,):
                        print(
                            f"Warning: Action mask from info is not as expected: {action_mask}. Defaulting to all actions valid.")
                        # ---- START DEBUG PRINT ----
                        # 如果找到了 'action_mask' 但格式不对，也打印出来看看
                        print(f"DEBUG: Info dict for agent {agent_to_act} when action_mask is malformed: {info}")
                        print(f"DEBUG: Malformed action_mask received: {action_mask}")
                        # ---- END DEBUG PRINT ----
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)  # Fallback

                action_to_take = RL.choose_action(flat_obs_for_action, action_mask)

                s_for_last_action[agent_to_act] = flat_obs_for_action
                last_action_taken[agent_to_act] = action_to_take

            env.step(action_to_take)

            global_step_counter += 1
            if global_step_counter > WARMUP_STEPS and \
                    global_step_counter % LEARN_EVERY_STEPS == 0 and \
                    RL.memory_counter >= RL.batch_size:  # Use >= for safety, > is also fine
                RL.learn()
        # --- End of Agent Iteration Loop (Episode End) ---

        avg_episode_reward = sum(episode_rewards_sum.values()) / len(env.possible_agents)

        with open(LOG_FILE, 'a', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([episode + 1, avg_episode_reward])

        if (episode + 1) % 10 == 0:
            print(
                f"Episode: {episode + 1}/{NUM_EPISODES}, Avg Reward: {avg_episode_reward:.2f}, Epsilon: {RL.epsilon:.2f}, Global Steps: {global_step_counter}, Buffer: {RL.memory_counter}")
            if RL.cost_his:  # Check if cost_his has any values
                print(f"Last training cost: {RL.cost_his[-1]:.4f}")
            RL.save_model(MODEL_SAVE_PATH)

    env.close()
    RL.save_model(MODEL_SAVE_PATH)
    print("Training finished.")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    run_pursuit()