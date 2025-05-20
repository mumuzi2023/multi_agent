# 1. IMPORT YOUR CUSTOM ENVIRONMENT
# from pettingzoo.sisl import pursuit_v4 # Remove this
from cooperative_harvesting_env import \
    CooperativeHarvestingEnvAutoRespawn  # Assuming your file is cooperative_harvesting_env.py

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from RL_brain_DQN_torch import DeepQNetwork
import csv
import numpy as np
import torch
import random
import gymnasium

# --- Configuration ---
SEED = 1
NUM_EPISODES = 1000
# 2. UPDATE LOG AND MODEL PATHS (ADJUST AS NEEDED)
LOG_FILE = 'cooperative_harvesting_dqn_parallel_torch.csv'
MODEL_SAVE_PATH = "./tmp/dqn_harvesting_parallel_torch.ckpt"

# Learning hyperparameters
LEARNING_RATE = 0.00004
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 200  # Max cycles for your harvesting environment

# --- Configuration for CooperativeHarvestingEnvAutoRespawn ---
GRID_HEIGHT = 10
GRID_LENGTH = 10
NUM_HARVESTERS = 3
NUM_FRUITS = 5
OBSERVATION_RADIUS = 3  # Example, use float('inf') for global view or adjust


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
    if is_dict_observation_space:  # This will be FALSE for your custom env
        pixel_observation = observation_data['observation']
    else:  # This path will be taken
        pixel_observation = observation_data

    # Your observation is already flat and float32, but flatten() is harmless for 1D arrays.
    return pixel_observation.flatten().astype(np.float32)


# --- Main Execution ---
def run_custom_env_parallel():  # Renamed function for clarity
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. INITIALIZE YOUR CUSTOM AEC ENVIRONMENT
    aec_env = CooperativeHarvestingEnvAutoRespawn(
        grid_height=GRID_HEIGHT,
        grid_length=GRID_LENGTH,
        num_harvesters=NUM_HARVESTERS,
        num_fruits=NUM_FRUITS,
        max_cycles=MAX_STEPS_PER_EPISODE,
        observation_radius=OBSERVATION_RADIUS,
        render_mode=None  # No rendering during training script execution usually
    )
    env = aec_to_parallel_wrapper(aec_env)

    first_agent_for_space_check = env.possible_agents[0]
    agent_raw_obs_space = env.observation_space(first_agent_for_space_check)

    print(f"Detected observation space type for agent '{first_agent_for_space_check}': {type(agent_raw_obs_space)}")
    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    # 4. OBSERVATION SPACE HANDLING (Your env uses Box, so the 'elif' path will be taken)
    if is_dict_observation_space:
        print("Observation space is a Dict. Extracting 'observation' for pixel data.")
        pixel_obs_space = agent_raw_obs_space['observation']
    elif isinstance(agent_raw_obs_space, gymnasium.spaces.Box):
        print("Observation space is a Box. Assuming direct observation data.")
        # For your environment, action_mask is not provided in info.
        # The script will default to all actions being valid, which is fine.
        print("Action mask will default to all valid if not found in 'info' dict from env.step().")
        pixel_obs_space = agent_raw_obs_space
    else:
        env.close()
        raise TypeError(f"Unexpected observation space type: {type(agent_raw_obs_space)}")

    obs_shape = pixel_obs_space.shape
    # Your observation is already 1D, so np.prod might be slightly redundant but correct.
    N_FEATURES = np.prod(obs_shape) if obs_shape else 0  # Handle case where obs_shape might be None or empty
    if N_FEATURES == 0 and pixel_obs_space.shape is not None and len(
            pixel_obs_space.shape) > 0:  # If it's a Box (len > 0) but prod is 0
        N_FEATURES = pixel_obs_space.shape[0]  # For 1D Box space

    N_ACTIONS = env.action_space(first_agent_for_space_check).n

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
        writer.writerow(["Episode", "AverageReward_AllAgents"])  # Changed header for clarity

    global_step_counter = 0

    print("Starting Training (Parallel API with Custom Environment)...")
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)

        current_processed_observations = {}
        for agent_id, raw_obs in current_observations_dict.items():
            current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

        episode_rewards_sum = {agent: 0.0 for agent in env.possible_agents}

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            if not env.agents:
                break

            actions_to_take_dict = {}
            for agent_id in env.agents:
                # raw_obs_for_agent = current_observations_dict[agent_id] # Not directly used if not Dict obs
                processed_obs_for_agent = current_processed_observations[agent_id]
                info_for_agent = current_infos_dict.get(agent_id, {})

                if is_dict_observation_space:  # Will be FALSE for your custom env
                    action_mask = current_observations_dict[agent_id]['action_mask']
                else:  # This path will be taken
                    # Your custom environment does not provide 'action_mask' in 'info'.
                    # The script defaults to all actions being valid.
                    action_mask = info_for_agent.get('action_mask')
                    if action_mask is None:
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)
                    elif not isinstance(action_mask, np.ndarray) or action_mask.shape != (N_ACTIONS,):
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)

                # Ensure action_mask is on the correct device if RL.choose_action expects it
                # action_mask_tensor = torch.tensor(action_mask, device=device, dtype=torch.float32) # Or int8
                actions_to_take_dict[agent_id] = RL.choose_action(processed_obs_for_agent,
                                                                  action_mask)  # Pass np array mask

            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)

            for agent_id_acted in actions_to_take_dict.keys():
                s = current_processed_observations[agent_id_acted]
                a = actions_to_take_dict[agent_id_acted]
                r = rewards_dict.get(agent_id_acted, 0.0)
                terminated = terminated_dict.get(agent_id_acted, False)
                # truncated = truncated_dict.get(agent_id_acted, False) # For DQN, often terminated or truncated both mean "done"

                # Check if agent still exists in next_observations_dict
                if agent_id_acted in next_observations_dict:
                    s_prime = change_observation(next_observations_dict[agent_id_acted], is_dict_observation_space)
                else:  # Agent is done
                    s_prime = np.zeros_like(s)

                    # For DQN, usually "done" means the episode has ended for this agent from its perspective
                # This can be either due to termination or truncation.
                done_flag = terminated_dict.get(agent_id_acted, False) or truncated_dict.get(agent_id_acted, False)
                RL.store_transition(s, a, r, s_prime, done_flag)  # Use done_flag

                episode_rewards_sum[agent_id_acted] += r

            current_observations_dict = next_observations_dict
            current_infos_dict = next_infos_dict
            current_processed_observations.clear()
            for agent_id, raw_obs in current_observations_dict.items():
                current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

            global_step_counter += 1
            if global_step_counter > WARMUP_STEPS and \
                    global_step_counter % LEARN_EVERY_STEPS == 0 and \
                    RL.memory_counter >= RL.batch_size:
                RL.learn()

        num_total_agents = len(env.possible_agents)
        avg_episode_reward = sum(episode_rewards_sum.values()) / num_total_agents if num_total_agents > 0 else 0.0

        with open(LOG_FILE, 'a', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([episode + 1, avg_episode_reward])

        if (episode + 1) % 10 == 0:
            print(
                f"Episode: {episode + 1}/{NUM_EPISODES}, Avg Reward: {avg_episode_reward:.2f}, Epsilon: {RL.epsilon:.2f}, Global Steps: {global_step_counter}, Buffer: {RL.memory_counter}")
            if RL.cost_his:
                print(f"Last training cost: {RL.cost_his[-1]:.4f}")
            RL.save_model(MODEL_SAVE_PATH)

    env.close()
    RL.save_model(MODEL_SAVE_PATH)
    print("Training finished (Parallel API with Custom Environment).")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    run_custom_env_parallel()  # Call the renamed function