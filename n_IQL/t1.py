from cooperative_harvesting_env import CooperativeHarvestingEnvAutoRespawn
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from QNetwork import DeepQNetwork
import csv
import numpy as np
import torch
import random
import gymnasium

# Configure
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'IQL_log.csv'
MODEL_SAVE_PATH = "./tmp/IQL.ckpt"

# hyperparameters
LEARNING_RATE = 0.0001
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 200
GRID_HEIGHT = 10
GRID_LENGTH = 10
NUM_HARVESTERS = 3
NUM_FRUITS = 5
OBSERVATION_RADIUS = 3

def set_seeds(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def change_observation(observation_data, is_dict_observation_space):
    pixel_observation = observation_data
    return pixel_observation.flatten().astype(np.float32)


def run_pursuit_parallel():
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    pixel_obs_space = agent_raw_obs_space

    obs_shape = pixel_obs_space.shape
    N_FEATURES = np.prod(obs_shape)
    N_ACTIONS = env.action_space(first_agent_for_space_check).n

    print(f"Number of features: {N_FEATURES}")
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
    with open(LOG_FILE, 'w+', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(["Episode", "AverageReward_IQL"])

    global_step_counter = 0

    print("======================= Training =======================")
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)
        # Init observations
        current_processed_observations = {}
        for agent_id, raw_obs in current_observations_dict.items():
            current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

        episode_rewards_sum = {agent: 0.0 for agent in env.possible_agents}

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            if not env.agents:
                break

            actions_to_take_dict = {}
            for agent_id in env.agents:
                processed_obs_for_agent = current_processed_observations[agent_id]
                action_mask = np.ones(N_ACTIONS, dtype=np.int8)
                actions_to_take_dict[agent_id] = RL.choose_action(processed_obs_for_agent,action_mask)

            # Act
            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)

            # Store transitions
            for agent_id_acted in actions_to_take_dict.keys():
                s = current_processed_observations[agent_id_acted]
                a = actions_to_take_dict[agent_id_acted]
                r = rewards_dict.get(agent_id_acted, 0.0)
                terminated = terminated_dict.get(agent_id_acted, False) or truncated_dict.get(agent_id_acted, False)
                if agent_id_acted in next_observations_dict:
                    s_prime = change_observation(next_observations_dict[agent_id_acted], is_dict_observation_space)
                else:
                    s_prime = np.zeros_like(s)

                RL.store_transition(s, a, r, s_prime, terminated)

                episode_rewards_sum[agent_id_acted] += r

            current_observations_dict = next_observations_dict
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
    print("========================Training finished========================")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    run_pursuit_parallel()