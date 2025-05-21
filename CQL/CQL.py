from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from QNetwork_CQL import DeepQNetwork
import csv
import numpy as np
import torch
import random
import gymnasium

# Configure
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'CQL_log.csv'
MODEL_SAVE_PATH = "./tmp/CQL.ckpt"

# hyperparameters
LEARNING_RATE = 0.0003
REWARD_DECAY = 0.99
E_GREEDY_INITIAL = 0.9
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 500
SAVE_MODEL_FREQUENCY = 10
# NUM_PURSUERS = 8
# NUM_EVADERS = 10

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


def run_pursuit_parallel():
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    aec_env = pursuit_v4.env(
        # n_pursuers=NUM_PURSUERS,
        # n_evaders=NUM_EVADERS,
        max_cycles=MAX_STEPS_PER_EPISODE
    )
    env = aec_to_parallel_wrapper(aec_env)

    ORDERED_POSSIBLE_AGENTS = sorted(list(env.possible_agents))
    print(f"Agents: {ORDERED_POSSIBLE_AGENTS}")

    first_agent_id_for_space = ORDERED_POSSIBLE_AGENTS[0]
    agent_raw_obs_space = env.observation_space(first_agent_id_for_space)
    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    pixel_obs_space = agent_raw_obs_space
    single_agent_n_features = np.prod(pixel_obs_space.shape)
    TOTAL_N_FEATURES = len(ORDERED_POSSIBLE_AGENTS) * single_agent_n_features
    N_ACTIONS = env.action_space(first_agent_id_for_space).n

    print(f"Number of features for an agent: {single_agent_n_features}")
    print(f"Features in total: {TOTAL_N_FEATURES}")
    print(f"Actions: {N_ACTIONS}")

    RL = DeepQNetwork(
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
    with open(LOG_FILE, 'w+', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(["Episode", "AverageReward_CQL"])

    global_step_counter = 0

    print("======================= Training =======================")
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)
        # Init observations
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
                action_mask = np.ones(N_ACTIONS, dtype=np.int8)
                actions_to_take_dict[agent_id] = RL.choose_action(current_joint_state, action_mask)

            # Act
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

            global_step_counter += 1
            if global_step_counter > WARMUP_STEPS and  global_step_counter % LEARN_EVERY_STEPS == 0 and RL.memory_counter >= RL.batch_size:
                RL.learn()

        num_total_possible_agents = len(env.possible_agents)
        avg_episode_reward = sum(
            episode_rewards_sum.values()) / num_total_possible_agents if num_total_possible_agents > 0 else 0.0

        with open(LOG_FILE, 'a', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([episode + 1, avg_episode_reward])

        if (episode + 1) % SAVE_MODEL_FREQUENCY == 0:
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