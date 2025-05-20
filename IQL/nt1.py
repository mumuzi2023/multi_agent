from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper  # Import the wrapper
from RL_brain_DQN_torch import DeepQNetwork
import csv
import numpy as np
import torch
import random
import gymnasium

# --- Configuration ---
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'pettingzoosislpursuitDQN_parallel_torch.csv'  # Changed log file name
MODEL_SAVE_PATH = "./tmp/dqn_pursuit_parallel_torch.ckpt"  # Changed model save path

# Learning hyperparameters
LEARNING_RATE = 0.0001
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 500  # pursuit_v4 default max_cycles
# NUM_PURSUERS = 8  # 可调参数: 追捕者数量 (pursuit_v4 默认是 8)
# NUM_EVADERS = 10  # 可调参数: 逃跑者数量 (pursuit_v4 默认是 10)

# --- Helper Functions --- (set_seeds and change_observation remain the same)
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
def run_pursuit_parallel():
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize AEC environment first
    aec_env = pursuit_v4.env(
        # n_pursuers=NUM_PURSUERS,  # 使用配置的数量
        # n_evaders=NUM_EVADERS,  # 使用配置的数量
        max_cycles=MAX_STEPS_PER_EPISODE
    )
    # Wrap it to become a Parallel environment
    env = aec_to_parallel_wrapper(aec_env)

    # Use one of the possible agents to check space structure
    # Note: After wrapping, env.possible_agents might be available directly from parallel_env
    # For consistency, we can check the underlying aec_env's space structure
    first_agent_for_space_check = env.possible_agents[0]
    # For parallel env, observation_space and action_space also take an agent ID
    agent_raw_obs_space = env.observation_space(first_agent_for_space_check)

    print(f"Detected observation space type for agent '{first_agent_for_space_check}': {type(agent_raw_obs_space)}")
    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    if is_dict_observation_space:
        print("Observation space is a Dict. Extracting 'observation' for pixel data.")
        pixel_obs_space = agent_raw_obs_space['observation']
    elif isinstance(agent_raw_obs_space, gymnasium.spaces.Box):
        print("Warning: Observation space is a Box, not a Dict. Assuming direct pixel data.")
        print("Action mask will be expected in the 'info' dictionary from env.step().")
        pixel_obs_space = agent_raw_obs_space
    else:
        env.close()
        raise TypeError(f"Unexpected observation space type: {type(agent_raw_obs_space)}")

    obs_shape = pixel_obs_space.shape
    N_FEATURES = np.prod(obs_shape)
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
        writer.writerow(["Episode", "AverageReward_DQN"])

    global_step_counter = 0

    print("Starting Training (Parallel API)...")
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)

        # Process initial observations
        current_processed_observations = {}
        for agent_id, raw_obs in current_observations_dict.items():
            current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

        episode_rewards_sum = {agent: 0.0 for agent in env.possible_agents}  # Use possible_agents for stable keys

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            if not env.agents:  # No active agents left (all terminated/truncated)
                break

            actions_to_take_dict = {}
            # For all currently active agents, choose an action
            for agent_id in env.agents:
                raw_obs_for_agent = current_observations_dict[agent_id]
                processed_obs_for_agent = current_processed_observations[agent_id]
                # Get agent-specific info from the dictionary returned by reset() or step()
                info_for_agent = current_infos_dict.get(agent_id, {})

                if is_dict_observation_space:
                    action_mask = raw_obs_for_agent['action_mask']
                else:  # Box observation, get mask from info
                    action_mask = info_for_agent.get('action_mask')
                    if action_mask is None:
                        # print(f"Warning: Action mask not found in 'info' dict for agent {agent_id} (obs type: Box). Defaulting to all actions valid.")
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)
                    elif not isinstance(action_mask, np.ndarray) or action_mask.shape != (N_ACTIONS,):
                        # print(f"Warning: Action mask from info is not as expected for agent {agent_id}: {action_mask}. Defaulting to all actions valid.")
                        action_mask = np.ones(N_ACTIONS, dtype=np.int8)

                actions_to_take_dict[agent_id] = RL.choose_action(processed_obs_for_agent, action_mask)

            # Step the environment with actions for all active agents
            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)

            # Store transitions for each agent that took an action
            for agent_id_acted in actions_to_take_dict.keys():
                s = current_processed_observations[agent_id_acted]
                a = actions_to_take_dict[agent_id_acted]
                r = rewards_dict.get(agent_id_acted, 0.0)
                terminated = terminated_dict.get(agent_id_acted, False)
                # truncated = truncated_dict.get(agent_id_acted, False) # For future use if needed

                if agent_id_acted in next_observations_dict:
                    s_prime = change_observation(next_observations_dict[agent_id_acted], is_dict_observation_space)
                else:  # Agent is done (terminated or truncated and removed from next_observations_dict)
                    s_prime = np.zeros_like(s)  # Dummy s_prime, as its value won't be used if done=True

                # The 'done' flag for DQN buffer should indicate if the episode ended for this agent at this step from its perspective.
                # 'terminated' is a good signal for this.
                RL.store_transition(s, a, r, s_prime, terminated)

                episode_rewards_sum[agent_id_acted] += r

            # Update current observations and infos for the next iteration
            current_observations_dict = next_observations_dict
            current_infos_dict = next_infos_dict
            current_processed_observations.clear()  # Clear and repopulate
            for agent_id, raw_obs in current_observations_dict.items():  # Only for agents present in next_observations_dict
                current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

            global_step_counter += 1  # Counts parallel environment steps
            if global_step_counter > WARMUP_STEPS and \
                    global_step_counter % LEARN_EVERY_STEPS == 0 and \
                    RL.memory_counter >= RL.batch_size:
                RL.learn()
        # --- End of Steps in Episode Loop ---

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
    print("Training finished (Parallel API).")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    run_pursuit_parallel()  # Changed function name for clarity