# Presumed start of your main script (e.g., main_iql.py)
from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from QNetwork import DeepQNetwork  # Assuming QNetwork.py is in the same directory orPYTHONPATH
import csv
import numpy as np
import torch
import random
import gymnasium

# Configure
SEED = 1
NUM_EPISODES = 1000
LOG_FILE = 'IQL_log1.csv'
MODEL_SAVE_PATH = "./tmp/IQL.ckpt"

# hyperparameters
LEARNING_RATE = 0.0001
REWARD_DECAY = 0.95
E_GREEDY_INITIAL = 0.9  # Prob of EXPLOITATION during training
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 500
LEARN_EVERY_STEPS = 4
MAX_STEPS_PER_EPISODE = 500


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



def evaluate_agent_performance(eval_env_parallel, rl_agent, is_dict_obs_space_flag, n_actions_param, eval_seed,
                               max_steps_eval=500):
    """
    Evaluates the agent's performance for one episode with exploration turned off.
    """
    current_observations_dict_eval, _ = eval_env_parallel.reset(seed=eval_seed)

    current_processed_observations_eval = {}
    # Store raw observations to access action_mask if needed from the Dict observation space
    raw_observations_for_eval_step = current_observations_dict_eval.copy()

    for agent_id, raw_obs in current_observations_dict_eval.items():
        current_processed_observations_eval[agent_id] = change_observation(raw_obs, is_dict_obs_space_flag)

    # Initialize rewards for all possible agents to ensure correct averaging
    episode_rewards_sum_eval = {agent_id: 0.0 for agent_id in eval_env_parallel.possible_agents}

    for _ in range(max_steps_eval):
        if not eval_env_parallel.agents:  # No active agents
            break

        actions_to_take_dict_eval = {}
        active_agents_in_step = list(eval_env_parallel.agents)

        for agent_id in active_agents_in_step:
            # Ensure agent is still active and has observations
            if agent_id not in current_processed_observations_eval:
                continue
            processed_obs_for_agent = current_processed_observations_eval[agent_id]

            action_mask_for_agent = np.ones(n_actions_param, dtype=np.int8)  # Default mask
            if is_dict_obs_space_flag:
                agent_raw_obs_at_step = raw_observations_for_eval_step.get(agent_id)
                if agent_raw_obs_at_step and 'action_mask' in agent_raw_obs_at_step:
                    action_mask_for_agent = agent_raw_obs_at_step['action_mask']

            actions_to_take_dict_eval[agent_id] = rl_agent.choose_action(
                processed_obs_for_agent,
                action_mask_for_agent,
                execution=True  # <<< Important: Turns off exploration
            )

        if not actions_to_take_dict_eval:  # No actions if all agents terminated previously
            break

        next_raw_observations_dict_eval, rewards_dict_eval, terminated_dict_eval, truncated_dict_eval, _ = eval_env_parallel.step(
            actions_to_take_dict_eval)

        for agent_id_acted in actions_to_take_dict_eval.keys():
            # Sum reward for the agent that acted, if it is a possible agent
            if agent_id_acted in episode_rewards_sum_eval:
                episode_rewards_sum_eval[agent_id_acted] += rewards_dict_eval.get(agent_id_acted, 0.0)

        current_processed_observations_eval.clear()
        raw_observations_for_eval_step = next_raw_observations_dict_eval.copy()

        for agent_id, raw_obs in next_raw_observations_dict_eval.items():
            current_processed_observations_eval[agent_id] = change_observation(raw_obs, is_dict_obs_space_flag)

    num_total_agents_eval = len(eval_env_parallel.possible_agents)
    avg_reward_eval = sum(
        episode_rewards_sum_eval.values()) / num_total_agents_eval if num_total_agents_eval > 0 else 0.0
    return avg_reward_eval



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

    first_agent_for_space_check = env.possible_agents[0]
    agent_raw_obs_space = env.observation_space(first_agent_for_space_check)

    is_dict_observation_space = isinstance(agent_raw_obs_space, gymnasium.spaces.Dict)

    if is_dict_observation_space:
        pixel_obs_space = agent_raw_obs_space['observation']
    else:
        # Fallback if observation space is not a Dict (e.g. a simple Box)
        # This might need adjustment if your env can have non-Dict, non-'observation' key structures
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
    # MODIFICATION: CSV header remains the same, but its meaning changes
    with open(LOG_FILE, 'w+', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(["Episode", "AverageReward_IQL"])  # This will now be no-exploration reward

    global_step_counter = 0

    print("======================= Training =======================")
    for episode in range(NUM_EPISODES):
        current_observations_dict, current_infos_dict = env.reset(seed=SEED + episode)

        current_processed_observations = {}
        # Store raw observations for training step to get action_mask if needed
        raw_observations_for_train_step = current_observations_dict.copy()

        for agent_id, raw_obs in current_observations_dict.items():
            current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

        # This sum is for rewards collected DURING training (with exploration)
        # We can still compute it for comparison or verbose logging if desired, but not logging to CSV.
        # episode_rewards_sum_exploratory = {agent: 0.0 for agent in env.possible_agents}

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            if not env.agents:  # If all agents are done
                break

            actions_to_take_dict = {}
            active_agents_in_training_step = list(env.agents)  # Get current list of active agents

            for agent_id in active_agents_in_training_step:
                if agent_id not in current_processed_observations:  # Agent might have terminated last step
                    continue
                processed_obs_for_agent = current_processed_observations[agent_id]

                action_mask = np.ones(N_ACTIONS, dtype=np.int8)  # Default mask
                if is_dict_observation_space:
                    agent_raw_obs_at_train_step = raw_observations_for_train_step.get(agent_id)
                    if agent_raw_obs_at_train_step and 'action_mask' in agent_raw_obs_at_train_step:
                        action_mask = agent_raw_obs_at_train_step['action_mask']

                # MODIFICATION: Ensure execution=False for training (to use RL.epsilon)
                actions_to_take_dict[agent_id] = RL.choose_action(processed_obs_for_agent, action_mask, execution=False)

            if not actions_to_take_dict:  # No actions if all agents terminated previously
                break

            next_observations_dict, rewards_dict, terminated_dict, truncated_dict, next_infos_dict = env.step(
                actions_to_take_dict)

            for agent_id_acted in actions_to_take_dict.keys():
                s = current_processed_observations[agent_id_acted]
                a = actions_to_take_dict[agent_id_acted]
                r = rewards_dict.get(agent_id_acted, 0.0)
                terminated = terminated_dict.get(agent_id_acted, False)  # Use .get for safety

                if agent_id_acted in next_observations_dict:
                    s_prime = change_observation(next_observations_dict[agent_id_acted], is_dict_observation_space)
                else:  # Agent terminated or is otherwise not in next_observations
                    s_prime = np.zeros_like(s)  # Use a zeroed state for s_prime

                RL.store_transition(s, a, r, s_prime, terminated)
                # if agent_id_acted in episode_rewards_sum_exploratory:
                #     episode_rewards_sum_exploratory[agent_id_acted] += r

            current_observations_dict = next_observations_dict
            raw_observations_for_train_step = current_observations_dict.copy()  # Update for next training step's mask retrieval
            current_processed_observations.clear()
            for agent_id, raw_obs in current_observations_dict.items():
                current_processed_observations[agent_id] = change_observation(raw_obs, is_dict_observation_space)

            global_step_counter += 1
            if global_step_counter > WARMUP_STEPS and \
                    global_step_counter % LEARN_EVERY_STEPS == 0 and \
                    RL.memory_counter >= RL.batch_size:
                RL.learn()

        # ==============================================================================
        # START OF MODIFICATIONS/ADDITIONS
        # ==============================================================================
        # End of training episode. Now, evaluate the agent's performance without exploration.
        eval_seed = SEED + episode + NUM_EPISODES  # Use a different seed for evaluation run
        avg_reward_no_exploration = evaluate_agent_performance(
            env,  # Use the same env object; it will be reset inside the function
            RL,
            is_dict_observation_space,
            N_ACTIONS,
            eval_seed,
            MAX_STEPS_PER_EPISODE  # Use same max steps for eval episodes
        )

        # Log the no-exploration reward
        with open(LOG_FILE, 'a', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow([episode + 1, avg_reward_no_exploration])

        if (episode + 1) % 10 == 0:
            # Optional: calculate exploratory reward for printing if you still want to see it
            # num_total_agents = len(env.possible_agents)
            # exploratory_reward_to_print = sum(episode_rewards_sum_exploratory.values()) / num_total_agents if num_total_agents > 0 else 0.0
            print(
                f"Episode: {episode + 1}/{NUM_EPISODES}, Avg Reward (No Explore): {avg_reward_no_exploration:.2f}, "
                # f"Exploratory Reward: {exploratory_reward_to_print:.2f}, " # Uncomment if you calculate and want to print
                f"Epsilon (Exploit Prob): {RL.epsilon:.2f}, Global Steps: {global_step_counter}, Buffer: {RL.memory_counter}"
            )
            if RL.cost_his:  # Check if cost_his is not empty
                print(f"Last training cost: {RL.cost_his[-1]:.4f}")
            RL.save_model(MODEL_SAVE_PATH)
        # ==============================================================================
        # END OF MODIFICATIONS/ADDITIONS
        # ==============================================================================

    env.close()
    RL.save_model(MODEL_SAVE_PATH)
    print("========================Training finished========================")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    run_pursuit_parallel()