import functools
import gymnasium
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType, ActionType

try:
    import pygame
except ImportError:
    pygame = None


class CooperativeHarvestingEnvAutoRespawn(AECEnv):
    metadata = {
        "name": "cooperative_harvesting_v2.4_square_obs_viz",  # Updated name
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, grid_height=10, grid_length=10, num_harvesters=3,
                 num_fruits=5,
                 max_cycles=500,
                 render_mode=None,
                 cell_size=30,
                 observation_radius=float('inf')):
        super().__init__()

        assert num_harvesters > 0, "Number of harvesters must be positive."
        assert num_fruits >= 0, "Number of fruits must be non-negative."
        assert observation_radius >= 0, "Observation radius must be non-negative."

        self.grid_height = grid_height
        self.grid_length = grid_length
        self.num_harvesters = num_harvesters
        self.initial_num_fruits = num_fruits
        self.observation_radius = observation_radius

        self._harvester_levels_initial_list = [np.random.randint(1, 3 + 1) for _ in range(self.num_harvesters)]
        if self.initial_num_fruits == 0:
            self._fruit_levels_initial_list = []
        else:
            self._fruit_levels_initial_list = [np.random.randint(1, 4 + 1) for _ in range(self.initial_num_fruits)]

        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.cell_size = cell_size

        self.possible_agents = [f"harvester_{i}" for i in range(num_harvesters)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        self._agent_selector = agent_selector.agent_selector(self.possible_agents)

        self.harvester_positions = {}
        self.harvester_levels = {}
        self.fruits_data = []
        self.next_fruit_id_counter = 0

        self.window = None
        self.clock = None
        self.font = None

        max_dim = max(self.grid_height, self.grid_length)
        max_h_level = 3
        max_f_level = 4
        # For min/max_obs_val, consider that relative positions can go up to observation_radius or grid dimension
        effective_max_coord_diff = max_dim
        if self.observation_radius != float('inf'):
            effective_max_coord_diff = max(max_dim, self.observation_radius)

        max_abs_obs_coord = effective_max_coord_diff  # Max absolute value for coordinates or relative positions

        self.max_obs_val = float(max(max_abs_obs_coord, max_h_level, max_f_level))
        self.min_obs_val = float(-max_abs_obs_coord)  # Relative positions can be negative

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id: str) -> gymnasium.spaces.Space:
        obs_dim = (2 + 1 +
                   self.initial_num_fruits * 4 +
                   (self.num_harvesters - 1) * 3)
        return gymnasium.spaces.Box(low=self.min_obs_val, high=self.max_obs_val,
                                    shape=(obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id: str) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(5)

    def _place_entities(self):
        all_coords = [(c, r) for r in range(self.grid_height) for c in range(self.grid_length)]
        np.random.shuffle(all_coords)
        coord_idx = 0
        self.harvester_positions = {}
        for i, agent_id in enumerate(self.possible_agents):
            if coord_idx < len(all_coords):
                self.harvester_positions[agent_id] = all_coords[coord_idx]; coord_idx += 1
            else:
                self.harvester_positions[agent_id] = (
                np.random.randint(self.grid_length), np.random.randint(self.grid_height))
        self.fruits_data = []
        for i in range(self.initial_num_fruits):
            if coord_idx < len(all_coords):
                pos = all_coords[coord_idx]; coord_idx += 1
            else:
                pos = (np.random.randint(self.grid_length), np.random.randint(self.grid_height))
            level = self._fruit_levels_initial_list[i]
            self.fruits_data.append({"id": f"fruit_{i}", "pos": pos, "level": level, "status": "available"})
        self.next_fruit_id_counter = self.initial_num_fruits

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents};
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents};
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._harvester_levels_initial_list = [np.random.randint(1, 3 + 1) for _ in range(self.num_harvesters)]
        if self.initial_num_fruits == 0:
            self._fruit_levels_initial_list = []
        else:
            self._fruit_levels_initial_list = [np.random.randint(1, 4 + 1) for _ in range(self.initial_num_fruits)]
        self.harvester_levels = {agent: self._harvester_levels_initial_list[self.agent_name_mapping[agent]] for agent in
                                 self.possible_agents}
        self._place_entities()
        self.next_fruit_id_counter = self.initial_num_fruits
        self.num_cycles = 0
        self._agent_selector.reinit(self.agents);
        self.agent_selection = self._agent_selector.next()

    def _respawn_fruit_from_harvested(self, harvested_fruit_id_to_remove):
        # ... (no changes) ...
        found_fruit_obj = next((f for f in self.fruits_data if f["id"] == harvested_fruit_id_to_remove), None)
        if found_fruit_obj:
            self.fruits_data.remove(found_fruit_obj)
        else:
            return
        if self.initial_num_fruits == 0: return
        new_level = np.random.randint(1, 4 + 1)
        occupied_coords = set(self.harvester_positions.values()) | set(f["pos"] for f in self.fruits_data)
        available_coords = [(c, r) for r in range(self.grid_height) for c in range(self.grid_length) if
                            (c, r) not in occupied_coords]
        new_pos = available_coords[np.random.choice(len(available_coords))] if available_coords else (
        np.random.randint(self.grid_length), np.random.randint(self.grid_height))
        new_fruit_id = f"fruit_{self.next_fruit_id_counter}";
        self.next_fruit_id_counter += 1
        self.fruits_data.append({"id": new_fruit_id, "pos": new_pos, "level": new_level, "status": "available"})

    def _check_and_process_automatic_harvests(self):
        # ... (no changes related to observation radius, harvest condition is adjacency) ...
        harvests_to_process = []
        for fruit_obj in self.fruits_data:
            if fruit_obj["status"] != "available": continue
            harvesters_near_or_on_fruit = []
            fruit_pos_x, fruit_pos_y = fruit_obj["pos"]
            for agent_id, harvester_pos_tuple in self.harvester_positions.items():
                if not (self.terminations.get(agent_id, False) or self.truncations.get(agent_id, False)):
                    harvester_pos_x, harvester_pos_y = harvester_pos_tuple
                    if abs(harvester_pos_x - fruit_pos_x) + abs(harvester_pos_y - fruit_pos_y) <= 1:
                        harvesters_near_or_on_fruit.append(agent_id)
            if not harvesters_near_or_on_fruit: continue
            solo_h_id = next(
                (h_id for h_id in harvesters_near_or_on_fruit if self.harvester_levels[h_id] >= fruit_obj["level"]),
                None)
            if solo_h_id:
                if not any(h["fruit_id"] == fruit_obj["id"] for h in harvests_to_process): harvests_to_process.append(
                    {"fruit_id": fruit_obj["id"], "harvesters": [solo_h_id], "type": "solo"})
                continue
            if sum(self.harvester_levels[h_id] for h_id in harvesters_near_or_on_fruit) >= fruit_obj["level"]:
                if not any(h["fruit_id"] == fruit_obj["id"] for h in harvests_to_process): harvests_to_process.append(
                    {"fruit_id": fruit_obj["id"], "harvesters": list(harvesters_near_or_on_fruit), "type": "coop"})
        for h_info in harvests_to_process:
            target_fruit = next((f for f in self.fruits_data if f["id"] == h_info["fruit_id"]), None)
            if target_fruit and target_fruit["status"] == "available":
                target_fruit["status"] = "harvested_processing"
                if h_info["type"] == "solo":
                    self.rewards[h_info["harvesters"][0]] = self.rewards.get(h_info["harvesters"][0], 0.0) + 10.0
                elif h_info["type"] == "coop":
                    rew = 8.0 / len(h_info["harvesters"]) if h_info["harvesters"] else 0.0
                    for h_id in h_info["harvesters"]: self.rewards[h_id] = self.rewards.get(h_id, 0.0) + rew
                self._respawn_fruit_from_harvested(target_fruit["id"])

    def step(self, action: ActionType):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]: self._was_dead_step(action); return
        for r_agent in self.possible_agents: self.rewards[r_agent] = 0.0
        current_pos = self.harvester_positions[agent];
        step_reward = -0.1
        new_pos_x, new_pos_y = current_pos[0], current_pos[1]
        if action == 0:
            new_pos_y = max(0, current_pos[1] - 1)
        elif action == 1:
            new_pos_y = min(self.grid_height - 1, current_pos[1] + 1)
        elif action == 2:
            new_pos_x = max(0, current_pos[0] - 1)
        elif action == 3:
            new_pos_x = min(self.grid_length - 1, current_pos[0] + 1)
        self.harvester_positions[agent] = (new_pos_x, new_pos_y);
        self.rewards[agent] += step_reward
        self._check_and_process_automatic_harvests()
        if self._agent_selector.is_last(): self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            for ag in self.possible_agents: self.truncations[ag] = True
        self.agent_selection = self._agent_selector.next()
        if self.render_mode == "human": self.render()

    def observe(self, agent_id: str) -> ObsType:
        obs_list = []
        agent_pos_tuple = self.harvester_positions[agent_id]
        agent_x, agent_y = agent_pos_tuple
        agent_level = self.harvester_levels[agent_id]

        obs_list.extend([float(agent_x), float(agent_y), float(agent_level)])

        # Fruit information
        for i in range(self.initial_num_fruits):
            if i < len(self.fruits_data):
                fruit = self.fruits_data[i]
                fruit_x_coord, fruit_y_coord = fruit["pos"]
                # Use Chebyshev distance (square range)
                chebyshev_distance_to_fruit = max(abs(fruit_x_coord - agent_x), abs(fruit_y_coord - agent_y))

                if chebyshev_distance_to_fruit <= self.observation_radius:
                    rel_pos_x = float(fruit_x_coord - agent_x)
                    rel_pos_y = float(fruit_y_coord - agent_y)
                    status_numeric = 1.0 if fruit["status"] == "available" else 0.0
                    obs_list.extend([rel_pos_x, rel_pos_y, float(fruit["level"]), status_numeric])
                else:
                    obs_list.extend([0.0, 0.0, 0.0, 0.0])  # Placeholder
            else:
                obs_list.extend([0.0, 0.0, 0.0, 0.0])  # Placeholder

        # Other harvester information
        for other_agent_id_val in self.possible_agents:
            if other_agent_id_val == agent_id: continue
            other_harvester_pos_tuple = self.harvester_positions[other_agent_id_val]
            other_x, other_y = other_harvester_pos_tuple
            # Use Chebyshev distance (square range)
            chebyshev_distance_to_other = max(abs(other_x - agent_x), abs(other_y - agent_y))

            if chebyshev_distance_to_other <= self.observation_radius:
                other_level_val = self.harvester_levels[other_agent_id_val]
                rel_pos_x = float(other_x - agent_x)
                rel_pos_y = float(other_y - agent_y)
                obs_list.extend([rel_pos_x, rel_pos_y, float(other_level_val)])
            else:
                obs_list.extend([0.0, 0.0, 0.0])  # Placeholder

        expected_obs_dim = self.observation_space(agent_id).shape[0]
        current_obs_len = len(obs_list)
        if current_obs_len != expected_obs_dim:
            if current_obs_len < expected_obs_dim:
                obs_list.extend([0.0] * (expected_obs_dim - current_obs_len))
            else:
                obs_list = obs_list[:expected_obs_dim]
        return np.array(obs_list, dtype=np.float32)

    def render(self):
        if self.render_mode is None: return
        if pygame is None and self.render_mode == "human": raise ImportError("Pygame not installed.")
        if self.render_mode == "human" and self.window is None:
            pygame.init();
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_length * self.cell_size, self.grid_height * self.cell_size))
            pygame.display.set_caption(self.metadata["name"]);
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.Font(None, int(self.cell_size * 0.5))
            except pygame.error:
                self.font = pygame.font.SysFont("arial", int(self.cell_size * 0.5))

        canvas = pygame.Surface((self.grid_length * self.cell_size, self.grid_height * self.cell_size))
        canvas.fill((255, 255, 255))  # White background

        # 1. Draw grid lines
        for r in range(self.grid_height + 1): pygame.draw.line(canvas, (200, 200, 200), (0, r * self.cell_size),
                                                               (self.grid_length * self.cell_size, r * self.cell_size))
        for c in range(self.grid_length + 1): pygame.draw.line(canvas, (200, 200, 200), (c * self.cell_size, 0),
                                                               (c * self.cell_size, self.grid_height * self.cell_size))

        # 2. Prepare to draw all observation square fills
        overlay_surface = None
        if self.render_mode == "human" and self.observation_radius != float('inf'):
            overlay_surface = pygame.Surface((self.grid_length * self.cell_size, self.grid_height * self.cell_size),
                                             pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 0))  # Fill with fully transparent
            obs_area_fill_color = (100, 100, 255, 30)  # Light blue, very semi-transparent for fill

            for agent_id_to_visualize in self.possible_agents:
                if agent_id_to_visualize in self.harvester_positions and \
                        not (self.terminations.get(agent_id_to_visualize, True) or self.truncations.get(
                            agent_id_to_visualize, True)):

                    agent_pos_x, agent_pos_y = self.harvester_positions[agent_id_to_visualize]
                    R = self.observation_radius

                    # Iterate over cells in this agent's observation square
                    # Clamped to grid boundaries for iteration
                    min_r_clamped = max(0, agent_pos_y - R)
                    max_r_clamped = min(self.grid_height - 1, agent_pos_y + R)
                    min_c_clamped = max(0, agent_pos_x - R)
                    max_c_clamped = min(self.grid_length - 1, agent_pos_x + R)

                    for r_idx in range(int(min_r_clamped), int(max_r_clamped) + 1):
                        for c_idx in range(int(min_c_clamped), int(max_c_clamped) + 1):
                            # Check if this cell (c_idx, r_idx) is TRULY within the Chebyshev distance for this agent
                            if max(abs(c_idx - agent_pos_x), abs(r_idx - agent_pos_y)) <= R:
                                cell_rect = pygame.Rect(c_idx * self.cell_size,
                                                        r_idx * self.cell_size,
                                                        self.cell_size,
                                                        self.cell_size)
                                pygame.draw.rect(overlay_surface, obs_area_fill_color, cell_rect)

            canvas.blit(overlay_surface, (0, 0))  # Blit all fills at once

        # 3. Draw all observation square outlines
        if self.render_mode == "human" and self.observation_radius != float('inf'):
            obs_area_outline_color = (40, 40, 100, 150)  # Darker, slightly more opaque blue for outline

            for agent_id_to_visualize in self.possible_agents:
                if agent_id_to_visualize in self.harvester_positions and \
                        not (self.terminations.get(agent_id_to_visualize, True) or self.truncations.get(
                            agent_id_to_visualize, True)):

                    agent_pos_x, agent_pos_y = self.harvester_positions[agent_id_to_visualize]
                    R = self.observation_radius

                    # Clamped grid coordinates for the outline rectangle
                    outline_start_c = max(0, agent_pos_x - R)
                    outline_end_c = min(self.grid_length - 1, agent_pos_x + R)
                    outline_start_r = max(0, agent_pos_y - R)
                    outline_end_r = min(self.grid_height - 1, agent_pos_y + R)

                    if outline_start_c <= outline_end_c and outline_start_r <= outline_end_r:  # Check if visible part
                        outline_rect_pixel = pygame.Rect(
                            outline_start_c * self.cell_size,
                            outline_start_r * self.cell_size,
                            (outline_end_c - outline_start_c + 1) * self.cell_size,
                            (outline_end_r - outline_start_r + 1) * self.cell_size
                        )
                        pygame.draw.rect(canvas, obs_area_outline_color, outline_rect_pixel, width=1)  # Thin outline

        # 4. Draw fruits
        for fruit in self.fruits_data:
            fruit_color = (0, 180, 0) if fruit["status"] == "available" else (100, 100, 100)
            cx = int((fruit["pos"][0] + 0.5) * self.cell_size);
            cy = int((fruit["pos"][1] + 0.5) * self.cell_size)
            pygame.draw.circle(canvas, fruit_color, (cx, cy), int(self.cell_size * 0.30))  # Slightly smaller fruits
            lvl_txt = self.font.render(str(fruit["level"]), True, (0, 0, 0))
            canvas.blit(lvl_txt, (cx - lvl_txt.get_width() // 2,
                                  cy - int(self.cell_size * 0.35) - lvl_txt.get_height() // 2))  # Adjusted text pos

        # 5. Draw harvesters
        for agent_id_render in self.possible_agents:
            if agent_id_render not in self.harvester_positions: continue
            is_term = self.terminations.get(agent_id_render, False);
            is_trun = self.truncations.get(agent_id_render, False)
            pos = self.harvester_positions[agent_id_render];
            color = (255, 0, 0)
            if is_term or is_trun: color = (150, 0, 0)  # Darker red if done

            # Highlight current acting agent slightly differently if desired, e.g. brighter outline or small mark
            # For now, all active are bright red, inactive are dark red.
            if agent_id_render == self.agent_selection and not (is_term or is_trun):
                color = (255, 50, 50)  # Slightly brighter red for current agent

            rect = pygame.Rect(pos[0] * self.cell_size + 2, pos[1] * self.cell_size + 2, self.cell_size - 4,
                               self.cell_size - 4)
            pygame.draw.rect(canvas, color, rect, border_radius=3)
            lvl_val = self.harvester_levels.get(agent_id_render, "X")
            lvl_txt = self.font.render(str(lvl_val), True, (255, 255, 255))
            canvas.blit(lvl_txt,
                        (rect.centerx - lvl_txt.get_width() // 2, rect.centery - lvl_txt.get_height() // 2 - 5))

        # Finalize display
        if self.render_mode == "human":
            if self.window: self.window.blit(canvas, (0, 0)); pygame.display.flip(); pygame.event.pump()
            if self.clock: self.clock.tick(10)  # Control FPS
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            if pygame and pygame.display.get_init(): pygame.display.quit()
            if pygame and pygame.get_init(): pygame.quit()
            self.window = None;
            self.clock = None;
            self.font = None


if __name__ == "__main__":
    env_config = {
        "grid_height": 10, "grid_length": 10,
        "num_harvesters": 2, "num_fruits": 4,
        "max_cycles": 150,
        "render_mode": "human",
        "cell_size": 35,
        "observation_radius": 2  # Example: Square radius of 2 (total 5x5 area)
    }

    env = CooperativeHarvestingEnvAutoRespawn(**env_config)
    print(f"Observation radius: {env.observation_radius} (Chebyshev distance)")
    print(f"Observation space for harvester_0: {env.observation_space('harvester_0')}")

    max_episodes = 3
    for episode_count in range(max_episodes):
        print(f"\n--- Episode {episode_count + 1} ---")
        env.reset(seed=np.random.randint(100000) + episode_count)
        for agent_step_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent_step_id).sample()
            env.step(action)
        if env.render_mode == "human": env.render()  # Render final state too for clarity
        print(f"Episode {episode_count + 1} finished after {env.num_cycles} cycles.")
        print(f"Cumulative rewards for episode: {env._cumulative_rewards}")
    env.close();
    print("\nEnvironment closed.")