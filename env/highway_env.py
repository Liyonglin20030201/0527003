import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HighwayDrivingEnv(gym.Env):
    """
    Multi-lane highway driving environment for lane keeping and overtaking.

    Road: 3 lanes, scrolling vertically.
    Ego vehicle: controlled by the agent.
    NPC vehicles: move at varying speeds, serve as obstacles to overtake.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    LANE_WIDTH = 1.0
    NUM_LANES = 3
    ROAD_LENGTH = 200.0
    MAX_SPEED = 8.0
    MIN_SPEED = 1.0
    NPC_COUNT = 6
    VISION_RANGE = 40.0

    def __init__(self, render_mode=None, style="balanced"):
        super().__init__()
        self.render_mode = render_mode
        self.style = style

        # Actions: 0=keep, 1=accelerate, 2=brake, 3=lane_left, 4=lane_right
        self.action_space = spaces.Discrete(5)

        # Observation: [ego_lane, ego_speed, ego_y,
        #               npc1_rel_lane, npc1_rel_y, npc1_speed,
        #               npc2_rel_lane, npc2_rel_y, npc2_speed,
        #               npc3_rel_lane, npc3_rel_y, npc3_speed,
        #               npc4_rel_lane, npc4_rel_y, npc4_speed]
        # 3 ego features + 4 nearest NPCs * 3 features = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.ego = None
        self.npcs = None
        self.steps = 0
        self.max_steps = 500
        self.total_reward = 0.0
        self.overtake_count = 0
        self.collision_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.total_reward = 0.0
        self.overtake_count = 0
        self.collision_count = 0

        self.ego = {
            "lane": 1,
            "y": 0.0,
            "speed": 4.0,
            "x": self._lane_to_x(1),
        }

        self.npcs = []
        for i in range(self.NPC_COUNT):
            lane = self.np_random.integers(0, self.NUM_LANES)
            y_offset = self.np_random.uniform(15, 80) * (1 if i < self.NPC_COUNT // 2 else -1)
            speed = self.np_random.uniform(2.0, 5.0)
            self.npcs.append({
                "lane": int(lane),
                "y": self.ego["y"] + y_offset,
                "speed": speed,
                "x": self._lane_to_x(int(lane)),
                "overtaken": False,
            })

        self._passed_npcs = set()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # Execute action
        if action == 1:  # accelerate
            self.ego["speed"] = min(self.ego["speed"] + 0.5, self.MAX_SPEED)
        elif action == 2:  # brake
            self.ego["speed"] = max(self.ego["speed"] - 0.5, self.MIN_SPEED)
        elif action == 3:  # lane left
            if self.ego["lane"] > 0:
                self.ego["lane"] -= 1
                self.ego["x"] = self._lane_to_x(self.ego["lane"])
        elif action == 4:  # lane right
            if self.ego["lane"] < self.NUM_LANES - 1:
                self.ego["lane"] += 1
                self.ego["x"] = self._lane_to_x(self.ego["lane"])

        # Move ego
        self.ego["y"] += self.ego["speed"]

        # Move NPCs
        for npc in self.npcs:
            npc["y"] += npc["speed"]

        # Check overtaking
        for i, npc in enumerate(self.npcs):
            if i not in self._passed_npcs:
                if self.ego["y"] > npc["y"] + 2.0:
                    self._passed_npcs.add(i)
                    self.overtake_count += 1
                    reward += 5.0

        # Check collision
        collision = False
        for npc in self.npcs:
            if (self.ego["lane"] == npc["lane"] and
                    abs(self.ego["y"] - npc["y"]) < 2.5):
                collision = True
                break

        if collision:
            reward -= 20.0
            self.collision_count += 1

        # Lane keeping reward
        reward += 0.5

        # Speed reward (encourage faster driving)
        reward += self.ego["speed"] * 0.1

        # Penalty for lane changes (smoothness)
        if action in [3, 4]:
            reward -= 0.3

        # Respawn NPCs that are too far behind
        for npc in self.npcs:
            if self.ego["y"] - npc["y"] > 50:
                npc["lane"] = int(self.np_random.integers(0, self.NUM_LANES))
                npc["y"] = self.ego["y"] + self.np_random.uniform(30, 60)
                npc["speed"] = self.np_random.uniform(2.0, 5.0)
                npc["x"] = self._lane_to_x(npc["lane"])

        self.total_reward += reward

        terminated = collision
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "overtakes": self.overtake_count,
            "collisions": self.collision_count,
            "distance": self.ego["y"],
            "speed": self.ego["speed"],
        }

    def _get_obs(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = self.ego["lane"] / (self.NUM_LANES - 1)
        obs[1] = self.ego["speed"] / self.MAX_SPEED
        obs[2] = 0.0  # ego is reference point

        # Find 4 nearest NPCs
        dists = []
        for i, npc in enumerate(self.npcs):
            d = abs(npc["y"] - self.ego["y"])
            dists.append((d, i))
        dists.sort()

        for idx, (_, npc_i) in enumerate(dists[:4]):
            npc = self.npcs[npc_i]
            base = 3 + idx * 3
            obs[base] = (npc["lane"] - self.ego["lane"]) / (self.NUM_LANES - 1)
            obs[base + 1] = (npc["y"] - self.ego["y"]) / self.VISION_RANGE
            obs[base + 2] = npc["speed"] / self.MAX_SPEED

        return obs

    def _lane_to_x(self, lane):
        return (lane + 0.5) * self.LANE_WIDTH

    def get_state_for_render(self):
        return {
            "ego": {
                "lane": self.ego["lane"],
                "x": self.ego["x"],
                "y": self.ego["y"],
                "speed": self.ego["speed"],
            },
            "npcs": [
                {"lane": n["lane"], "x": n["x"], "y": n["y"], "speed": n["speed"]}
                for n in self.npcs
            ],
            "step": self.steps,
            "total_reward": self.total_reward,
            "overtakes": self.overtake_count,
            "collisions": self.collision_count,
        }
