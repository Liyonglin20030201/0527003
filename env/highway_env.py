"""
Autonomous driving environment built on top of gymnasium CarRacing.

Wraps CarRacing-v2 with:
- Discrete action space (5 actions) for DQN compatibility
- Style-based reward shaping (aggressive / conservative / balanced)
- get_state_for_render() interface for frontend visualization
"""

import base64
import io

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image


class CarRacingDrivingEnv(gym.Env):
    """
    Wraps gymnasium CarRacing-v2 to build a driving scenario with
    style-dependent reward shaping for lane keeping and overtaking behavior.

    CarRacing discrete actions:
        0 = do nothing
        1 = steer left
        2 = steer right
        3 = gas (accelerate)
        4 = brake

    Observation: (96, 96, 3) RGB image from CarRacing top-down view.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    ACTION_NAMES = ["keep", "left", "right", "accelerate", "brake"]

    def __init__(self, render_mode=None, style="balanced"):
        super().__init__()
        self.style = style
        self.render_mode = render_mode

        self._base_env = gym.make(
            "CarRacing-v2",
            continuous=False,
            render_mode="rgb_array",
        )

        # Discrete(5): 0=nothing, 1=left, 2=right, 3=gas, 4=brake
        self.action_space = self._base_env.action_space
        # (96, 96, 3) uint8 image
        self.observation_space = self._base_env.observation_space

        self.steps = 0
        self.max_steps = 1000
        self.total_reward = 0.0
        self.tiles_visited = 0
        self.off_track_count = 0
        self.speed = 0.0
        self.last_frame = None
        self._consecutive_off_track = 0

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0.0
        self.tiles_visited = 0
        self.off_track_count = 0
        self.speed = 0.0
        self._consecutive_off_track = 0

        obs, info = self._base_env.reset(seed=seed, options=options)
        self.last_frame = obs
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self._base_env.step(action)
        self.steps += 1
        self.last_frame = obs

        # Extract speed from car physics
        car = self._base_env.unwrapped.car
        if car is not None:
            vel = car.hull.linearVelocity
            self.speed = float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))
        else:
            self.speed = 0.0

        # Detect whether car is on track by checking pixel colors
        on_track = self._check_on_track(obs)

        if not on_track:
            self._consecutive_off_track += 1
            if self._consecutive_off_track == 1:
                self.off_track_count += 1
        else:
            self._consecutive_off_track = 0

        # Track tile progress (CarRacing gives positive reward per new tile)
        if base_reward > 0:
            self.tiles_visited += 1

        # Apply style-based reward shaping
        reward = self._shape_reward(base_reward, on_track, action)
        self.total_reward += reward

        # Early termination if stuck off-track too long
        if self._consecutive_off_track > 50:
            terminated = True

        truncated = truncated or self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {
            "tiles_visited": self.tiles_visited,
            "off_track_count": self.off_track_count,
            "speed": self.speed,
            "distance": self.tiles_visited,
            "overtakes": self.tiles_visited,
            "collisions": self.off_track_count,
        }

    def _check_on_track(self, obs):
        """Check if car is on the road by sampling pixel colors near the car."""
        # The car is at the bottom-center of the 96x96 frame
        # Road is grey/dark, grass is green
        region = obs[60:80, 38:58]
        green = region[:, :, 1].astype(float)
        red = region[:, :, 0].astype(float)
        # Grass has much higher green than red
        grass_ratio = np.mean(green) / (np.mean(red) + 1e-6)
        return grass_ratio < 1.4

    def _shape_reward(self, base_reward, on_track, action):
        """Apply style-dependent reward shaping on top of CarRacing base reward."""
        if self.style == "aggressive":
            # Aggressive: amplify tile rewards, big speed bonus, tolerate off-track
            reward = base_reward * 1.5
            reward += self.speed * 0.04
            if not on_track:
                reward -= 0.5
            # Bonus for gas pedal
            if action == 3:
                reward += 0.2

        elif self.style == "conservative":
            # Conservative: penalize off-track heavily, reward on-track stability
            reward = base_reward * 0.8
            if not on_track:
                reward -= 3.0
            if on_track:
                reward += 0.5
            # Penalize aggressive steering
            if action in [1, 2]:
                reward -= 0.1
            # Penalize braking less (safety)
            if action == 4:
                reward += 0.1

        else:
            # Balanced: moderate shaping
            reward = base_reward
            reward += self.speed * 0.01
            if not on_track:
                reward -= 1.5
            if on_track:
                reward += 0.2

        return reward

    def get_state_for_render(self):
        """Return state dict compatible with the frontend/backend interface."""
        frame_b64 = self._frame_to_base64(self.last_frame)
        return {
            "frame": frame_b64,
            "speed": float(self.speed),
            "step": self.steps,
            "total_reward": float(self.total_reward),
            "tiles_visited": self.tiles_visited,
            "off_track_count": self.off_track_count,
            "overtakes": self.tiles_visited,
            "collisions": self.off_track_count,
        }

    def _frame_to_base64(self, frame):
        """Encode observation frame as base64 JPEG for frontend display."""
        if frame is None:
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def close(self):
        self._base_env.close()
