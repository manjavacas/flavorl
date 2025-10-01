import gymnasium as gym
from gymnasium import spaces

import numpy as np


class MealRec(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, obs_dim, act_dim, max_episode_steps = 21, render_mode=None):
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(act_dim)

        self.render_mode = render_mode

        self.current_obs = None
        self.current_step = 0
        self.max_episode_steps = max_episode_steps


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # TODO: Modify obs, info
        self.current_obs = self.observation_space.sample()
        info = {"step": self.current_step, "observation": self.current_obs}

        return self.current_obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.current_step += 1

        # TODO: Simulate interaction...
        # ...
        self.current_obs = self.observation_space.sample()
        # ...

        # TODO: Compute reward
        reward = 0.0

        # TODO: Truncation condition
        truncated = self.current_step >= self.max_episode_steps

        # TODO: Modify info
        info = {"step": self.current_step, "truncated": truncated}

        return self.current_obs, reward, False, truncated, info

    def render(self):
        # TODO: implement render function
        raise NotImplementedError

    def close(self):
        # TODO: improve env closing
        print("Closing environment...")
