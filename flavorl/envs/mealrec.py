import numpy as np

import gymnasium as gym
from gymnasium import spaces

from flavorl.mealrec_data import UserDataset, MealDataset

# --- TODO: determine dimensions ---
OBS_SPACE_DIM = 10
ACTION_SPACE_DIM = 5
MAX_EPISODE_STEPS = 21

DAYS_DICT = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
MEALS_DICT = {0: "breakfast", 1: "lunch", 2: "dinner"}


class MealRec(gym.Env):
    """
    Gym environment for a meal recommendation system.

    Simulates recommending meals to users over multiple days and meal times.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, user_csv: str, meal_csv: str, render_mode: str = None):
        """
        Initializes the MealRec environment.

        Args:
            user_csv (str): Path to the CSV file containing user data.
            meal_csv (str): Path to the CSV file containing meal data.
            render_mode (str, optional): Render mode. Defaults to None.
        """
        self.render_mode = render_mode

        self.user_dataset = UserDataset(user_csv)
        self.meal_dataset = MealDataset(meal_csv)

        self.current_user = None
        self.current_obs = None
        self.current_day = None
        self.current_meal = None
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBS_SPACE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_DIM)

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the environment to start a new episode.

        Samples a new user and initializes observation, day, and meal indices.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for reset. Defaults to None.

        Returns:
            tuple:
                - observation (dict): Initial observation of the environment.
                - info (dict): Additional information including step, day, and meal.
        """
        super().reset(seed=seed)

        self.current_step = 0

        # Sample new user
        self.current_user = self.user_dataset.sample()

        # Initialize observation dictionary
        # --- TODO: define obs data ---
        self.current_obs = {
            "day": 0,
            "meal": 0,
            "nutrient_X": 0,
            "nutrient_Y": 0,
            "nutrient_Z": 0,
            "preference_X": 0,
            "preference_Y": 0,
        }

        self.current_day = 0
        self.current_meal = 0

        info = {
            "step": self.current_step,
            "day": DAYS_DICT[self.current_day],
            "meal": MEALS_DICT[self.current_meal],
        }

        return self.current_obs, info

    def step(self, action: int):
        """
        Takes an action in the environment and returns the next state.

        Args:
            action (int): Action chosen by the agent.

        Returns:
            tuple:
                - observation (dict): Next observation after taking the action.
                - reward (float): Reward obtained from the action.
                - terminated (bool): True if the episode has terminated.
                - truncated (bool): True if the episode was truncated due to max steps.
                - info (dict): Additional info including step, day, meal, and termination flags.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- TODO: simulate interaction ---
        # --- TODO: define obs data ---
        self.current_obs = self.observation_space.sample()

        terminated = self._check_termination()
        truncated = self.current_step >= MAX_EPISODE_STEPS

        reward = self._compute_reward()

        self.current_step += 1
        self.current_day = (self.current_day + 1) % len(DAYS_DICT)
        self.current_meal = (self.current_meal + 1) % len(MEALS_DICT)

        info = {
            "step": self.current_step,
            "day": DAYS_DICT[self.current_day],
            "meal": MEALS_DICT[self.current_meal],
            "terminated": terminated,
            "truncated": truncated,
        }

        return self.current_obs, reward, terminated, truncated, info

    def render(self):
        """
        Renders the environment.
        """
        # --- TODO: implement rendering ---
        raise NotImplementedError

    def close(self):
        """
        Closes the environment and cleans up resources.
        """
        # --- TODO: implement env closing ---
        print("Closing environment...")

    def _check_termination(self) -> bool:
        """
        Checks if the episode should terminate.

        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        # --- TODO: implement termination logic ---
        return False

    def _compute_reward(self) -> float:
        """
        Computes the reward for the current step.

        Returns:
            float: Reward value.
        """
        # --- TODO: implement reward computation ---
        return 0.0
