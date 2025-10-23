import numpy as np

import gymnasium as gym
from gymnasium import spaces

from flavorl.dataclasses import User, Meal, UserDataset, MealDataset, MealType, Day

# --- TODO: determine dimensions ---
OBS_SPACE_DIM = 10  # n_features
ACTION_SPACE_DIM = 5  # n_meals
MAX_EPISODE_STEPS = 21  # 3 meals x 7 days


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
        self.render_mode: str = render_mode

        self.user_dataset: UserDataset = UserDataset(user_csv)
        self.meal_dataset: MealDataset = MealDataset(meal_csv)

        self.current_user: User = None
        self.current_meal: Meal = None
        self.current_day: Day = None
        self.current_mealtype: MealType = None
        self.current_step: int = 0
        self.current_obs: dict = None

        # --- TODO: confirm obs data ---
        self.observation_space = spaces.Dict(
            {
                "day": spaces.Discrete(7),
                "meal_type": spaces.Discrete(3),
                "rem_cal": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "rem_prot": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "rem_ch": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "rem_fib": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                # --- TODO: complete ... ---
                "user_vegan": spaces.Discrete(2),
                "user vegetarian": spaces.Discrete(2),
            }
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
                - info (dict): Additional information.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_day = Day.MONDAY
        self.current_mealtype = MealType.BREAKFAST

        # Sample new user
        self.current_user = self.user_dataset.sample()

        # Initialize observation dictionary
        # --- TODO: confirm obs data ---
        self.current_obs = {
            "day": self.current_day.value,
            "meal_type": self.current_mealtype.value,
            "rem_cal": self.current_user.daily_cal,
            "rem_prot": self.current_user.daily_nutr["protein"],
            "rem_ch": self.current_user.daily_nutr["ch"],
            "rem_fib": self.current_user.daily_nutr["fib"],
            # --- TODO: complete ... ---
            "user_vegan": self.current_user.vegan,
            "user vegetarian": self.current_user.vegetarian,
        }

        info = {
            "step": self.current_step,
            "day": self.current_day.name,
            "meal_type": self.current_mealtype.name,
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
                - info (dict): Additional info.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.current_obs = self._get_next_observation(action)

        terminated = self._check_termination()
        truncated = self.current_step >= MAX_EPISODE_STEPS

        reward = self._compute_reward()

        self.current_step += 1

        info = {
            "step": self.current_step,
            "day": self.current_day.name,
            "meal_type": self.current_mealtype.name,
            "terminated": terminated,
            "truncated": truncated,
        }

        return self.current_obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Renders the environment.
        """
        # --- TODO: implement rendering ---
        raise NotImplementedError

    def close(self) -> None:
        """
        Closes the environment and cleans up resources.
        """
        # --- TODO: implement env closing ---
        print("Closing environment...")

    def _get_next_observation(self, action) -> dict:
        """
        Returns the next observation.

        Returns:
            np.array: next observation.
        """

        obs = self.current_obs.deepcopy()

        # Update obs time variables
        obs["day"] = Day((self.current_day.value + 1) % len(Day))
        obs["meal_type"] = MealType((self.current_mealtype.value + 1) % len(MealType))

        # Update remaining calories / nutrients
        self.current_meal = self._get_dataset_meal(action)

        obs["rem_cal"] -= self.current_meal.calories
        obs["rem_prot"] -= self.current_meal.nutrients["prot"]
        obs["rem_ch"] -= self.current_meal.nutrients["ch"]
        obs["rem_fib"] -= self.current_meal.nutrients["fib"]

        # --- TODO: complete obs ---
        # ...

        return obs

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

    def _get_dataset_meal(self, meal_idx: int) -> Meal:
        """
        Looks for a given meal in the MealDataset

        Args:
            meal_idx (int): meal index.

        Return:
            Meal: corresponding meal.
        """

        # --- TODO: implement meal search ---
        raise NotImplementedError
