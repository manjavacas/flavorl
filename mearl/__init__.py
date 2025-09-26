from gymnasium.envs.registration import register

register(
    id="mearl/MealRecPlus-v0",
    entry_point="mearl.envs:MealRecPlus",
)
