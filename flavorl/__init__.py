from gymnasium.envs.registration import register

register(
    id="flavorl/MealRec-v0",
    entry_point="flavorl.envs:MealRec",
)
