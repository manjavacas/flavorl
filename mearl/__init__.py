from gymnasium.envs.registration import register

register(
    id="mearl/GridWorld-v0",
    entry_point="mearl.envs:GridWorldEnv",
)
