import gymnasium as gym
import flavorl


def run_mealrec_env():
    """
    Run MealRec Gymnasium environment
    """

    # Create env
    env = gym.make("flavorl/MealRec-v0", render_mode="human", obs_dim=20, act_dim=10)

    print(f"Environment: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Run test episode
    obs, _ = env.reset(seed=123)
    terminated = truncated = False
    step = 0

    print(f"Initial observation: {obs[:3]}")

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step += 1
        print(f"Step {step}: action={action}, reward={reward:.3f}, info={info}")

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    run_mealrec_env()
