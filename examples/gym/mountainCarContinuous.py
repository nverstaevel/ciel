# run pip install gymnasium[classic-control]
import gymnasium as gym
env = gym.make('MountainCarContinuous-v0',render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)
for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    """print("-- STEP --")
    print("observation", observation)
    print("reward", reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("info", info)"""


    if terminated or truncated:
        print("Ended with",_,"cycles")
        observation, info = env.reset()
env.close()
