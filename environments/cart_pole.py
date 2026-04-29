import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()