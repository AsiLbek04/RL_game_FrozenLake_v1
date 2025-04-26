import gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
episodes = 1000

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state] + np.random.randn(1, env.action_space.n) * (1 / (episode + 1)))
        new_state, reward, done, _, _ = env.step(action)
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])
        state = new_state
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Training finished.")

state = env.reset()[0]
env.render()
for _ in range(1000):
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break
    